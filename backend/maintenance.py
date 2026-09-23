"""Maintenance jobs: retention, orphan files, missing files and the vector checkpoint.

`Maintenance.run_once` does the jobs one time. `start` runs it in a background
thread: first at once, then every `maintenance.interval_minutes`.

The jobs, in this order:

1. Retention. If `retention.days` in `storage/config.toml` is more than 0, delete the
   images that are older. The rows and the vectors go first (db), the files last (storage).
   A stop in between leaves orphan files, and the orphan job removes them.
2. Orphan files. Delete files that no database row owns, at most
   `maintenance.orphan_delete_limit` in one run. A larger number is only reported.
3. Missing files. Report the rows whose file does not exist. Nothing is changed.
4. Checkpoint. Save the vector index to disk.

An error in one job does not stop the other jobs.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import timedelta

import db.api as db
import storage.api as storage
from backend.settings import BackendConfig, MaintenanceConfig
from db.schema import Image, utc_now

logger: logging.Logger = logging.getLogger(__name__)

# A safety stop for the retention loop. Each loop deletes at least one image.
MAX_RETENTION_BATCHES: int = 10_000
# How many paths a warning lists.
LISTED_PATHS: int = 5


@dataclass(frozen=True)
class MaintenanceReport:
    """What one run did."""

    expired_images_deleted: int
    orphan_files_found: int
    orphan_files_deleted: int
    missing_files: int
    checkpoint_saved: bool


class Maintenance:
    """Runs the maintenance jobs. Use `start` for a background thread."""

    def __init__(self, config: BackendConfig) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        self._config: MaintenanceConfig = config.maintenance
        self._stop: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_report: MaintenanceReport | None = None

    @property
    def last_report(self) -> MaintenanceReport | None:
        """The report of the last run, or None if no run is finished."""
        return self._last_report

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- Control of the thread ----

    def start(self) -> None:
        """Start the background thread. The first run starts at once."""
        if self.is_running:
            raise RuntimeError("Maintenance is already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="saarthi-maintenance", daemon=True
        )
        self._thread.start()
        logger.info(
            "Maintenance started. It runs every %d minute(s)",
            self._config.interval_minutes,
        )

    def stop(self, timeout: float | None = None) -> None:
        """Ask the thread to stop and wait for it. A run that is in progress finishes first."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("Maintenance did not stop within %s seconds", timeout)
                return
        logger.info("Maintenance stopped")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self.run_once()
                except Exception:  # The loop must never die.
                    logger.exception("Unexpected error in the maintenance run")
                self._stop.wait(self._config.interval_minutes * 60)
        finally:
            db.close_connection()

    # ---- The jobs ----

    def run_once(self) -> MaintenanceReport:
        """Do all jobs one time. An error in one job is logged and does not stop the others."""
        expired: int = self._delete_expired_images()
        found, deleted, missing = self._check_files()
        saved: bool = self._save_vectors()
        report: MaintenanceReport = MaintenanceReport(
            expired, found, deleted, missing, saved
        )
        self._last_report = report
        logger.info(
            "Maintenance: %d expired image(s) deleted, %d orphan file(s) found and %d deleted, "
            "%d missing file(s), vectors saved: %s",
            expired,
            found,
            deleted,
            missing,
            saved,
        )
        return report

    def _delete_expired_images(self) -> int:
        """Retention. Return the number of images that were deleted."""
        days: int = storage.current_config().retention_days
        if days == 0:
            return 0  # The owner keeps all images
        cutoff = utc_now() - timedelta(days=days)
        total: int = 0
        try:
            for _ in range(MAX_RETENTION_BATCHES):
                expired: list[Image] = db.list_images_older_than(
                    cutoff, self._config.retention_batch
                )
                if not expired:
                    break
                # Rows and vectors first, files last. See the module text.
                paths: list[str] = db.delete_images([image.id for image in expired])
                total += len(expired)
                report: storage.DeleteReport = storage.delete_images(paths)
                if report.failed:
                    logger.warning(
                        "%d expired file(s) could not be deleted", len(report.failed)
                    )
        except (storage.StorageError, db.DbError):
            logger.exception("The retention job failed")
        if total:
            logger.info(
                "Retention: deleted %d image(s) older than %d day(s)", total, days
            )
        return total

    def _check_files(self) -> tuple[int, int, int]:
        """Orphan and missing files. Return (orphans found, orphans deleted, missing files)."""
        try:
            known: list[str] = db.list_image_paths()
            orphans: list[str] = storage.find_orphan_files(known)
            missing: list[str] = storage.find_missing_files(known)
        except (storage.StorageError, db.DbError):
            logger.exception("The file check failed")
            return 0, 0, 0

        if missing:
            logger.warning(
                "%d image(s) have no file on disk, for example: %s",
                len(missing),
                ", ".join(missing[:LISTED_PATHS]),
            )
        limit: int = self._config.orphan_delete_limit
        deleted: int = 0
        if orphans:
            logger.warning(
                "%d orphan file(s) found (no database row owns them)", len(orphans)
            )
            if len(orphans) > limit:
                logger.warning(
                    "More orphan files than the limit of %d. Only %d are deleted. "
                    "Check that the database is the right one.",
                    limit,
                    limit,
                )
            try:
                report: storage.DeleteReport = storage.delete_images(orphans[:limit])
                deleted = len(report.deleted)
            except storage.StorageError:
                logger.exception("The orphan job failed")
        return len(orphans), deleted, len(missing)

    def _save_vectors(self) -> bool:
        """Save the vector index. Return True if the save worked."""
        try:
            db.checkpoint_vectors()
        except db.DbError:  # VectorIndexError derives from DbError.
            logger.exception("The vector index could not be saved")
            return False
        return True
