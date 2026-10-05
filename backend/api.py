"""Public interface of the backend component.

Start the other components first: `storage.api.init_storage()` and
`db.api.init_db()`. Then load the ai engine and make the backend. The start-up
order and the shutdown steps belong to `main.py`, not to this class.
"""

import logging
from pathlib import Path

from fastapi import APIRouter

import db.api as db
import storage.api as storage
from ai.api import AiEngine
from backend.describe import DescribeService
from backend.enrollment import EnrollmentService
from backend.errors import BackendConfigError
from backend.http import create_router, read_token
from backend.http_objects import create_objects_router
from backend.http_search import create_search_router
from backend.ingest import IngestService
from backend.maintenance import Maintenance
from backend.query import QueryResolver
from backend.search import SearchService
from backend.settings import (
    DEFAULT_CONFIG_PATH,
    BackendConfig,
    ServerConfig,
    load_config,
)
from backend.ui_adapter import (
    UiEnrollmentService,
    UiFindService,
    UiUploadService,
    UiVoiceService,
)
from backend.voice import VoiceService
from backend.worker import Worker

logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["Backend"]


class Backend:
    """The upload step, the worker and the maintenance jobs, with the HTTP route and the UI service."""

    def __init__(
        self, engine: AiEngine, config_path: Path = DEFAULT_CONFIG_PATH
    ) -> None:
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        assert isinstance(config_path, Path), "config_path must be a Path"
        self._config: BackendConfig = load_config(config_path)

        vector_size: int = db.current_config().vector_dimension
        if engine.embedding_dimension != vector_size:
            raise BackendConfigError(
                f"The ai model makes vectors of size {engine.embedding_dimension}, "
                f"but vectors.dimension in db/config.toml is {vector_size}"
            )

        self._worker: Worker = Worker(self._config, engine)
        # An upload wakes the worker, so a new image starts at once.
        self._ingest: IngestService = IngestService(
            self._config, on_stored=self._worker.wake
        )
        # A new reference photo wakes the worker, as an upload does.
        self._enrollment: EnrollmentService = EnrollmentService(
            self._config, engine, on_photo_added=self._worker.wake
        )
        self._search: SearchService = SearchService(self._config, engine)
        self._query: QueryResolver = QueryResolver(self._config, engine)
        self._voice: VoiceService = VoiceService(self._config, engine)
        self._describe: DescribeService = DescribeService(self._config, engine)
        self._maintenance: Maintenance = Maintenance(self._config)
        self._token: str | None = read_token(self._config)
        self._max_upload_bytes: int = storage.current_config().max_bytes
        logger.info(
            "Backend ready. Upload access token: %s",
            "required" if self._token is not None else "not required",
        )

    @property
    def server(self) -> ServerConfig:
        """The settings of the web server (host, port, shutdown wait)."""
        return self._config.server

    @property
    def max_upload_bytes(self) -> int:
        """The largest upload that the HTTP route reads. It comes from the storage config."""
        return self._max_upload_bytes

    @property
    def worker(self) -> Worker:
        return self._worker

    @property
    def enrollment(self) -> EnrollmentService:
        """The steps of the carer: objects, reference photos, confirmation, calibration."""
        return self._enrollment

    @property
    def search(self) -> SearchService:
        """The question "where is my object?": the last-seen record, then the archive."""
        return self._search

    @property
    def query(self) -> QueryResolver:
        """Turns the words of a user into an object (matched, a choice, or none)."""
        return self._query

    @property
    def maintenance(self) -> Maintenance:
        return self._maintenance

    def start(self) -> None:
        """Start the worker and the maintenance thread."""
        self._worker.start()
        self._maintenance.start()

    def stop(self) -> None:
        """Stop the maintenance and the worker.

        The worker finishes the image that it holds, up to `server.worker_stop_seconds`.
        """
        self._maintenance.stop(timeout=self._config.server.worker_stop_seconds)
        self._worker.stop(timeout=self._config.server.worker_stop_seconds)

    def create_router(self) -> APIRouter:
        """Return the router with all routes (upload, gallery and search).

        Add it with `app.include_router`.
        """
        router: APIRouter = create_router(
            self._ingest, self._max_upload_bytes, self._token
        )
        router.include_router(
            create_objects_router(self._enrollment, self._max_upload_bytes, self._token)
        )
        router.include_router(
            create_search_router(self._query, self._search, self._token)
        )
        return router

    @property
    def upload_service(self) -> UiUploadService:
        """The service that the UI calls. It implements `ui.ports.UploadService`."""
        return UiUploadService(self._ingest)

    @property
    def enrollment_service(self) -> UiEnrollmentService:
        """The gallery service that the UI calls. It implements `ui.ports.EnrollmentService`."""
        return UiEnrollmentService(self._enrollment)

    @property
    def find_service(self) -> UiFindService:
        """The search that the UI calls. It implements `ui.ports.FindService`."""
        return UiFindService(
            self._query, self._search, self._enrollment, self._describe
        )

    @property
    def voice_service(self) -> UiVoiceService | None:
        """The voice input that the UI calls (`ui.ports.VoiceService`). None if listening is switched off."""
        return UiVoiceService(self._voice) if self._voice.enabled else None
