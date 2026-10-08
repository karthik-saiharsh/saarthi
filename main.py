"""Entry point of Saarthi. It owns the life of the application.

Start-up order (each step needs the one before it):

1. Storage: read the config, prepare the root folder.
2. Database and vector index.
3. Instance lock: only one copy may use the data. A second copy stops here.
4. Start-up recovery: repair what a stop or a crash left behind.
5. The ai models (the first start downloads them).
6. The backend: the upload step, the worker and the maintenance jobs.
7. The web server: the upload page `/upload`, the upload route `/api/images`.
   The worker and the maintenance thread start when the server starts.

Shutdown: the worker finishes its image, the maintenance thread stops, the
vector index is saved to disk, the database connection is closed, and the
instance lock is released.

Run it with `uv run python main.py`.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Protocol

from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from nicegui import app as nicegui_app
from nicegui import ui

import db.api as db
import storage.api as storage
from ai.api import AiEngine, AiError
from backend.api import Backend
from backend.errors import BackendError
from backend.instance_lock import InstanceLock
from ui.registry import register_pages
from ui.routes import UPLOAD_ROUTE as UPLOAD_PAGE

logger: logging.Logger = logging.getLogger("main")
if not logger.handlers:
    _handler: logging.StreamHandler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s-%(asctime)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

APP_TITLE: str = "Saarthi"


class WebApp(Protocol):
    """The part of the NiceGUI `app` that `attach` uses. A test can give a stand-in."""

    def include_router(self, router: APIRouter) -> None: ...

    def on_startup(self, handler: object) -> None: ...

    def on_shutdown(self, handler: object) -> None: ...


@dataclass
class Components:
    """The parts of the running application."""

    backend: Backend
    recovery: db.StartupReport
    lock: InstanceLock


def start_components(engine: AiEngine | None = None) -> Components:
    """Start storage, database, the lock, recovery, the ai models and the backend, in this order.

    The threads do not start here. They start with the web server (see `attach`).
    Pass `engine` only in tests, to avoid loading the real models.
    Raise `InstanceRunningError` if another copy of the application uses the data.
    """
    logger.info("Starting storage")
    storage.init_storage()
    logger.info("Starting the database and the vector index")
    db.init_db()
    # The lock comes before the recovery. The recovery would reset the images
    # that another running copy is working on.
    lock: InstanceLock = InstanceLock(
        db.current_config().sqlite_path.with_suffix(".lock")
    )
    lock.acquire()
    try:
        recovery: db.StartupReport = db.startup_recovery()
        if engine is None:
            logger.info("Loading the ai models (the first start downloads them)")
            engine = AiEngine()
        backend: Backend = Backend(engine)
    except BaseException:
        lock.release()
        raise
    logger.info("All components are ready")
    return Components(backend, recovery, lock)


def stop_components(components: Components) -> None:
    """Stop the threads, save the vector index and close the database connection."""
    logger.info("Shutting down")
    components.backend.stop()
    try:
        db.checkpoint_vectors()
    except db.DbError:  # VectorIndexError derives from DbError.
        logger.exception("The vector index could not be saved at shutdown")
    db.close_connection()
    components.lock.release()
    logger.info("Shutdown is complete")


def attach(components: Components, web_app: WebApp) -> None:
    """Add the upload route to the web app and connect the threads to its life."""
    web_app.include_router(components.backend.create_router())
    web_app.on_startup(components.backend.start)
    web_app.on_shutdown(lambda: stop_components(components))


def create_pages(components: Components) -> None:
    """Add the pages of the UI. `/` leads to the upload page."""
    register_pages(
        components.backend.upload_service,
        components.backend.enrollment_service,
        components.backend.find_service,
        components.backend.voice_service,
        components.backend.browse_service,
    )

    @ui.page("/")
    def home() -> RedirectResponse:
        return RedirectResponse(UPLOAD_PAGE)


def main() -> int:
    """Start the application. Return the exit code."""
    try:
        components: Components = start_components()
    except (storage.StorageError, db.DbError, AiError, BackendError) as error:
        logger.error("Cannot start: %s: %s", type(error).__name__, error)
        return 1
    attach(components, nicegui_app)
    create_pages(components)
    server = components.backend.server
    ssl_options: dict[str, str] = {}
    if server.tls_enabled:
        try:
            certfile, keyfile = server.tls_paths()
        except BackendError as error:
            logger.error("Cannot start: %s", error)
            return 1
        ssl_options = {"ssl_certfile": str(certfile), "ssl_keyfile": str(keyfile)}
    logger.info(
        "Serving on %s://%s:%d (upload page: %s)",
        "https" if server.tls_enabled else "http",
        server.host,
        server.port,
        UPLOAD_PAGE,
    )
    ui.run(
        host=server.host,
        port=server.port,
        title=APP_TITLE,
        reload=False,
        show=False,
        **ssl_options,
    )
    return 0


if __name__ in {"__main__", "__mp_main__"}:
    sys.exit(main())
