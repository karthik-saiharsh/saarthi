"""Public interface of the backend component.

Start the other components first: `storage.api.init_storage()` and
`db.api.init_db()`. Then make the backend. The start-up and shutdown steps
belong to `main.py`, not to this class.
"""

import logging
from pathlib import Path

from fastapi import APIRouter

import storage.api as storage
from backend.http import create_router, read_token
from backend.ingest import IngestService
from backend.settings import DEFAULT_CONFIG_PATH, BackendConfig, load_config
from backend.ui_adapter import UiUploadService

logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["Backend"]


class Backend:
    """The upload step, as an HTTP route and as a service for the UI."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        assert isinstance(config_path, Path), "config_path must be a Path"
        self._config: BackendConfig = load_config(config_path)
        self._ingest: IngestService = IngestService(self._config)
        self._token: str | None = read_token(self._config)
        self._max_upload_bytes: int = storage.current_config().max_bytes
        logger.info(
            "Backend ready. Upload access token: %s",
            "required" if self._token is not None else "not required",
        )

    @property
    def max_upload_bytes(self) -> int:
        """The largest upload that the HTTP route reads. It comes from the storage config."""
        return self._max_upload_bytes

    def create_router(self) -> APIRouter:
        """Return the router with `POST /api/images`. Add it with `app.include_router`."""
        return create_router(self._ingest, self._max_upload_bytes, self._token)

    @property
    def upload_service(self) -> UiUploadService:
        """The service that the UI calls. It implements `ui.ports.UploadService`."""
        return UiUploadService(self._ingest)
