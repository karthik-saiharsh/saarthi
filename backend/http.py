"""The HTTP route for uploads: `POST /api/images`.

The request body is the image itself (JPEG, PNG or WebP bytes). The
`Content-Type` header is not used. The storage component finds the format
from the content.

Answers (JSON):

| Code | Meaning |
|------|---------|
| 201  | A new image is stored: `{"status": "stored", "image_id": 1}` |
| 200  | The same image was stored before: `{"status": "duplicate", "image_id": 1}` |
| 400  | The body is empty or the headers are wrong |
| 401  | The token is missing or wrong (only if a token is set) |
| 413  | The upload has too many bytes or too many pixels |
| 415  | The data is not a usable JPEG, PNG or WebP image |
| 503  | Too many images wait for the worker. See `Retry-After` |
| 507  | The disk of the server is full |
| 500  | Another error. The message does not show internal details |
"""

import logging
import os

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import db.api as db
import storage.api as storage
from backend.errors import BackendConfigError, BackendError, QueueFullError
from backend.http_common import (
    MESSAGE_DISK_FULL,
    MESSAGE_FAILED,
    MESSAGE_TOO_LARGE,
    MESSAGE_UNSUPPORTED,
    error_response,
    read_body,
    token_refusal,
)
from backend.ingest import IngestResult, IngestService, IngestStatus
from backend.settings import BackendConfig

logger: logging.Logger = logging.getLogger(__name__)

UPLOAD_ROUTE: str = "/api/images"
RETRY_AFTER_SECONDS: int = 30

MESSAGE_BUSY: str = "The server is busy. Try again later."


def read_token(config: BackendConfig) -> str | None:
    """Return the access token from the environment, or None if no token is set.

    A variable that is set but empty is a mistake, not a switch. It raises an
    error, so the endpoint never opens by accident.
    """
    assert isinstance(config, BackendConfig), "config must be a BackendConfig"
    value: str | None = os.environ.get(config.token_env_var)
    if value is None:
        return None
    if not value.strip():
        raise BackendConfigError(f"{config.token_env_var} is set but empty")
    return value


def _answer(result: IngestResult) -> JSONResponse:
    status_code: int = 201 if result.status == IngestStatus.STORED else 200
    return JSONResponse(
        {"status": result.status.value, "image_id": result.image_id},
        status_code=status_code,
    )


def create_router(
    service: IngestService, max_bytes: int, token: str | None = None
) -> APIRouter:
    """Make the router. Add it to the application with `app.include_router`.

    `max_bytes` is the largest body that the route reads. If `token` is set,
    every request needs the header `Authorization: Bearer <token>`.
    """
    assert isinstance(service, IngestService), "service must be an IngestService"
    assert isinstance(max_bytes, int) and max_bytes > 0, "max_bytes must be positive"
    assert token is None or token.strip(), "token must not be empty"
    router: APIRouter = APIRouter()

    @router.post(UPLOAD_ROUTE)
    async def upload(request: Request) -> JSONResponse:
        refusal: JSONResponse | None = token_refusal(request, token)
        if refusal is not None:
            logger.warning("Upload refused: the token is missing or wrong")
            return refusal

        body: bytes | JSONResponse = await read_body(request, max_bytes)
        if isinstance(body, JSONResponse):
            logger.info(
                "Upload refused with status %d before it was read", body.status_code
            )
            return body

        try:
            result: IngestResult = await run_in_threadpool(service.submit, body)
        except QueueFullError:
            return error_response(
                503, MESSAGE_BUSY, {"Retry-After": str(RETRY_AFTER_SECONDS)}
            )
        except storage.ImageTooLargeError:
            return error_response(413, MESSAGE_TOO_LARGE)
        except storage.UnsupportedImageError:
            return error_response(415, MESSAGE_UNSUPPORTED)
        except storage.StorageFullError:
            logger.error("Upload failed: the disk is full")
            return error_response(507, MESSAGE_DISK_FULL)
        except (storage.StorageError, db.DbError, BackendError):
            logger.exception("Upload failed")
            return error_response(500, MESSAGE_FAILED)
        except (
            Exception
        ):  # The boundary of the server: any other error becomes a plain 500.
            logger.exception("Upload failed with an unexpected error")
            return error_response(500, MESSAGE_FAILED)
        return _answer(result)

    return router
