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

import hmac
import logging
import os

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import db.api as db
import storage.api as storage
from backend.errors import BackendConfigError, BackendError, QueueFullError
from backend.ingest import IngestResult, IngestService, IngestStatus
from backend.settings import BackendConfig

logger: logging.Logger = logging.getLogger(__name__)

UPLOAD_ROUTE: str = "/api/images"
RETRY_AFTER_SECONDS: int = 30

# Short messages for the client. They never contain internal details.
MESSAGE_EMPTY: str = "The request body is empty."
MESSAGE_BAD_LENGTH: str = "The Content-Length header is not valid."
MESSAGE_TOKEN: str = "A valid access token is required."
MESSAGE_TOO_LARGE: str = "The upload is too large."
MESSAGE_UNSUPPORTED: str = "The data is not a usable JPEG, PNG or WebP image."
MESSAGE_BUSY: str = "The server is busy. Try again later."
MESSAGE_DISK_FULL: str = "The disk of the server is full."
MESSAGE_FAILED: str = "The server could not store the image."


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


def _error(
    status_code: int, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code, headers=headers)


def _token_is_valid(request: Request, token: str) -> bool:
    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return False
    # Compare in constant time, so the answer time does not show how many characters match.
    return hmac.compare_digest(supplied.strip().encode(), token.encode())


async def _read_body(request: Request, max_bytes: int) -> bytes | JSONResponse:
    """Read the body in chunks. Stop at `max_bytes`, so a big upload never fills the memory."""
    declared: str | None = request.headers.get("content-length")
    if declared is not None:
        if not declared.isdigit():
            return _error(400, MESSAGE_BAD_LENGTH)
        if int(declared) > max_bytes:
            return _error(413, MESSAGE_TOO_LARGE)
    body: bytearray = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            return _error(413, MESSAGE_TOO_LARGE)
    if not body:
        return _error(400, MESSAGE_EMPTY)
    return bytes(body)


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
        if token is not None and not _token_is_valid(request, token):
            logger.warning("Upload refused: the token is missing or wrong")
            return _error(401, MESSAGE_TOKEN, {"WWW-Authenticate": "Bearer"})

        body: bytes | JSONResponse = await _read_body(request, max_bytes)
        if isinstance(body, JSONResponse):
            logger.info(
                "Upload refused with status %d before it was read", body.status_code
            )
            return body

        try:
            result: IngestResult = await run_in_threadpool(service.submit, body)
        except QueueFullError:
            return _error(503, MESSAGE_BUSY, {"Retry-After": str(RETRY_AFTER_SECONDS)})
        except storage.ImageTooLargeError:
            return _error(413, MESSAGE_TOO_LARGE)
        except storage.UnsupportedImageError:
            return _error(415, MESSAGE_UNSUPPORTED)
        except storage.StorageFullError:
            logger.error("Upload failed: the disk is full")
            return _error(507, MESSAGE_DISK_FULL)
        except (storage.StorageError, db.DbError, BackendError):
            logger.exception("Upload failed")
            return _error(500, MESSAGE_FAILED)
        except (
            Exception
        ):  # The boundary of the server: any other error becomes a plain 500.
            logger.exception("Upload failed with an unexpected error")
            return _error(500, MESSAGE_FAILED)
        return _answer(result)

    return router
