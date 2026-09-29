"""The HTTP routes of the gallery: objects and their reference photos.

| Route | Does |
|-------|------|
| `POST /api/objects` | Create an object. The body is JSON: `{"name": "...", "category": "a bottle", "synonyms": [...], "tracked": true}` |
| `GET /api/objects` | List the objects |
| `GET /api/objects/{id}` | One object with its photos |
| `DELETE /api/objects/{id}` | Delete an object, its photos and files |
| `POST /api/objects/{id}/photos` | Add a reference photo. The body is the image (as for `POST /api/images`) |
| `GET /api/objects/{id}/photos/{photo}/crop` | A JPEG of the crop where the object was found. `?max_side=` from 32 to 1024 |
| `POST /api/objects/{id}/photos/{photo}/confirm` | The carer accepts the crop |
| `POST /api/objects/{id}/photos/{photo}/retry` | Try a failed photo again |
| `DELETE /api/objects/{id}/photos/{photo}` | Remove a photo (a wrong crop) |

Answers are JSON. An error is `{"error": "<fixed message>"}`: 400 (not valid), 401 (token),
404 (not found), 409 (name or photo exists, wrong state, limit), 413, 415, 507, 500.
The messages never show internal details. The routes use the same optional token as the upload route.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import db.api as db
import storage.api as storage
from ai.api import AiError
from backend.enrollment import EnrollmentService
from backend.errors import (
    BackendError,
    InvalidInputError,
    InvalidQueryError,
    LimitReachedError,
    NotFoundError,
    ObjectNotReadyError,
)
from backend.http_common import (
    MESSAGE_DISK_FULL,
    MESSAGE_FAILED,
    MESSAGE_TOO_LARGE,
    MESSAGE_UNSUPPORTED,
    error_response,
    read_body,
    token_refusal,
)

logger: logging.Logger = logging.getLogger(__name__)

OBJECTS_ROUTE: str = "/api/objects"
MAX_JSON_BYTES: int = 16 * 1024
MAX_SYNONYMS_IN_REQUEST: int = 50
MIN_CROP_SIDE: int = 32
MAX_CROP_SIDE: int = 1024
DEFAULT_CROP_SIDE: int = 512

MESSAGE_BODY: str = "The request body is not valid."
MESSAGE_QUERY: str = (
    f"max_side must be a number from {MIN_CROP_SIDE} to {MAX_CROP_SIDE}."
)
MESSAGE_NOT_FOUND: str = "Not found."
MESSAGE_NAME_EXISTS: str = "An object with this name exists."
MESSAGE_PHOTO_EXISTS: str = "This photo is already added to the object."
MESSAGE_WRONG_STATE: str = (
    "The photo or the object is in the wrong state for this action."
)
MESSAGE_FILE_MISSING: str = "The file of the photo is missing."
MESSAGE_NOT_READY: str = "This object cannot be searched yet."


def parse_id(text: str) -> int | None:
    """Return a positive id from a path part, or None."""
    return (
        int(text)
        if text.isascii() and text.isdigit() and 0 < int(text) < 2**31
        else None
    )


def _parse_object_body(body: bytes) -> dict[str, Any] | None:
    """Check the JSON body of a new object. Return the arguments for the service, or None."""
    try:
        data: object = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    name, category = data.get("name"), data.get("category")
    synonyms, tracked = data.get("synonyms", []), data.get("tracked", True)
    if (
        not isinstance(name, str)
        or not isinstance(category, str)
        or not isinstance(tracked, bool)
    ):
        return None
    if not isinstance(synonyms, list) or len(synonyms) > MAX_SYNONYMS_IN_REQUEST:
        return None
    if not all(isinstance(s, str) for s in synonyms):
        return None
    return {
        "name": name,
        "category_phrase": category,
        "synonyms": synonyms,
        "tracked": tracked,
    }


def map_error(error: Exception) -> JSONResponse:
    """Turn an error of the service into an answer with a fixed message."""
    if isinstance(error, InvalidInputError):
        return error_response(400, str(error))  # The service messages are fixed texts
    if isinstance(error, InvalidQueryError):
        return error_response(400, str(error))  # A fixed text
    if isinstance(error, ObjectNotReadyError):
        return error_response(409, MESSAGE_NOT_READY)
    if isinstance(error, NotFoundError):
        return error_response(404, MESSAGE_NOT_FOUND)
    if isinstance(error, LimitReachedError):
        return error_response(409, str(error))
    if isinstance(error, db.DuplicateObjectError):
        return error_response(409, MESSAGE_NAME_EXISTS)
    if isinstance(error, db.DuplicateImageError):
        return error_response(409, MESSAGE_PHOTO_EXISTS)
    if isinstance(error, db.InvalidStateError):
        return error_response(409, MESSAGE_WRONG_STATE)
    if isinstance(error, storage.ImageTooLargeError):
        return error_response(413, MESSAGE_TOO_LARGE)
    if isinstance(error, storage.UnsupportedImageError):
        return error_response(415, MESSAGE_UNSUPPORTED)
    if isinstance(error, storage.StorageFullError):
        logger.error("Request failed: the disk is full")
        return error_response(507, MESSAGE_DISK_FULL)
    if isinstance(error, storage.StorageNotFoundError):
        return error_response(404, MESSAGE_FILE_MISSING)
    if isinstance(error, (storage.StorageError, db.DbError, BackendError, AiError)):
        logger.exception("Request failed")
        return error_response(500, MESSAGE_FAILED)
    logger.exception("Request failed with an unexpected error")
    return error_response(500, MESSAGE_FAILED)


def create_objects_router(
    service: EnrollmentService, max_bytes: int, token: str | None = None
) -> APIRouter:
    """Make the router of the gallery. `max_bytes` is the largest photo that a route reads."""
    assert isinstance(
        service, EnrollmentService
    ), "service must be an EnrollmentService"
    assert isinstance(max_bytes, int) and max_bytes > 0, "max_bytes must be positive"
    assert token is None or token.strip(), "token must not be empty"
    router: APIRouter = APIRouter()

    async def run(
        request: Request,
        function: Callable[[], Any],
        success: int = 200,
        build: Callable[[Any], Response] | None = None,
    ) -> Response:
        """Check the token, run `function` in a thread, and make the answer.

        The answer is JSON, or `build(result)` if `build` is given (for an image).
        """
        refusal: JSONResponse | None = token_refusal(request, token)
        if refusal is not None:
            logger.warning("Request refused: the token is missing or wrong")
            return refusal
        try:
            result: Any = await run_in_threadpool(function)
        # The boundary of the server: each error becomes an answer with a fixed message.
        except Exception as error:  # noqa: BLE001
            return map_error(error)
        if build is not None:
            return build(result)
        if success == 204:
            return Response(status_code=204)
        return JSONResponse(result, status_code=success)

    async def with_ids(
        request: Request,
        object_text: str,
        photo_text: str | None,
        handler: Callable[[int, int | None], Any],
        success: int = 200,
        build: Callable[[Any], Response] | None = None,
    ) -> Response:
        object_id: int | None = parse_id(object_text)
        photo_id: int | None = None if photo_text is None else parse_id(photo_text)
        if object_id is None or (photo_text is not None and photo_id is None):
            refusal = token_refusal(request, token)
            return refusal or error_response(404, MESSAGE_NOT_FOUND)
        return await run(request, lambda: handler(object_id, photo_id), success, build)

    @router.post(OBJECTS_ROUTE)
    async def create_object(request: Request) -> Response:
        refusal: JSONResponse | None = token_refusal(request, token)
        if refusal is not None:
            return refusal
        body: bytes | JSONResponse = await read_body(request, MAX_JSON_BYTES)
        if isinstance(body, JSONResponse):
            return body
        arguments: dict[str, Any] | None = _parse_object_body(body)
        if arguments is None:
            return error_response(400, MESSAGE_BODY)
        return await run(
            request, lambda: asdict(service.create_object(**arguments)), 201
        )

    @router.get(OBJECTS_ROUTE)
    async def list_objects(request: Request) -> Response:
        return await run(request, lambda: [asdict(o) for o in service.list_objects()])

    @router.get(OBJECTS_ROUTE + "/{object_id}")
    async def get_object(request: Request, object_id: str) -> Response:
        return await with_ids(
            request, object_id, None, lambda o, _: asdict(service.get_object(o))
        )

    @router.delete(OBJECTS_ROUTE + "/{object_id}")
    async def delete_object(request: Request, object_id: str) -> Response:
        return await with_ids(
            request, object_id, None, lambda o, _: service.delete_object(o), 204
        )

    @router.post(OBJECTS_ROUTE + "/{object_id}/photos")
    async def add_photo(request: Request, object_id: str) -> Response:
        refusal: JSONResponse | None = token_refusal(request, token)
        if refusal is not None:
            return refusal
        parsed: int | None = parse_id(object_id)
        if parsed is None:
            return error_response(404, MESSAGE_NOT_FOUND)
        body: bytes | JSONResponse = await read_body(request, max_bytes)
        if isinstance(body, JSONResponse):
            return body
        return await run(request, lambda: asdict(service.add_photo(parsed, body)), 201)

    @router.get(OBJECTS_ROUTE + "/{object_id}/photos/{photo_id}/crop")
    async def get_crop(request: Request, object_id: str, photo_id: str) -> Response:
        side_text: str = request.query_params.get("max_side", str(DEFAULT_CROP_SIDE))
        if not (
            side_text.isdigit() and MIN_CROP_SIDE <= int(side_text) <= MAX_CROP_SIDE
        ):
            return token_refusal(request, token) or error_response(400, MESSAGE_QUERY)
        side: int = int(side_text)

        def image_answer(data: bytes) -> Response:
            # A private image: the browser must not keep it, and must not guess another type.
            return Response(
                content=data,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )

        return await with_ids(
            request,
            object_id,
            photo_id,
            lambda o, p: service.get_crop(o, p, side),  # type: ignore[arg-type]
            build=image_answer,
        )

    @router.post(OBJECTS_ROUTE + "/{object_id}/photos/{photo_id}/confirm")
    async def confirm_photo(
        request: Request, object_id: str, photo_id: str
    ) -> Response:
        return await with_ids(
            request, object_id, photo_id,
            lambda o, p: asdict(service.confirm_photo(o, p)),  # type: ignore[arg-type]
        )  # fmt: skip

    @router.post(OBJECTS_ROUTE + "/{object_id}/photos/{photo_id}/retry")
    async def retry_photo(request: Request, object_id: str, photo_id: str) -> Response:
        return await with_ids(
            request, object_id, photo_id,
            lambda o, p: asdict(service.retry_photo(o, p)),  # type: ignore[arg-type]
        )  # fmt: skip

    @router.delete(OBJECTS_ROUTE + "/{object_id}/photos/{photo_id}")
    async def remove_photo(request: Request, object_id: str, photo_id: str) -> Response:
        return await with_ids(
            request, object_id, photo_id,
            lambda o, p: service.remove_photo(o, p),  # type: ignore[arg-type]
            204,
        )  # fmt: skip

    return router
