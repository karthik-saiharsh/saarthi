"""The HTTP routes of the search: from the words of a user to the last sighting of an object.

| Route | Does |
|-------|------|
| `POST /api/search` | The body is JSON: `{"query": "the blue thing I drink from"}`. The answer has `status` (`matched`, `ambiguous` or `none`), the `candidates` and, for `matched`, the `answer` |
| `GET /api/objects/{id}/last-seen` | Where and when the object was seen last: the `answer` |
| `GET /api/objects/{id}/last-seen/picture` | A JPEG of the answer. `?view=crop` (the default) or `frame`, and `?max_side=` from 32 to 1024 |

The `answer` has `found`, `source`, `seen_at` (UTC, ends with `Z`), `age_seconds`, `image_id`, `box`,
`similarity`, `strong` (false means "maybe"), `in_hand`, `stale` and `complete`.
The routes use the same optional token and the same fixed messages as the other routes.
A picture is a private image: the browser must not keep it. The words of a user are never logged.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from backend.http_common import error_response, read_body, token_refusal
from backend.http_objects import (
    DEFAULT_CROP_SIDE,
    MAX_CROP_SIDE,
    MAX_JSON_BYTES,
    MESSAGE_BODY,
    MESSAGE_NOT_FOUND,
    MESSAGE_QUERY,
    MIN_CROP_SIDE,
    OBJECTS_ROUTE,
    map_error,
    parse_id,
)
from backend.query import QueryResolver, QueryResult
from backend.search import PICTURE_VIEWS, Answer, SearchService

logger: logging.Logger = logging.getLogger(__name__)

SEARCH_ROUTE: str = "/api/search"
MESSAGE_VIEW: str = f"view must be one of: {', '.join(PICTURE_VIEWS)}."
# The answers are private: the browser must not keep them.
PRIVATE_HEADERS: dict[str, str] = {"Cache-Control": "no-store"}


def answer_json(answer: Answer) -> dict[str, Any]:
    """Make the JSON form of an answer. The time is UTC, written with a `Z`."""
    data: dict[str, Any] = asdict(answer)
    data["source"] = None if answer.source is None else answer.source.value
    data["seen_at"] = (
        None if answer.seen_at is None else answer.seen_at.isoformat() + "Z"
    )
    data["box"] = None if answer.box is None else list(answer.box)
    return data


def _parse_search_body(body: bytes) -> str | None:
    """Return the words from the JSON body, or None if the body is not valid."""
    try:
        data: object = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("query"), str):
        return None
    return data["query"]


def create_search_router(
    resolver: QueryResolver,
    search: SearchService,
    token: str | None = None,
) -> APIRouter:
    """Make the router of the search. The JSON body has its own size limit."""
    assert isinstance(resolver, QueryResolver), "resolver must be a QueryResolver"
    assert isinstance(search, SearchService), "search must be a SearchService"
    assert token is None or token.strip(), "token must not be empty"
    router: APIRouter = APIRouter()

    async def run(
        request: Request,
        function: Callable[[], Any],
        build: Callable[[Any], Response] | None = None,
    ) -> Response:
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
        return JSONResponse(result, headers=PRIVATE_HEADERS)

    def find_json(object_id: int) -> dict[str, Any]:
        return answer_json(search.find(object_id))

    def search_json(words: str) -> dict[str, Any]:
        result: QueryResult = resolver.resolve(words)
        answer: dict[str, Any] | None = None
        if result.status.value == "matched":
            answer = find_json(result.candidates[0].object_id)
        return {
            "status": result.status.value,
            "candidates": [asdict(c) for c in result.candidates],
            "answer": answer,
        }

    @router.post(SEARCH_ROUTE)
    async def post_search(request: Request) -> Response:
        refusal: JSONResponse | None = token_refusal(request, token)
        if refusal is not None:
            return refusal
        body: bytes | JSONResponse = await read_body(request, MAX_JSON_BYTES)
        if isinstance(body, JSONResponse):
            return body
        words: str | None = _parse_search_body(body)
        if words is None:
            return error_response(400, MESSAGE_BODY)
        return await run(request, lambda: search_json(words))

    @router.get(OBJECTS_ROUTE + "/{object_id}/last-seen")
    async def get_last_seen(request: Request, object_id: str) -> Response:
        parsed: int | None = parse_id(object_id)
        if parsed is None:
            return token_refusal(request, token) or error_response(
                404, MESSAGE_NOT_FOUND
            )
        return await run(request, lambda: find_json(parsed))

    @router.get(OBJECTS_ROUTE + "/{object_id}/last-seen/picture")
    async def get_picture(request: Request, object_id: str) -> Response:
        refusal: JSONResponse | None = token_refusal(request, token)
        if refusal is not None:
            return refusal
        parsed: int | None = parse_id(object_id)
        if parsed is None:
            return error_response(404, MESSAGE_NOT_FOUND)
        view: str = request.query_params.get("view", "crop")
        if view not in PICTURE_VIEWS:
            return error_response(400, MESSAGE_VIEW)
        side_text: str = request.query_params.get("max_side", str(DEFAULT_CROP_SIDE))
        if not (
            side_text.isdigit() and MIN_CROP_SIDE <= int(side_text) <= MAX_CROP_SIDE
        ):
            return error_response(400, MESSAGE_QUERY)
        side: int = int(side_text)

        def image_answer(data: bytes) -> Response:
            return Response(
                content=data,
                media_type="image/jpeg",
                headers={**PRIVATE_HEADERS, "X-Content-Type-Options": "nosniff"},
            )

        return await run(
            request, lambda: search.get_picture(parsed, view, side), image_answer
        )

    return router
