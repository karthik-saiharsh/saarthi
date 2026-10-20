"""The HTTP routes of the wearable: a spoken question and its answer. See docs/backend/Wearable.md.

| Route | Does |
|-------|------|
| `POST /api/talk` | The body is a WAV recording (16-bit, one channel, 16 kHz). The answer is `202` with `{"id": "...", "status": "pending"}` |
| `GET /api/talk/{id}` | `{"id", "status", "outcome", "text", "detail"}`. `status` is `pending` or `done`. When it is `done`, `text` is the sentence to say |

The routes always need the header `Authorization: Bearer <token>` (owner decision of 2026-11-14).
If no token is set, they answer `503`: anyone on the Wi-Fi could otherwise ask where the things of the patient are.

| Code | Meaning |
|------|---------|
| 202  | The question is kept. Ask for the answer with the id |
| 200  | The state of the job |
| 400  | The body is empty or the headers are wrong |
| 401  | The token is missing or wrong |
| 404  | No job has this id, or it was deleted (`talk.keep_seconds`) |
| 413  | The recording is too long |
| 503  | No token is set, listening is switched off, or too many questions wait (see `Retry-After`) |

The recording, the words and the answer are never logged. The answers have `Cache-Control: no-store`.
"""

import logging
import re
from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from backend.errors import QueueFullError
from backend.http_common import error_response, read_body, token_refusal
from backend.talk import TalkJob, TalkService

logger: logging.Logger = logging.getLogger(__name__)

TALK_ROUTE: str = "/api/talk"
RETRY_AFTER_SECONDS: int = 5
PRIVATE_HEADERS: dict[str, str] = {"Cache-Control": "no-store"}
_JOB_ID: re.Pattern[str] = re.compile(r"[A-Za-z0-9_-]{22}")

MESSAGE_NO_TOKEN: str = "The talk routes need an access token on the server."
MESSAGE_OFF: str = "Listening is switched off."
MESSAGE_BUSY: str = "Too many questions wait for an answer. Try again later."
MESSAGE_NOT_FOUND: str = "No question has this id."
MESSAGE_FAILED: str = "The server could not take the question."


def job_json(job: TalkJob) -> dict[str, Any]:
    """Make the JSON form of a job."""
    return {
        "id": job.id,
        "status": job.status.value,
        "outcome": None if job.outcome is None else job.outcome.value,
        "text": job.text,
        "detail": job.detail,
    }


def create_talk_router(service: TalkService, token: str | None) -> APIRouter:
    """Make the router of the wearable. Without a `token`, every route answers 503."""
    assert isinstance(service, TalkService), "service must be a TalkService"
    assert token is None or token.strip(), "token must not be empty"
    router: APIRouter = APIRouter()

    def refusal(request: Request) -> JSONResponse | None:
        if token is None:
            return error_response(503, MESSAGE_NO_TOKEN)
        refused: JSONResponse | None = token_refusal(request, token)
        if refused is not None:
            logger.warning("Talk request refused: the token is missing or wrong")
        return refused

    @router.post(TALK_ROUTE)
    async def post_talk(request: Request) -> JSONResponse:
        refused: JSONResponse | None = refusal(request)
        if refused is not None:
            return refused
        if not service.enabled:
            return error_response(503, MESSAGE_OFF)
        body: bytes | JSONResponse = await read_body(request, service.max_bytes)
        if isinstance(body, JSONResponse):
            return body
        try:
            job: TalkJob = await run_in_threadpool(service.submit, body)
        except QueueFullError:
            return error_response(
                503, MESSAGE_BUSY, {"Retry-After": str(RETRY_AFTER_SECONDS)}
            )
        except RuntimeError:  # The background thread was stopped (shutdown)
            logger.warning("Talk: a question came during the shutdown")
            return error_response(503, MESSAGE_FAILED)
        return JSONResponse(job_json(job), status_code=202, headers=PRIVATE_HEADERS)

    @router.get(TALK_ROUTE + "/{job_id}")
    async def get_talk(request: Request, job_id: str) -> JSONResponse:
        refused: JSONResponse | None = refusal(request)
        if refused is not None:
            return refused
        if not _JOB_ID.fullmatch(job_id):
            return error_response(404, MESSAGE_NOT_FOUND)
        job: TalkJob | None = service.get(job_id)
        if job is None:
            return error_response(404, MESSAGE_NOT_FOUND)
        return JSONResponse(job_json(job), headers=PRIVATE_HEADERS)

    return router
