"""The HTTP routes of the wearable: a spoken question and its answer. See docs/backend/Wearable.md.

| Route | Does |
|-------|------|
| `POST /api/talk` | The body is a WAV recording (16-bit, one channel, 16 kHz). The answer is `202` with `{"id": "...", "status": "pending"}` |
| `GET /api/talk/{id}` | `{"id", "status", "outcome", "text", "detail", "speech"}`. `status` is `pending` or `done`. When it is `done`, `text` is the sentence to say, and `speech` is true if its audio is ready |
| `GET /api/talk/{id}/speech` | The audio of the sentence: a WAV file (16-bit, mono, 22 050 Hz for the voice of today), sent in chunks with a `Content-Length` |

The routes always need the header `Authorization: Bearer <token>` (owner decision of 2026-11-14).
If no token is set, they answer `503`: anyone on the Wi-Fi could otherwise ask where the things of the patient are.

| Code | Meaning |
|------|---------|
| 202  | The question is kept. Ask for the answer with the id |
| 200  | The state of the job |
| 400  | The body is empty or the headers are wrong |
| 401  | The token is missing or wrong |
| 404  | No job has this id, or it was deleted (`talk.keep_seconds`). For `/speech`: also when the job has no audio (the voice is off or failed) |
| 409  | `/speech` only: the answer is not ready. Ask `GET /api/talk/{id}` again |
| 413  | The recording is too long |
| 503  | No token is set, listening is switched off, or too many questions wait (see `Retry-After`) |

The recording, the words and the answer are never logged. The answers have `Cache-Control: no-store`.
"""

import logging
import re
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from backend.errors import QueueFullError
from backend.http_common import error_response, read_body, token_refusal
from backend.talk import TalkJob, TalkService, TalkStatus

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
MESSAGE_NOT_READY: str = "The answer is not ready yet."
MESSAGE_NO_SPEECH: str = "This answer has no audio. Say the text with another voice."
# The audio goes out in parts of this size, so the wearable can play it while it arrives.
SPEECH_CHUNK_BYTES: int = 4096


def job_json(job: TalkJob) -> dict[str, Any]:
    """Make the JSON form of a job."""
    return {
        "id": job.id,
        "status": job.status.value,
        "outcome": None if job.outcome is None else job.outcome.value,
        "text": job.text,
        "detail": job.detail,
        "speech": job.speech is not None,
    }


def _chunks(data: bytes) -> Iterator[bytes]:
    for start in range(0, len(data), SPEECH_CHUNK_BYTES):
        yield data[start : start + SPEECH_CHUNK_BYTES]


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

    @router.get(TALK_ROUTE + "/{job_id}/speech")
    async def get_speech(request: Request, job_id: str) -> Response:
        refused: JSONResponse | None = refusal(request)
        if refused is not None:
            return refused
        if not _JOB_ID.fullmatch(job_id):
            return error_response(404, MESSAGE_NOT_FOUND)
        job: TalkJob | None = service.get(job_id)
        if job is None:
            return error_response(404, MESSAGE_NOT_FOUND)
        if job.status == TalkStatus.PENDING:
            return error_response(409, MESSAGE_NOT_READY)
        if job.speech is None:
            return error_response(404, MESSAGE_NO_SPEECH)
        return StreamingResponse(
            _chunks(job.speech),
            media_type="audio/wav",
            headers={
                **PRIVATE_HEADERS,
                "Content-Length": str(len(job.speech)),
                "X-Content-Type-Options": "nosniff",
            },
        )

    return router
