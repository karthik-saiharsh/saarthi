"""The route that receives the recording of the Talk button: `POST /ui/voice`.

The browser sends a WAV file. The route reads it in chunks, up to the limit of the service, and gives it to
the voice service. The answer is `{"text": "..."}` or `{"error": "..."}`. The audio is never stored or logged.
The page is open to the local network, as the other pages are, so this route has no token.
"""

import logging
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ui.ports import VoiceError, VoiceService

logger: logging.Logger = logging.getLogger(__name__)

VOICE_ROUTE: str = "/ui/voice"
MESSAGE_TOO_LARGE: str = "The recording is too long."
MESSAGE_EMPTY: str = "The recording is empty."
# The browser must not keep the words of a user.
PRIVATE_HEADERS: dict[str, str] = {"Cache-Control": "no-store"}


class WebApp(Protocol):
    """The part of the NiceGUI `app` that this module uses."""

    def post(self, path: str, **kwargs: object): ...


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status, headers=PRIVATE_HEADERS)


async def _read(request: Request, max_bytes: int) -> bytes | JSONResponse:
    """Read the body in chunks. Stop at `max_bytes`, so a big body never fills the memory."""
    declared: str | None = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > max_bytes):
        return _error(413, MESSAGE_TOO_LARGE)
    body: bytearray = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            return _error(413, MESSAGE_TOO_LARGE)
    if not body:
        return _error(400, MESSAGE_EMPTY)
    return bytes(body)


def register_voice_route(web_app: FastAPI | WebApp, service: VoiceService) -> None:
    """Add the route to the web application. Call it one time at start-up."""
    assert web_app is not None, "web_app is required"
    assert service is not None, "service is required"

    @web_app.post(VOICE_ROUTE)
    async def post_voice(request: Request) -> JSONResponse:
        body: bytes | JSONResponse = await _read(request, service.max_bytes)
        if isinstance(body, JSONResponse):
            return body
        try:
            text: str = await service.transcribe(body)
        except VoiceError as error:
            return _error(422, str(error))
        return JSONResponse({"text": text}, headers=PRIVATE_HEADERS)

    logger.info("Registered the voice route %s", VOICE_ROUTE)
