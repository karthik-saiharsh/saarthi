"""Parts that the HTTP routes share: error answers, the token check and the body reader."""

import hmac

from fastapi import Request
from fastapi.responses import JSONResponse

# Short messages for the client. They never contain internal details.
MESSAGE_EMPTY: str = "The request body is empty."
MESSAGE_BAD_LENGTH: str = "The Content-Length header is not valid."
MESSAGE_TOKEN: str = "A valid access token is required."
MESSAGE_TOO_LARGE: str = "The upload is too large."
MESSAGE_UNSUPPORTED: str = "The data is not a usable JPEG, PNG or WebP image."
MESSAGE_DISK_FULL: str = "The disk of the server is full."
MESSAGE_FAILED: str = "The server could not store the image."


def error_response(
    status_code: int, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    """Make the JSON answer for an error: `{"error": message}`."""
    return JSONResponse({"error": message}, status_code=status_code, headers=headers)


def token_is_valid(request: Request, token: str) -> bool:
    """Check the header `Authorization: Bearer <token>`. The scheme name is not case-sensitive."""
    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return False
    # Compare in constant time, so the answer time does not show how many characters match.
    return hmac.compare_digest(supplied.strip().encode(), token.encode())


def token_refusal(request: Request, token: str | None) -> JSONResponse | None:
    """Return the 401 answer if a token is required and the request has no valid one."""
    if token is None or token_is_valid(request, token):
        return None
    return error_response(401, MESSAGE_TOKEN, {"WWW-Authenticate": "Bearer"})


async def read_body(request: Request, max_bytes: int) -> bytes | JSONResponse:
    """Read the body in chunks. Stop at `max_bytes`, so a big upload never fills the memory."""
    declared: str | None = request.headers.get("content-length")
    if declared is not None:
        if not declared.isdigit():
            return error_response(400, MESSAGE_BAD_LENGTH)
        if int(declared) > max_bytes:
            return error_response(413, MESSAGE_TOO_LARGE)
    body: bytearray = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            return error_response(413, MESSAGE_TOO_LARGE)
    if not body:
        return error_response(400, MESSAGE_EMPTY)
    return bytes(body)
