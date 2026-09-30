"""Limits for the upload page.

The browser checks these limits for a quick answer only.
The backend must check every upload again.
"""

ACCEPTED_CONTENT_TYPES: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
ACCEPT_ATTRIBUTE: str = ",".join(ACCEPTED_CONTENT_TYPES)
MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024

# The longest question that the page sends. The backend checks it again.
MAX_QUERY_CHARS: int = 300
