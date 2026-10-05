"""Theme for the UI: serves the local CSS files and adds them to a page.

All colour values are in `static/css/tokens.css`. The CSS files load from
the local server, so the page needs no internet.
"""

import logging
from pathlib import Path

from nicegui import app, ui

logger: logging.Logger = logging.getLogger(__name__)

STATIC_DIR: Path = Path(__file__).resolve().parent / "static"
STATIC_URL: str = "/ui-static"

# The order is the load order. Tokens first, then base, then components.
CSS_FILES: tuple[str, ...] = (
    "css/tokens.css",
    "css/base.css",
    "css/components/header.css",
    "css/components/picker.css",
    "css/components/queue.css",
    "css/components/row.css",
    "css/components/badge.css",
    "css/components/summary.css",
    "css/components/nav.css",
    "css/components/enroll.css",
    "css/components/photos.css",
    "css/components/find.css",
    "css/components/talk.css",
)

_static_registered: bool = False


def register_static() -> None:
    """Serve the static folder. Call it once at start-up."""
    global _static_registered
    if _static_registered:
        return
    assert STATIC_DIR.is_dir(), f"Static folder not found: {STATIC_DIR}"
    app.add_static_files(STATIC_URL, str(STATIC_DIR))
    _static_registered = True
    logger.info("Serving static files from %s at %s", STATIC_DIR, STATIC_URL)


def stylesheet_links() -> str:
    """Return the <link> tags for all CSS files.

    The file time in the URL makes the browser load a changed file at once.
    """
    links: list[str] = []
    for relative_path in CSS_FILES:
        path: Path = STATIC_DIR / relative_path
        assert path.is_file(), f"CSS file not found: {path}"
        version: int = int(path.stat().st_mtime)
        links.append(
            f'<link rel="stylesheet" href="{STATIC_URL}/{relative_path}?v={version}">'
        )
    return "\n".join(links)


def apply_theme() -> None:
    """Add the theme to the page that is being built.

    Dark mode follows the system setting.
    """
    ui.dark_mode(value=None)
    ui.add_head_html(stylesheet_links())
