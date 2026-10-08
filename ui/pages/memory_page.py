"""The /memory page: the frames that the worker took, and what the pipeline stored for each of them."""

import logging

from nicegui import ui

from ui.browse_session import BrowseSession
from ui.components.app_header import app_header
from ui.components.app_nav import app_nav
from ui.components.frame_detail import frame_detail
from ui.components.memory_list import (
    LOADING_TEXT,
    memory_filters,
    memory_grid,
    memory_pager,
)
from ui.ports import BrowseService
from ui.routes import MEMORY_ROUTE
from ui.theme import apply_theme

logger: logging.Logger = logging.getLogger(__name__)

PRODUCT_NAME: str = "Saarthi"
PAGE_TITLE: str = "Memory"
PAGE_SUBTITLE: str = (
    "The pictures that Saarthi took in, and what it found in each of them. Tap a picture for the details."
)
# Seconds between two reloads of the list. A new frame shows up without a tap.
POLL_SECONDS: float = 10.0


def _notify(text: str, is_error: bool) -> None:
    ui.notify(text, type="negative" if is_error else "positive")


def build_memory_page(service: BrowseService) -> None:
    """Build the page for one browser tab. Each tab has its own session."""
    assert service is not None, "service is required"
    apply_theme()
    ui.page_title(f"{PAGE_TITLE} - {PRODUCT_NAME}")
    layers: dict[str, bool] = {"regions": True, "open": True}

    # The functions below use `session`, which is created after them.
    @ui.refreshable
    def filters_view() -> None:
        if session.page is not None:
            memory_filters(session.page.counts, session.status, session.set_status)

    @ui.refreshable
    def grid_view() -> None:
        if session.page is None:
            ui.label(LOADING_TEXT).classes("up-empty")
            return
        memory_grid(session.page, session.thumbnails, session.open)
        memory_pager(
            session.page,
            session.has_previous,
            session.has_next,
            session.previous_page,
            session.next_page,
        )

    @ui.refreshable
    def detail_view() -> None:
        if session.detail is not None:
            frame_detail(
                session.detail,
                session.detail_picture,
                layers,
                detail_view.refresh,
                session.close,
            )

    def on_changed() -> None:
        filters_view.refresh()
        grid_view.refresh()
        detail_view.refresh()
        if session.detail is not None:
            dialog.open()
        else:
            dialog.close()

    session: BrowseSession = BrowseSession(service, on_changed, _notify)

    with ui.dialog().props("maximized") as dialog, ui.card().classes("up-sheet"):
        detail_view()
    dialog.on("hide", session.close)

    with ui.element("main").classes("up-page"):
        app_header(PRODUCT_NAME, PAGE_TITLE, PAGE_SUBTITLE)
        app_nav(MEMORY_ROUTE)
        with ui.element("section").classes("up-card up-stack"):
            filters_view()
            grid_view()

    async def poll() -> None:
        # Do not reload while a frame is open or a load runs: the open frame would flicker.
        if session.detail is None and not session.busy:
            await session.load()

    ui.timer(POLL_SECONDS, poll)
    ui.timer(0.1, session.load, once=True)
    logger.info("Memory page built")
