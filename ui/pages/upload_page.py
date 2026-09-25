"""The /upload page. It builds the components and connects them to a session."""

import logging

from nicegui import ui

from ui.components.app_header import app_header
from ui.components.app_nav import app_nav
from ui.components.summary_bar import SummaryBar
from ui.components.upload_picker import upload_picker
from ui.components.upload_queue import QueueView
from ui.models import FINISHED_STATUSES, ItemStatus, UploadItem
from ui.ports import UploadService
from ui.routes import UPLOAD_ROUTE
from ui.theme import apply_theme
from ui.upload_session import UploadSession

logger: logging.Logger = logging.getLogger(__name__)

PRODUCT_NAME: str = "Saarthi"
PAGE_TITLE: str = "Upload pictures"
PAGE_SUBTITLE: str = "Send pictures from your wearable or phone to the memory."


def build_upload_page(service: UploadService) -> None:
    """Build the page for one browser tab. Each tab has its own session."""
    assert service is not None, "service is required"
    apply_theme()
    ui.page_title(f"{PAGE_TITLE} - {PRODUCT_NAME}")

    # The callbacks use `queue` and `summary`, which are created below.
    # Python looks them up when a callback runs, not when it is defined.
    def refresh_summary() -> None:
        counts: dict[ItemStatus, int] = {s: session.count(s) for s in ItemStatus}
        finished: int = sum(counts[s] for s in FINISHED_STATUSES)
        summary.update(counts, finished)

    def on_item_changed(item: UploadItem) -> None:
        queue.update_item(item)
        refresh_summary()

    def on_list_changed() -> None:
        queue.render(session.items)
        refresh_summary()

    session: UploadSession = UploadSession(service, on_item_changed, on_list_changed)

    with ui.element("main").classes("up-page"):
        app_header(PRODUCT_NAME, PAGE_TITLE, PAGE_SUBTITLE)
        app_nav(UPLOAD_ROUTE)
        with ui.element("section").classes("up-card"):
            upload_picker(
                lambda name, content_type, data: session.add(name, content_type, data)
            )
        with ui.element("section").classes("up-card up-stack"):
            summary: SummaryBar = SummaryBar(on_clear=session.clear_finished)
            queue: QueueView = QueueView(on_retry=session.retry)
    logger.info("Upload page built")
