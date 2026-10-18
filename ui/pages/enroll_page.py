"""The /enroll page. It builds the components and connects them to a session."""

import logging

from nicegui import ui

from ui.components.app_header import app_header
from ui.components.app_nav import app_nav
from ui.components.enroll_guide import enroll_guide
from ui.components.object_detail import object_header, photo_grid
from ui.components.object_form import object_form
from ui.components.object_list import object_list
from ui.components.object_review import object_review
from ui.components.upload_picker import upload_picker
from ui.enroll_session import EnrollSession
from ui.ports import EnrollmentService
from ui.routes import ENROLL_ROUTE
from ui.theme import apply_theme

logger: logging.Logger = logging.getLogger(__name__)

PRODUCT_NAME: str = "Saarthi"
PAGE_TITLE: str = "My objects"
PAGE_SUBTITLE: str = (
    "Add the objects that you want to find. Saarthi learns them from photos."
)
PICKER_TITLE: str = "Add photos of this object"
# Seconds between two checks while the server works on a photo.
POLL_SECONDS: float = 2.0


def _notify(text: str, is_error: bool) -> None:
    ui.notify(text, type="negative" if is_error else "positive")


def build_enroll_page(service: EnrollmentService) -> None:
    """Build the page for one browser tab. Each tab has its own session."""
    assert service is not None, "service is required"
    apply_theme()
    ui.page_title(f"{PAGE_TITLE} - {PRODUCT_NAME}")

    # The functions below use `session`, which is created after them.
    # Python looks it up when they run, not when they are defined.
    @ui.refreshable
    def objects_view() -> None:
        object_list(session.objects, session.selected_id, session.select)

    @ui.refreshable
    def header_view() -> None:
        if session.detail is not None:
            object_header(
                session.detail,
                session.delete_selected,
                session.close,
                session.set_small,
            )

    @ui.refreshable
    def photos_view() -> None:
        if session.detail is not None:
            photo_grid(
                session.detail,
                session.crop,
                session.confirm,
                session.remove,
                session.retry,
            )

    @ui.refreshable
    def review_view() -> None:
        if session.detail is not None:
            object_review(
                session.detail.info.name,
                session.review,
                session.review_picture,
                session.answer_review,
            )

    def on_changed() -> None:
        objects_view.refresh()
        header_view.refresh()
        photos_view.refresh()
        review_view.refresh()
        detail_card.set_visibility(session.selected_id is not None)

    session: EnrollSession = EnrollSession(service, on_changed, _notify)

    with ui.element("main").classes("up-page"):
        app_header(PRODUCT_NAME, PAGE_TITLE, PAGE_SUBTITLE)
        app_nav(ENROLL_ROUTE)
        with ui.element("section").classes("up-card"):
            enroll_guide()
        with ui.element("section").classes("up-card up-stack"):
            object_form(session.create_object)
        with ui.element("section").classes("up-card up-stack"):
            objects_view()
        with ui.element("section").classes("up-card up-stack") as detail_card:
            header_view()
            upload_picker(session.add_photo, title=PICKER_TITLE)
            photos_view()
            review_view()
        detail_card.set_visibility(False)

    ui.timer(POLL_SECONDS, session.poll)
    ui.timer(0.1, session.refresh, once=True)
    logger.info("Enrollment page built")
