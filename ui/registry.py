"""Page registration. The application calls this once at start-up."""

import logging

from nicegui import ui

from ui.pages.enroll_page import build_enroll_page
from ui.pages.upload_page import build_upload_page
from ui.ports import EnrollmentService, UploadService
from ui.routes import ENROLL_ROUTE, UPLOAD_ROUTE
from ui.theme import register_static

logger: logging.Logger = logging.getLogger(__name__)


def register_pages(
    upload_service: UploadService, enrollment_service: EnrollmentService
) -> None:
    """Register the pages of the UI. The caller starts the server."""
    assert upload_service is not None, "upload_service is required"
    assert enrollment_service is not None, "enrollment_service is required"
    register_static()

    @ui.page(UPLOAD_ROUTE)
    def upload() -> None:
        build_upload_page(upload_service)

    @ui.page(ENROLL_ROUTE)
    def enroll() -> None:
        build_enroll_page(enrollment_service)

    logger.info("Registered pages %s and %s", UPLOAD_ROUTE, ENROLL_ROUTE)
