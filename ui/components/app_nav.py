"""Navigation between the pages."""

from nicegui import ui

from ui.routes import ENROLL_ROUTE, FIND_ROUTE, MEMORY_ROUTE, UPLOAD_ROUTE

# (route, label) for each page, in display order.
PAGES: tuple[tuple[str, str], ...] = (
    (FIND_ROUTE, "Find"),
    (UPLOAD_ROUTE, "Upload"),
    (ENROLL_ROUTE, "My objects"),
    (MEMORY_ROUTE, "Memory"),
)


def app_nav(current_route: str) -> None:
    """Build the links. The link of the present page is marked for screen readers."""
    assert current_route in dict(PAGES), f"Unknown route: {current_route}"
    with ui.element("nav").classes("up-nav").props('aria-label="Pages"'):
        for route, label in PAGES:
            link = ui.link(label, route).classes("up-nav__link")
            if route == current_route:
                link.classes("up-nav__link--current").props("aria-current=page")
