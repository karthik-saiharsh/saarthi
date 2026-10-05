"""The /find page. It builds the components and connects them to a session."""

import logging

from nicegui import ui

from ui.components.app_header import app_header
from ui.components.app_nav import app_nav
from ui.components.find_ask import find_ask
from ui.components.find_choices import find_choices
from ui.components.find_result import find_result
from ui.find_session import FindSession
from ui.ports import FindResult, FindService
from ui.routes import FIND_ROUTE
from ui.speech import add_speech_script, speak
from ui.theme import apply_theme

logger: logging.Logger = logging.getLogger(__name__)

PRODUCT_NAME: str = "Saarthi"
PAGE_TITLE: str = "Find my things"
PAGE_SUBTITLE: str = "Choose an object, or tell Saarthi what you are looking for."


def _notify(text: str, is_error: bool) -> None:
    ui.notify(text, type="negative" if is_error else "positive")


def build_find_page(service: FindService) -> None:
    """Build the page for one browser tab. Each tab has its own session."""
    assert service is not None, "service is required"
    apply_theme()
    add_speech_script()
    ui.page_title(f"{PAGE_TITLE} - {PRODUCT_NAME}")

    # The functions below use `session`, which is created after them.
    # Python looks it up when they run, not when they are defined.
    @ui.refreshable
    def result_view() -> None:
        find_result(session.result, session.busy, session.pick, speak)

    @ui.refreshable
    def choices_view() -> None:
        find_choices(session.choices, session.busy, session.pick)

    @ui.refreshable
    def ask_view() -> None:
        find_ask(session.ask, session.busy)

    spoken: list[FindResult | None] = [None]

    def on_changed() -> None:
        result: FindResult | None = session.result
        # Say each new answer one time. The tap that started the search lets the browser play sound.
        if (
            not session.busy
            and result is not None
            and result is not spoken[0]
            and result.description
        ):
            spoken[0] = result
            speak(result.description)
        result_view.refresh()
        choices_view.refresh()
        ask_view.refresh()
        # The result is the first card, so it is on the screen of a phone. It is hidden until there is one.
        result_card.set_visibility(session.busy or session.result is not None)
        if session.busy:
            # The user tapped a button that can be far down the page. Show the result card.
            ui.run_javascript("window.scrollTo(0, 0)")

    session: FindSession = FindSession(service, on_changed, _notify)

    with ui.element("main").classes("up-page"):
        app_header(PRODUCT_NAME, PAGE_TITLE, PAGE_SUBTITLE)
        app_nav(FIND_ROUTE)
        with ui.element("section").classes("up-card up-stack") as result_card:
            result_view()
        with ui.element("section").classes("up-card up-stack"):
            choices_view()
        with ui.element("section").classes("up-card up-stack"):
            ask_view()
        result_card.set_visibility(False)

    ui.timer(0.1, session.load, once=True)
    logger.info("Find page built")
