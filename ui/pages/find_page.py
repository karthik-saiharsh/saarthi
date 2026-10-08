"""The /find page. It builds the components and connects them to a session."""

import logging

from nicegui import ui

from ui.components.app_header import app_header
from ui.components.app_nav import app_nav
from ui.components.find_ask import find_ask
from ui.components.find_choices import find_choices
from ui.components.find_result import find_result
from ui.components.find_talk import find_talk
from ui.find_session import FindSession
from ui.ports import FindResult, FindService, VoiceService
from ui.routes import FIND_ROUTE
from ui.speech import add_speech_script, speak
from ui.theme import STATIC_URL, apply_theme
from ui.voice_route import VOICE_ROUTE

logger: logging.Logger = logging.getLogger(__name__)

PRODUCT_NAME: str = "Saarthi"
PAGE_TITLE: str = "Find my things"
# NiceGUI needs a JavaScript function here. A bare expression runs at once when the page is built, and it breaks the card.
TALK_SCRIPT: str = "() => window.saarthiTalk.toggle()"
MESSAGE_VOICE: str = "The computer could not listen. Please try again."
PAGE_SUBTITLE: str = "Choose an object, or tell Saarthi what you are looking for."


def _notify(text: str, is_error: bool) -> None:
    ui.notify(text, type="negative" if is_error else "positive")


def build_find_page(service: FindService, voice: VoiceService | None = None) -> None:
    """Build the page for one browser tab. Each tab has its own session."""
    assert service is not None, "service is required"
    apply_theme()
    add_speech_script()
    ui.page_title(f"{PAGE_TITLE} - {PRODUCT_NAME}")
    # The client outlives the cards that the page replaces. Run scripts through it, not through `ui.run_javascript`.
    browser = ui.context.client

    def say(text: str) -> None:
        speak(text, browser)

    # The functions below use `session`, which is created after them.
    # Python looks it up when they run, not when they are defined.
    @ui.refreshable
    def result_view() -> None:
        find_result(session.result, session.busy, session.pick, say)

    @ui.refreshable
    def choices_view() -> None:
        find_choices(session.choices, session.busy, session.pick)

    @ui.refreshable
    def talk_view() -> None:
        find_talk(TALK_SCRIPT, session.heard)

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
            say(result.description)
        result_view.refresh()
        choices_view.refresh()
        ask_view.refresh()
        if voice is not None:
            talk_view.refresh()
        # The result is the first card, so it is on the screen of a phone. It is hidden until there is one.
        result_card.set_visibility(session.busy or session.result is not None)
        if session.busy:
            # The user tapped a button that can be far down the page. Show the result card.
            browser.run_javascript("window.scrollTo(0, 0)")

    session: FindSession = FindSession(service, on_changed, _notify)

    with ui.element("main").classes("up-page"):
        app_header(PRODUCT_NAME, PAGE_TITLE, PAGE_SUBTITLE)
        app_nav(FIND_ROUTE)
        with ui.element("section").classes("up-card up-stack") as result_card:
            result_view()
        if voice is not None:
            with ui.element("section").classes("up-card up-stack"):
                talk_view()
        with ui.element("section").classes("up-card up-stack"):
            choices_view()
        with ui.element("section").classes("up-card up-stack"):
            ask_view()
        result_card.set_visibility(False)

    if voice is not None:
        # The browser records. It sends the sound to the server, and gives the words to the session.
        ui.add_head_html(f'<script src="{STATIC_URL}/js/talk.js?v=1"></script>')
        ui.on(
            "saarthi_heard", lambda event: session.hear(str(event.args.get("text", "")))
        )
        ui.on(
            "saarthi_voice_error",
            lambda event: session.voice_failed(
                str(event.args.get("message") or MESSAGE_VOICE)
            ),
        )
        configure: str = (
            f"window.saarthiTalk.configure({{maxSeconds: {voice.max_seconds}, "
            f"endpoint: '{VOICE_ROUTE}'}})"
        )
        ui.timer(0.1, lambda: ui.run_javascript(configure), once=True)

    ui.timer(0.1, session.load, once=True)
    logger.info("Find page built")
