"""The Talk button: the user speaks, and the page finds the object. The words that the computer heard show under it."""

from nicegui import ui

TITLE: str = "Say what you look for"
HINT: str = "Tap Talk, say for example: where is my bottle. Tap Stop when you finish."
HEARD_PREFIX: str = "I heard: "
BUTTON_ID: str = "saarthi-talk"


def find_talk(on_talk_tap_script: str, heard: str | None) -> None:
    """Build the Talk card. The script of the browser (`talk.js`) controls the text and the state of the button."""
    assert on_talk_tap_script, "the script is required"
    ui.label(TITLE).classes("up-section-title")
    ui.label(HINT).classes("up-find-text")
    button = ui.element("button").props(f"id={BUTTON_ID} type=button data-mode=idle")
    button.classes("up-talk")
    button.on("click", js_handler=on_talk_tap_script)
    with button:
        ui.element("span").classes("up-talk__text").text = "Talk"
    if heard:
        ui.label(f"{HEARD_PREFIX}{heard}").classes("up-find-text up-talk__heard")
