"""The voice of the page. The browser speaks a sentence with a voice of the device.

Only a local voice is used. Some voices of a browser send the text to a cloud
service, so the script skips every voice that is not local. A device without a
local English voice stays silent. The picture and the text stay on the page.
"""

import json
import logging
from typing import Protocol

from nicegui import ui

logger: logging.Logger = logging.getLogger(__name__)

MAX_SPOKEN_CHARS: int = 600
SAY_AGAIN_TEXT: str = "Say it again"

# The function is defined one time for each page. It returns true if it spoke.
SPEECH_SCRIPT: str = """
window.saarthiSpeak = function (text) {
  const synth = window.speechSynthesis;
  if (!synth) { return false; }
  const voice = synth.getVoices().find(
    (v) => v.localService && v.lang && v.lang.toLowerCase().startsWith('en')
  );
  if (!voice) { return false; }
  synth.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = voice;
  utterance.lang = voice.lang;
  utterance.rate = 0.9;
  synth.speak(utterance);
  return true;
};
if (window.speechSynthesis) { window.speechSynthesis.getVoices(); }
"""


def add_speech_script() -> None:
    """Add the speech function to the page. Call it while the page is built."""
    ui.add_head_html(f"<script>{SPEECH_SCRIPT}</script>")


class Browser(Protocol):
    """The part of a NiceGUI client that this module uses."""

    def run_javascript(self, code: str) -> object: ...


def speak(text: str, browser: Browser) -> None:
    """Ask the browser to say the text. The text is never logged.

    Pass the client of the page (`ui.context.client`, taken while the page is built).
    `ui.run_javascript` uses the element of the current code, and that element can be gone:
    the page replaces the result card after each search.
    """
    assert isinstance(text, str), "text must be a string"
    assert browser is not None, "browser is required"
    words: str = " ".join(text.split())[:MAX_SPOKEN_CHARS]
    if not words:
        return
    browser.run_javascript(f"window.saarthiSpeak({json.dumps(words)})")
