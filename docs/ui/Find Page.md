# Find Page

## Purpose

The page `/find` lets a person ask where an object is.
The person can choose the object, or write what they look for in their own words.
The page shows where and when the system saw the object, with a picture.
The design of the search is in `docs/backend/Search.md`.

The person who uses this page may be confused or upset. So the page has big buttons, large text and short sentences.
It never shows a technical word. It says "maybe" when the system is not sure.

The page has no application logic. It shows state and calls the backend through `ui/ports.py`.
It works on phones and on desktop screens, in light and in dark mode.

## Route

| Route | Page |
|-------|------|
| `/find` | Find my things |

A navigation links `/find`, `/upload`, `/enroll` and `/memory` on all pages. `/find` is the first link.
The address `/` still leads to `/upload`.

## Interface to the backend

`ui/ports.py` defines `FindService`. All methods are `async`. Each raises `FindError` for an expected problem.
The text of the error is safe to show.

| Method | Result |
|--------|--------|
| `list_choices()` | The objects that a search can find: `FindChoice` (`object_id`, `name`) |
| `ask(text)` | A `FindResult` for the words of the user |
| `find(object_id)` | A `FindResult` for an object that the user chose |
| `ask_open(text)` | A `FindResult` from the open search only (objects that nobody registered): `found` (always weak, with a picture) or `no_match`. The button "Something else" calls it |

The real implementation is `Backend.find_service` (`UiFindService` in `backend/ui_adapter.py`).
It runs the blocking work in a thread and turns every error into a fixed text.
`StubFindService` in `ui/stub_service.py` works with four fixed objects, for work on the UI.

A `FindResult` has a `status`:

| Status | Meaning | What the page shows |
|--------|---------|---------------------|
| `found` | The system saw the object | The name, the time and the picture |
| `not_seen` | The object is known, but the system did not see it | "The computer has not seen it lately." |
| `choose` | The words fit several objects | "Which one do you mean?", a button for each object, and the button "Something else" |
| `no_match` | The words fit no object | "The computer does not know which object you mean. Choose an object from the list." |

A `found` result has a `Sighting`: the age in seconds, `in_hand`, `strong`, `stale` and the picture (a JPEG, or none).
A `not_seen` result also has `complete`. If it is False, the page adds: "The computer could not look at all the pictures. Try again later."

## What the user sees

From top to bottom:

1. The header and the navigation.
2. The **result card**. It is hidden until a search starts.
3. **Choose what you look for**: one big button for each object.
4. **Or write it in your own words**: a text box and the button "Find it". The key Enter also starts the search.

The result card is the first card on purpose. On a phone, a card below the buttons is out of sight.
When a search starts, the page also scrolls to the top, so the person sees the result.

### The lines of a result

The lines come before the picture. A tall object, for example a bottle, makes a tall picture.
On a phone, such a picture can fill the whole screen. The page limits the height of the picture to 45 % of the screen.
The picture is shown whole, and is never cut.

The page shows the name of the object, then these lines, from top to bottom:

| Line | When |
|------|------|
| "Seen 12 minutes ago." | Always (large text). The time is an age: just now, minutes, hours or days |
| "You were holding it." | `in_hand` is true |
| "Maybe. The computer is not sure. Look at the picture." | `strong` is false |
| "This was a long time ago. The computer found nothing newer." | `stale` is true |
| The sentence of the description | When the system saw the object. It says the time, the surface and the room, for example "It was on a table." The voice says the same text (see "The voice") |
| The button "Say it again" | Under the sentence of the description |
| The picture | Under the lines. "There is no picture." if it is none |
| "Maybe it is here now. The computer is not sure." | The answer has a newer weak sighting in another place (`FindResult.maybe`). Under it: "Seen 5 minutes ago.", "You were holding it." if true, and the second picture |

While a search runs, the card shows a spinner and "Looking...". The buttons are off.

### The button "Something else"

The resolver can give `choose` for an object that nobody registered. For example, "where is my laptop" can score above
`suggest_min_score` against a registered object. Then the choices are all wrong. The owner decided to add a way out.

- The card of a `choose` result has the button **Something else** under the choices.
  It is as large as the choice buttons (`up-find-choice`), with the secondary style (`up-btn--secondary`). The style uses the existing tokens only.
- A tap runs the open search with the same words that the user asked (`ask_open`). The registered objects are not used for this search.
- The open search uses the mode of `[open_search].mode` in `backend/config.toml` (`classic` or `verified`, see `docs/backend/Open Search.md`).
- The answer is `found` (always weak: "Maybe", and the sentence says "I am not sure") or `no_match`.
  The card, the sentence and the voice work as for any open answer.
- If the open search is off, the result is `no_match`.
- The resolver, its thresholds and the route `POST /api/search` do not change.

## The session

`ui/find_session.py` has `FindSession`. It has no NiceGUI code. Each browser tab has its own session.

- It keeps the objects, the result of the last search and a `busy` flag.
- It cleans the words (extra spaces). It does not send empty words or words with more than 300 characters.
  The user sees a short notice. The backend checks the words again.
- It removes the old result when a new search starts.
- A request while a search runs is ignored. A double tap does not start two searches.
- It keeps the words of the last `choose` result (`choose_words`). A tap on "Something else" (`ask_something_else`) sends these words to `ask_open`.
  Each new search removes the words. A tap without a `choose` result does nothing. The busy flag and the error rules are the same as for the other searches.
- A `FindError` shows its text as a notice. Any other error shows "The search did not work. Please try again."
  After an error the page still works.
- It calls `on_changed` when the state changes, and `on_notice(text, is_error)` for a short message.
- A listener error is logged and does not stop the session.

## Components

| File | Purpose |
|------|---------|
| `components/find_choices.py` | The buttons of the objects, and the text for an empty list |
| `components/find_ask.py` | The text box and the button |
| `components/find_result.py` | The result card. `age_text` and `sighting_lines` make the text |
| `pages/find_page.py` | Builds the page and connects the components to the session |

The picture goes to the browser as a data URI in the page. It is not a link to the server.
This works also when the token is on. The browser does not keep it as a file.

## Styling

The CSS file is `components/find.css`. It uses the tokens of `tokens.css` only. A test checks that only `tokens.css` has colour values.

- The object buttons are at least 4.5 rem high. They use a grid with at least 10 rem for each button.
- The text is large (`--text-l` and `--text-xl`).
- The result area is a live region (`role=status`, `aria-live=polite`). A screen reader reads a new result.
- Dark mode follows the system setting.

## Known limits

- The page shows an age, not a time of day. The server stores UTC and does not know the time zone of the browser.
- The page does not show the whole frame. The route `GET /api/objects/{id}/last-seen/picture?view=frame` gives it. No button uses it yet.
- The voice and the Talk button are not tested on a real phone or tablet (see "The voice" and "The Talk button").
- A choice between objects ("Which one do you mean?") needs a tap. An answer by voice is not built.
- An answer of the open search shows the name that the broad detector gave (for example "Laptop"). It is always weak.
- The page cannot ask the wearable to take a new picture.
- A weak match is shown as "maybe". If the check (I5) passes, the match counts as strong and the page does not say "maybe".

## Tests

| File | What it checks |
|------|----------------|
| `tests/test_ui_find_session.py` | Load, clean words, limits, the busy flag, the old result, double taps, errors, listener errors, "Something else" (the kept words, a tap, double taps, errors) |
| `tests/test_ui_stub_find.py` | The stub: the four results, the fixed objects, an unknown object, `ask_open` ("laptop" finds a weak answer) |
| `tests/test_backend_open_search.py` | `ask_open` in the real adapter: found, no match, the open search off, the mode, fixed error texts, no logging of the words |
| `tests/test_ui_pages.py` | The pages build. A `choose` result has the button "Something else" |
| `tests/test_ui_components.py` | `age_text`, the lines of a result, the data URI, the navigation, short texts |
| `tests/test_backend_ui_find.py` | The real adapter: the results, fixed error texts, no logging of the words |
| `tests/test_ui_voice_route.py` | The route of the Talk button: the text, the size limit, an empty body, a fixed error text, no caching |
| `tests/test_ui_speech.py` | The script uses local English voices only, empty text, quoting and cutting |
| `tests/test_ui_theme.py` | All CSS files exist and are listed. Colours only in `tokens.css` |

The page also ran in a headless Chromium browser with the stub service. The test used light and dark mode, at 390 pixels width.
The test chose objects, wrote words, and covered all four results and the empty-words notice.
The page had no horizontal scroll. The test did not read the browser console.
The test found two faults, and both are fixed:
the result was below the buttons, and the page did not scroll to it.
A later run with the real application and real photos found a third fault: the picture of a bottle pushed the time off the screen.
The lines now come first.

## The voice

The page says each new answer with a voice (stage Q3, `ui/speech.py`). The browser speaks the sentence of the description.
It uses a voice of the device, and only a **local** English voice. A voice that may use a cloud service is never used.
A device without a local English voice stays silent, and the text and the picture stay on the page.
A "Say it again" button repeats the sentence. The tap that starts a search lets the browser play sound.
The sound depends on the device. A test on the real phone and tablet is still to do.

## The Talk button (voice input, stage Q0)

The page has a large **Talk** button, above the object buttons. It shows only if listening is switched on (`speech.enabled`).

1. The user taps **Talk**. The button says **Stop**. The browser asks for the microphone one time.
2. The user says what they look for, and taps **Stop**. A recording stops by itself after `speech.max_seconds` (15).
3. The browser makes a WAV file (16 kHz, 16-bit, one channel) in its memory. It sends the file to the same server (`POST /ui/voice`).
4. The server turns the sound into text (`docs/backend/Voice.md`). The page shows "I heard: ..." and starts the search with these words.
   The answer comes with the picture, the sentence and the voice, as for a typed question.

If the server hears no speech, the page says so and starts no search. The audio is never stored, never logged and never sent to another service.
The microphone works only on a page with HTTPS (see `docs/backend/HTTPS.md`) or on `localhost`. Without it, the page shows a short notice.
The script is `ui/static/js/talk.js`. It uses a script processor node, which is old but works in all browsers.
The buttons and the text box stay. They are the way for a carer to use the page.

How to use it on a phone:
1. Open the page with `https://` and trust the local certificate (`docs/backend/HTTPS.md`). Without HTTPS the browser blocks the microphone.
2. The **Talk** card is the first card under the menu. It shows only if `speech.enabled` is true in `ai/config.toml`.
3. The first tap on **Talk** asks for the permission to use the microphone. Allow it.
4. Speak, and tap **Stop**. A recording of more than 15 seconds stops by itself. The words show under the button, and the answer shows above it.

The test with a fake microphone in a headless Chromium (a beep as the sound) went through all steps: the button, the recording, the upload, "I heard: ...", and the search.
A first version of the card showed an empty box, because the click handler was not a JavaScript function. `tests/test_ui_pages.py` now builds the pages and checks the button and its handler.
A test on the real phone and tablet is still to do.
