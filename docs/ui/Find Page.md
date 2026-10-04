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

A navigation links `/find`, `/upload` and `/enroll` on all pages. `/find` is the first link.
The address `/` still leads to `/upload`.

## Interface to the backend

`ui/ports.py` defines `FindService`. All methods are `async`. Each raises `FindError` for an expected problem.
The text of the error is safe to show.

| Method | Result |
|--------|--------|
| `list_choices()` | The objects that a search can find: `FindChoice` (`object_id`, `name`) |
| `ask(text)` | A `FindResult` for the words of the user |
| `find(object_id)` | A `FindResult` for an object that the user chose |

The real implementation is `Backend.find_service` (`UiFindService` in `backend/ui_adapter.py`).
It runs the blocking work in a thread and turns every error into a fixed text.
`StubFindService` in `ui/stub_service.py` works with four fixed objects, for work on the UI.

A `FindResult` has a `status`:

| Status | Meaning | What the page shows |
|--------|---------|---------------------|
| `found` | The system saw the object | The name, the time and the picture |
| `not_seen` | The object is known, but the system did not see it | "The computer has not seen it lately." |
| `choose` | The words fit several objects | "Which one do you mean?" and a button for each object |
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

| Line | When |
|------|------|
| "Seen 12 minutes ago." | Always (large text). The time is an age: just now, minutes, hours or days |
| "You were holding it." | `in_hand` is true |
| "Maybe. The computer is not sure. Look at the picture." | `strong` is false |
| "This was a long time ago. The computer found nothing newer." | `stale` is true |
| "There is no picture." | The picture is none, for example when the file of the frame is gone |

While a search runs, the card shows a spinner and "Looking...". The buttons are off.

## The session

`ui/find_session.py` has `FindSession`. It has no NiceGUI code. Each browser tab has its own session.

- It keeps the objects, the result of the last search and a `busy` flag.
- It cleans the words (extra spaces). It does not send empty words or words with more than 300 characters.
  The user sees a short notice. The backend checks the words again.
- It removes the old result when a new search starts.
- A request while a search runs is ignored. A double tap does not start two searches.
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
- The page does not speak. Speech input and speech output are not built.
- The page cannot ask the wearable to take a new picture.
- A weak match is shown as "maybe". If the check (I5) passes, the match counts as strong and the page does not say "maybe".

## Tests

| File | What it checks |
|------|----------------|
| `tests/test_ui_find_session.py` | Load, clean words, limits, the busy flag, the old result, double taps, errors, listener errors |
| `tests/test_ui_stub_find.py` | The stub: the four results, the fixed objects, an unknown object |
| `tests/test_ui_components.py` | `age_text`, the lines of a result, the data URI, the navigation, short texts |
| `tests/test_backend_ui_find.py` | The real adapter: the results, fixed error texts, no logging of the words |
| `tests/test_ui_theme.py` | All CSS files exist and are listed. Colours only in `tokens.css` |

The page also ran in a headless Chromium browser with the stub service. The test used light and dark mode, at 390 pixels width.
The test chose objects, wrote words, and covered all four results and the empty-words notice.
The page had no horizontal scroll. The test did not read the browser console.
The test found two faults, and both are fixed:
the result was below the buttons, and the page did not scroll to it.
A later run with the real application and real photos found a third fault: the picture of a bottle pushed the time off the screen.
The lines now come first.

## Planned: voice

The page will get a voice input and a voice output. They are planned and not built. The design is in `docs/backend/Voice.md`.

- A large "Talk" button. The patient speaks, and the page shows the words that the system heard.
- The result card keeps the picture. The page also says the answer with a voice, and describes the place. A "Say it again" button repeats it.
- The microphone works only on `localhost` or on a page with HTTPS. This decides where the UI must run.
- The first version keeps the buttons and the text box. They are the way for a carer to use the page.
