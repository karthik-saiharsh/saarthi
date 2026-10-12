# Enrollment Page

## Purpose

The page `/enroll` lets a carer add the objects that the owner wants to find.
The carer gives each object a name and a category phrase, adds photos, and checks the crop of each photo.
The design of the steps is in `docs/backend/Enrollment.md`.

The page has no application logic. It shows state and calls the backend through `ui/ports.py`.
It works on phones and on desktop screens, in light and in dark mode.

## Route

| Route | Page |
|-------|------|
| `/enroll` | My objects |

A small navigation links `/find`, `/upload`, `/enroll` and `/memory` on all pages.

## Interface to the backend

`ui/ports.py` defines `EnrollmentService`. All methods are `async`.

| Method | Result |
|--------|--------|
| `create_object(name, category_phrase, synonyms, tracked)` | `ObjectInfo` |
| `list_objects()` | A list of `ObjectInfo` |
| `get_object(object_id)` | `ObjectDetail`: the object and its photos |
| `delete_object(object_id)` | Nothing |
| `add_photo(object_id, data)` | `PhotoInfo` |
| `confirm_photo`, `remove_photo`, `retry_photo` (`object_id`, `photo_id`) | Nothing |
| `get_crop(object_id, photo_id, max_side)` | The bytes of a JPEG |

Each method raises `EnrollmentError` for an expected problem. The text of the error is safe to show.
The real implementation is `Backend.enrollment_service` (`backend/ui_adapter.py`).
It runs the blocking service in a thread and turns every error into a fixed text.
`StubEnrollmentService` in `ui/stub_service.py` works in memory, for work on the UI.

## What the user sees

From top to bottom:

1. The header and the navigation.
2. A help text, closed at the start. It explains the name, the category phrase, the photos and the crop check.
3. The form: name, kind of object (the category phrase), other words (synonyms, separated by commas),
   and the check box "This object only looks like one of mine. Never report it."
4. The list of objects. Each row has the name, the phrase, a count of the photos, a note, and a badge: Ready, Needs photos or Look-alike.
5. The open object: its name, its words, the buttons Close and Delete object, the photo picker, and one card for each photo.

The photo picker is the same component as on the upload page. On a phone it also has a camera button.
The section of the open object is hidden while no object is open.

### Photo cards

| State | Badge | Buttons |
|-------|-------|---------|
| `pending`, `processing` | Waiting, Looking | None. The server holds the photo |
| `located` | Check the crop | Accept, Remove |
| `confirmed` | Accepted | Remove |
| `failed` | Not found | Try again, Remove |

A card shows the crop of the photo. A failed photo shows a text instead.
The text for `object not found` tells the carer to remove the photo or to try again.
Another reason is not shown, because it is an internal error name.

"Delete object" asks for a confirmation first.

## The session

`ui/enroll_session.py` has `EnrollSession`. It has no NiceGUI code. Each browser tab has its own session.

- It keeps the objects, the open object and the crops of its photos.
- It runs one service call at a time. A second call waits. So an old reload cannot overwrite a newer state.
- After each action it loads the objects and the open object again, also after an error.
- It loads the crop of a located photo one time (at most 320 pixels on the long side). It drops the crop when the photo is gone.
- It calls `on_changed` only when something is new. A reload without a change does not draw the page again.
- It calls `on_notice(text, is_error)` for a short message. The page shows it as a toast.
- The browser checks the type and the size of a photo before it sends the photo. The backend checks again.
- A page timer calls `poll()` every 2 seconds. `poll()` loads again only while a photo of the open object is `pending` or `processing`.

When the carer adds an object, the session opens it at once, so the carer can add photos.

## Components

| File | Purpose |
|------|---------|
| `components/app_nav.py` | The links between the pages |
| `components/enroll_guide.py` | The help text |
| `components/object_form.py` | The form |
| `components/object_list.py` | The list of objects. `photo_summary` makes the count text |
| `components/object_detail.py` | The header of the open object, the photo grid and the photo cards |
| `components/upload_picker.py` | The file picker. It takes a title and a sync or async handler |
| `components/status_badge.py` | The badges for files, photos and objects. The text is always shown, so colour is not the only signal |
| `pages/enroll_page.py` | Builds the page and connects the components to the session |

The crop goes to the browser as a data URI in the page. It is not a link to the server. This works also when the token is on.
The browser does not keep it as a file.

## Styling

The CSS files are `components/nav.css`, `enroll.css` and `photos.css`. They use the tokens of `tokens.css` only.
A test checks that only `tokens.css` has colour values. All CSS files are in `ui/theme.py`.
The photo cards use a grid with at least 9 rem for each card, so a phone shows two columns.

## Known limits

- The page does not change an object. The carer deletes it and adds it again.
- The page has no way to draw the box by hand.
- The carer cannot see the original photo. The carer sees the crop only.
- The page does not stop a double click on Accept. The second call gets a message that the state is wrong.
- A photo that the carer chooses goes to the open object only. A photo is ignored while no object is open.

## Tests

| File | What it checks |
|------|----------------|
| `tests/test_ui_enroll_session.py` | Limits, the order of calls, crops, polling, changes only when needed, delete, listener errors |
| `tests/test_ui_stub_enrollment.py` | The stub: the states over time, ready after 3 photos, a failed photo, wrong actions |
| `tests/test_ui_components.py` | The count text, the failure text, the image source, the labels, the help text |
| `tests/test_backend_ui_enrollment.py` | The real adapter. 13 kinds of error never show internal text |
| `tests/test_ui_theme.py` | All CSS files exist and are listed. Colours only in `tokens.css` |

The page also ran in a headless Chromium browser with the stub service, in light and dark mode, at 1280 and 390 pixels width.
The test created an object, added 4 photos, and accepted 2 crops. The browser console had no error.
