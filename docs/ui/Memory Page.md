# Memory Page

## Purpose

The page `/memory` shows the pictures that Saarthi took in, and what the pipeline found in each of them.
A carer or a developer uses it to check the system: Which frames were processed? What did the detectors find? Why did a search fail?

The page only reads. It changes nothing.

## Privacy

The page shows **every frame**, also the frames that no search used, and the names of the registered objects.
It is open to every device on the local network, as the other pages are. There is no sign-in.
Use it on a network that only the household uses. A carer-only page needs a sign-in. This is a decision of the owner (see "Open points").

## What the user sees

1. **The filter.** Buttons for all frames and for each status, with the number of frames:
   Processed (`done`), Rejected, Failed, Waiting (`pending`) and Working (`processing`).
2. **The cards.** One card for each frame, the newest first, 24 on a page. A card has a small picture, the status, the age
   and a line such as `1 match, 1 region, 14 other boxes`. A rejected frame shows the reason ("too blurry"). The list reloads every 10 seconds, so a new frame appears without a tap.
3. **The page buttons** (Newer and Older) show the position, for example `1 to 24 of 60`.
4. **The details.** A tap on a card opens a full-screen sheet:
   - The picture of the frame with the boxes over it. A green box is a match of a registered object (with its name).
     An orange box is a box of the registered-object detector without a match. A dashed box is a box of the broad detector (with its name).
   - Two check boxes switch the two groups of boxes on and off.
   - The size and the sharpness of the frame, and a note (for example the reason of a rejection).
   - **Registered objects:** the matches (object, similarity, strong or weak, box), a line if the frame is the last-seen record of an object,
     and every box of the registered-object detector (words, confidence, source, in a hand, and why a crop is not used).
   - **Other objects:** every box of the broad detector with its name and confidence, and the link to a registered box that it overlaps.
     The text reminds the reader that a name can be wrong.
   - **Description:** the checked sentence of the language model about the place of an object, if one was made for this frame.
   - **Room:** the best room scores, if a description already scored this frame.

## How it is built

```
ui/pages/memory_page.py ──▶ ui/browse_session.py ──▶ ui.ports.BrowseService
                                                          │
                          backend/ui_adapter.py: UiBrowseService ──▶ backend/browser.py: MemoryBrowser
                                                                          │
                                                                  db/browse.py (rows)  +  storage (pictures)
```

- `db/browse.py` reads the rows: the frames with their counts, and everything stored for one frame. It never writes.
- `backend/browser.py`, `MemoryBrowser`, makes the list and the details, and cuts the pictures
  (the whole frame, 320 pixels for a card and 1280 pixels for the details).
- `UiBrowseService` runs the calls in a thread and turns every error into a short fixed text.
- The session keeps the filter, the page, the pictures and the open frame. It has no NiceGUI code, so tests need no browser.
- A box is placed over the picture in percent of the frame, so it fits at every size (`box_style`).

## Tests

| File | What it checks |
|------|----------------|
| `tests/test_db_browse.py` | The order, the counts, the filter, the paging, the details, a frame without extras, an unknown frame |
| `tests/test_backend_browser.py` | The page and the details of the browser, wrong requests, the picture, a missing file, the UI adapter and its fixed texts |
| `tests/test_ui_browse_session.py` | Load, filter, paging, open and close, a missing picture, errors |
| `tests/test_ui_memory_components.py` | The summary line, the placement of a box, the texts of a box, the filters |

The page also ran in a headless Chromium with the real models and the dataset (6 frames, one registered object).
The list, the details and the phone layout (420 pixels wide) showed correctly, and the boxes fit the picture.

## Open points

- The page has no sign-in. The owner must decide if only the carer may open it.
- A frame is shown with its status and counts only. There is no search in the list, and no button to delete a frame.
- The room scores appear only for frames that a description scored.
