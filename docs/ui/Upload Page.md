# Upload Page

## Purpose

The upload page lets a user send pictures to the application.
The user can send one picture, or choose many pictures.
The page runs in a desktop browser and in a phone browser.
It is part of the image ingestion pipeline.

The page does not contain application logic.
It calls an `UploadService` and shows the result.
The real `UploadService` is `Backend.upload_service` (see `docs/backend/Upload Endpoint.md`). The stub is for work on the UI only.

## Route

| Route | Page |
|-------|------|
| `/upload` | Upload pictures |

`register_pages(upload_service, enrollment_service, find_service, voice_service, browse_service)` registers the routes and serves the CSS files.
`voice_service` adds the route of the Talk button. `browse_service` adds the page `/memory`. Both are optional.
A navigation links this page to `/find`, `/enroll` and `/memory` (see `docs/ui/Find Page.md`, `docs/ui/Enrollment Page.md` and `docs/ui/Memory Page.md`).
The caller starts the server.

## Interface to the backend

The UI reaches the backend only through `ui/ports.py`.

```python
class UploadService(Protocol):
    async def submit_image(
        self, filename: str, content_type: str, data: bytes
    ) -> UploadOutcome: ...
```

`UploadOutcome` has a `status` and a short `message` for the user.
The message must not contain personal data or internal error text.

| Outcome status | Meaning |
|----------------|---------|
| `stored` | The service accepted the image |
| `duplicate` | The service already has this image |
| `rejected` | The service cannot use the image (for example, too blurry) |
| `failed` | An error occurred. The user can try again |

`StubUploadService` is a fake service for work on the UI.
It waits 0.4 to 1.2 seconds, then returns an outcome from the file name:
`duplicate` gives `duplicate`, `reject` gives `rejected`, `fail` gives `failed`.
Any other name gives `stored`.

## What the user sees

1. A header with the product name and the page title.
2. A drop area with these elements:
   - **Choose images**: opens the file chooser. The user can select many files.
   - **Take photo**: opens the camera. It shows on touch devices only.
   - A line about drag and drop. It shows on devices that are not touch devices.
3. A list of the files, with the counts and the **Clear finished** button.

### Row status

| Status | Label | Meaning |
|--------|-------|---------|
| `waiting` | Waiting | The file is in the queue |
| `sending` | Sending | The service handles the file now |
| `stored` | Stored | The image is stored |
| `duplicate` | Already stored | The service has this image |
| `rejected` | Rejected | The file cannot be used |
| `failed` | Failed | An error occurred |

Each badge shows a text label. Colour is not the only signal.
A `failed` row shows the **Try again** button.

## Upload queue

`UploadSession` keeps the list of files of one browser tab.
Each tab has its own session.

- The session sends the files to the service one at a time, in the order they were added.
- Before it sends a file, the session checks the limits.
  An empty file, a file type that is not accepted, or a file that is too large becomes `rejected`.
  The service is not called for these files.
- If the service raises an error, the item becomes `failed`.
  The user sees a fixed message. The queue continues with the next file.
- The session keeps the bytes of a file only while the file is waiting, sending or failed.
  It drops the bytes after `stored`, `duplicate` or `rejected`.
- **Try again** puts a `failed` item back in the queue.
- **Clear finished** removes `stored`, `duplicate` and `rejected` items.
  It keeps `failed` items.
- When the user closes the tab, the session finishes the files that it already has.

### Limits

`ui/settings.py` holds the limits.

| Limit | Value |
|-------|-------|
| File types | JPEG, PNG, WebP |
| File size | 15 MB |

The browser checks these limits for a quick answer only.
The backend must check every upload again.

## Components

Each component does one job and has its own CSS file.

| Component | CSS file | Job |
|-----------|----------|-----|
| `app_header` | `header.css` | Product name, title, description |
| `upload_picker` | `picker.css` | Drop area and the buttons that open the chooser |
| `upload_queue` | `queue.css` | The list and the empty state |
| `upload_row` | `row.css` | One file |
| `status_badge` | `badge.css` | Text and colour of a status |
| `summary_bar` | `summary.css` | Counts and **Clear finished** |

`upload_page.py` connects the session to the components.
A row changes by itself when its item changes. The page does not draw the whole list again.

## Styling

- `static/css/tokens.css` holds all design values as CSS variables.
  It is the only file that contains colour values. A test enforces this rule.
- `static/css/base.css` holds the page column, the card, the buttons, the focus ring and the reduced-motion rule.
- Each component has one file in `static/css/components/`.
- Python code adds class names only. It does not set colours or sizes.
- Dark mode follows the system setting.
- The font is the system font. The page makes no request to the internet.
  This is important because the server can run on a network without internet.
- Text colours have a contrast of 4.5:1 or more in light and dark mode.
- Tap targets are at least 3 rem high.
- The page has no horizontal scroll at 390 px width.

### How to change the look

| Task | File |
|------|------|
| Change a colour | `tokens.css` |
| Change spacing or type size | `tokens.css` |
| Change one component | The CSS file of that component |
| Add a CSS file | Add the file, then add its path to `CSS_FILES` in `theme.py` |

## Known limits

- The page cannot show the upload progress of one file.
  It shows a status when the server has the file.
- The browser sends the files to the server. The queue limits how many files reach the service at a time.
  It does not limit how the browser sends them.
- Some phone browsers ignore the camera setting.
  **Take photo** then opens the normal file chooser.

## Tests

| File | What it tests |
|------|---------------|
| `tests/test_ui_session.py` | The upload queue: order, outcomes, errors, retry, limits, clear |
| `tests/test_ui_theme.py` | The CSS files exist, only `tokens.css` has colours, light and dark names match |
