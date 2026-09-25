# UI

This is the UI module.
All UI logic lives here.
The module uses `nicegui`.

The UI does not contain application logic.
It shows state and calls the application through `ports.py`.
The UI does not import `backend`, `db`, `ai` or `storage`.

For the full design, see `docs/ui/Upload Page.md`.

## Files

| Path | Purpose |
|------|---------|
| `ports.py` | The interface to the backend: `UploadService`, `EnrollmentService` and their data types |
| `stub_service.py` | Fake services for work on the UI: `StubUploadService`, `StubEnrollmentService` |
| `models.py` | `UploadItem` and `ItemStatus` |
| `settings.py` | Accepted file types and the maximum file size |
| `upload_session.py` | The upload queue of one browser tab. It has no NiceGUI code |
| `enroll_session.py` | The objects and photos of one browser tab. It has no NiceGUI code |
| `routes.py` | The routes of the pages |
| `theme.py` | Serves the CSS files and adds them to a page |
| `registry.py` | `register_pages(upload_service, enrollment_service)` |
| `pages/` | One file for each page |
| `components/` | One file for each component |
| `static/css/` | `tokens.css`, `base.css` and one file for each component |

## Pages

| Route | Page |
|-------|------|
| `/upload` | Upload pictures, one at a time or as a batch |
| `/enroll` | Add objects, add photos of them, and check the crop of each photo |

## Use

```python
from ui.registry import register_pages
from ui.stub_service import StubEnrollmentService, StubUploadService

register_pages(StubUploadService(), StubEnrollmentService())
```

The caller starts the server (`ui.run`).
This module does not start the server.

The real services are `Backend.upload_service` and `Backend.enrollment_service` (see `backend/README.md`).
The stub is for work on the UI only.

To change the colours, edit `static/css/tokens.css`.
Only this file can contain colour values.
A test enforces this rule.
