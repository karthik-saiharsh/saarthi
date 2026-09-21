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
| `ports.py` | The interface to the backend: `UploadService`, `UploadOutcome` |
| `stub_service.py` | A fake `UploadService` for work on the UI |
| `models.py` | `UploadItem` and `ItemStatus` |
| `settings.py` | Accepted file types and the maximum file size |
| `upload_session.py` | The upload queue of one browser tab. It has no NiceGUI code |
| `theme.py` | Serves the CSS files and adds them to a page |
| `registry.py` | `register_pages(service)` |
| `pages/` | One file for each page |
| `components/` | One file for each component |
| `static/css/` | `tokens.css`, `base.css` and one file for each component |

## Pages

| Route | Page |
|-------|------|
| `/upload` | Upload pictures, one at a time or as a batch |

## Use

```python
from ui.registry import register_pages
from ui.stub_service import StubUploadService

register_pages(StubUploadService())
```

The caller starts the server (`ui.run`).
This module does not start the server.

The real service is `Backend.upload_service` (see `backend/README.md`).
The stub is for work on the UI only.

To change the colours, edit `static/css/tokens.css`.
Only this file can contain colour values.
A test enforces this rule.
