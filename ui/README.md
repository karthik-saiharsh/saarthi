# UI

This is the UI module.
All UI logic lives here.
The module uses `nicegui`.

The UI does not contain application logic.
It shows state and calls the application through `ports.py`.
The UI does not import `backend`, `db`, `ai` or `storage`.

For the full design, see `docs/ui/Upload Page.md`, `docs/ui/Enrollment Page.md`, `docs/ui/Find Page.md` and `docs/ui/Memory Page.md`.

## Files

| Path | Purpose |
|------|---------|
| `ports.py` | The interface to the backend: `UploadService`, `EnrollmentService`, `FindService`, `VoiceService`, `BrowseService` and their data types |
| `stub_service.py` | Fake services for work on the UI: `StubUploadService`, `StubEnrollmentService`, `StubFindService` |
| `models.py` | `UploadItem` and `ItemStatus` |
| `settings.py` | Accepted file types and the maximum file size |
| `upload_session.py` | The upload queue of one browser tab. It has no NiceGUI code |
| `browse_session.py` | The filter, the page of frames, the pictures and the open frame of the memory page of one tab. It has no NiceGUI code |
| `find_session.py` | The objects, the busy flag and the result of the search of one browser tab. It has no NiceGUI code |
| `enroll_session.py` | The objects and photos of one browser tab. It has no NiceGUI code |
| `routes.py` | The routes of the pages |
| `theme.py` | Serves the CSS files and adds them to a page |
| `registry.py` | `register_pages(upload_service, enrollment_service, find_service, voice_service=None, browse_service=None)`. Without a voice service the find page has no Talk button. Without a browse service there is no memory page |
| `speech.py` | The voice of the page: the browser says the answer with a local voice of the device |
| `voice_route.py` | The route `POST /ui/voice`: the recording of the Talk button goes to the voice service |
| `pages/` | One file for each page |
| `components/` | One file for each component |
| `static/css/` | `tokens.css`, `base.css` and one file for each component |
| `static/js/talk.js` | Records the Talk button in the browser and makes a WAV file (16 kHz, 16-bit, mono) |

## Pages

| Route | Page |
|-------|------|
| `/find` | Choose an object, or write what you look for. The page shows where and when the object was seen |
| `/upload` | Upload pictures, one at a time or as a batch |
| `/memory` | The frames that Saarthi took in, with what the pipeline found in each. Tap a frame for the boxes, the matches and the room (`docs/ui/Memory Page.md`) |
| `/enroll` | Add objects, add photos of them, check the crop of each photo, mark a small object, and answer "Is this your object?" for boxes of recent frames |

## Use

```python
from ui.registry import register_pages
from ui.stub_service import StubEnrollmentService, StubFindService, StubUploadService

register_pages(StubUploadService(), StubEnrollmentService(), StubFindService())
```

The caller starts the server (`ui.run`).
This module does not start the server.

The real services are `Backend.upload_service`, `Backend.enrollment_service`, `Backend.find_service`, `Backend.voice_service` and `Backend.browse_service` (see `backend/README.md`).
The stub is for work on the UI only.

To change the colours, edit `static/css/tokens.css`.
Only this file can contain colour values.
A test enforces this rule.
