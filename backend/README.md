# Backend

This is the backend module.
It controls the flow of the application. It connects the other modules in the right order.

At this time, the backend has one job: the upload step.
It accepts one image, checks it, stores it, and puts it in the queue for the worker.
The worker loop, the retention job and the start-up steps are not built yet.

For the full design, see `docs/backend/Upload Endpoint.md`.

## Files

| File | Purpose |
|------|---------|
| `api.py` | The public interface: the `Backend` class |
| `ingest.py` | `IngestService.submit`: the upload step. It uses `storage` and `db` |
| `http.py` | The HTTP route `POST /api/images` |
| `ui_adapter.py` | Connects the UI to the upload step (`ui.ports.UploadService`) |
| `settings.py` | Reads `config.toml` |
| `errors.py` | The error classes |
| `config.toml` | The queue limit and the name of the token variable |

## Rules

- The backend is the only place where `storage`, `db`, `ai` and `ui` meet.
- The backend uses `api.py` of the other modules only.
- The UI uses the backend only, through `ui/ports.py`.
- The backend does not start the other modules. `main.py` does that, and it calls the backend after the start-up.

## Upload route

```
POST /api/images
```

The body of the request is the image (JPEG, PNG or WebP). The `Content-Type` header is not used.

```
curl -X POST --data-binary @photo.jpg http://<server>:<port>/api/images
```

With a token (see below):

```
curl -X POST -H "Authorization: Bearer $SAARTHI_API_TOKEN" --data-binary @photo.jpg http://<server>:<port>/api/images
```

| Code | Meaning |
|------|---------|
| 201 | A new image is stored. The answer is `{"status": "stored", "image_id": 1}` |
| 200 | The same image was stored before. The answer is `{"status": "duplicate", "image_id": 1}` |
| 400 | The body is empty, or `Content-Length` is not valid |
| 401 | The token is missing or wrong |
| 413 | The upload has too many bytes or too many pixels |
| 415 | The data is not a usable JPEG, PNG or WebP image |
| 503 | Too many images wait for the worker. Try again after the time in `Retry-After` |
| 507 | The disk of the server is full |
| 500 | Another error |

## Configuration

Edit `config.toml`.

| Key | Meaning |
|-----|---------|
| `ingest.max_pending` | New uploads are refused when this many images wait for the worker |
| `api.token_env_var` | The name of the environment variable that holds the access token |

If the environment variable is set, every request needs `Authorization: Bearer <token>`.
If it is not set, the route is open on the local network.
A variable that is set but empty stops the start, so the route cannot open by mistake.

The largest upload is `limits.max_bytes` in `storage/config.toml`.

## Usage

```python
import db.api as db
import storage.api as storage
from backend.api import Backend

storage.init_storage()
db.init_db()
backend = Backend()

app.include_router(backend.create_router())     # The HTTP route
register_pages(backend.upload_service)          # The UI
```

## Errors

| Error | Meaning |
|-------|---------|
| `BackendError` | Base class of the errors below |
| `BackendConfigError` | `config.toml` or the token variable is not correct |
| `QueueFullError` | Too many images wait for the worker |

The upload step also passes on the errors of the `storage` and `db` modules.
The route and the UI turn each of them into a fixed message without internal details.
