# Backend

This is the backend module.
It controls the flow of the application. It connects the other modules in the right order.

The backend has three jobs:

1. The upload step. It accepts one image, checks it, stores it, and puts it in the queue.
2. The worker. It takes the images from the queue and runs the ai stages on them.
3. The maintenance jobs. They delete old images and orphan files, report missing files, and save the vector index.

`main.py` starts the application and stops it. It calls the backend after all other modules are ready.

For the full design, see `docs/backend/Upload Endpoint.md`, `docs/backend/Worker.md`, `docs/backend/Maintenance.md`
and `docs/backend/Lifecycle.md`.

## Files

| File | Purpose |
|------|---------|
| `api.py` | The public interface: the `Backend` class |
| `ingest.py` | `IngestService.submit`: the upload step. It uses `storage` and `db` |
| `http.py` | The HTTP route `POST /api/images` |
| `ui_adapter.py` | Connects the UI to the upload step (`ui.ports.UploadService`) |
| `worker.py` | `Worker`: runs the ai stages on `pending` images, in a background thread |
| `maintenance.py` | `Maintenance`: retention, orphan and missing files, the vector checkpoint |
| `instance_lock.py` | `InstanceLock`: only one copy of the application may use the data |
| `settings.py` | Reads `config.toml` |
| `errors.py` | The error classes |
| `config.toml` | The queue limit, the token variable, the worker, the server and the maintenance settings |

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

| Key (table `[worker]`) | Meaning |
|------------------------|---------|
| `worker.idle_wait_seconds` | How long the worker waits for a wake-up signal when no image waits |
| `worker.generic_pass_every` | Run the generic pass on every Nth analysed image. 0 never runs it |
| `worker.duplicate_window_minutes` | Compare the hash of an image with the images from this time |
| `worker.duplicate_max_compared` | Compare with at most this number of images |
| `worker.error_pause_seconds` | The pause after an error in the loop of the worker |

| Key (table `[server]`) | Meaning |
|------------------------|---------|
| `server.host` | The address of the web server. `0.0.0.0` accepts the local network. `127.0.0.1` accepts only this computer |
| `server.port` | The port of the web server |
| `server.worker_stop_seconds` | At shutdown, the time that the server waits for the worker to finish its image |

| Key (table `[maintenance]`) | Meaning |
|-----------------------------|---------|
| `maintenance.interval_minutes` | The time between two maintenance runs |
| `maintenance.orphan_delete_limit` | The most orphan files that one run deletes. 0 never deletes one |
| `maintenance.retention_batch` | The number of expired images that are deleted in one group |

The largest upload is `limits.max_bytes` in `storage/config.toml`.

## Worker

`Worker.process_next()` takes the oldest `pending` image and does these steps:
read and decode the file, check the quality, reject a blurry image or a near duplicate, detect candidates,
make the vectors, store the regions and the vectors, and mark the image `done`.
An error marks the image `failed`. The reason is only the name of the error class.
`start()` runs the worker as a background thread. An upload calls `wake()`, so a new image starts at once.
`stop()` finishes the current image and stops the thread.

```python
worker = Worker(config, engine)
ingest = IngestService(config, on_stored=worker.wake)
worker.start()
worker.stop(timeout=60)
```

The `Backend` class creates the worker and connects `wake` to the upload step.
`Backend.start()` starts the worker and the maintenance thread. `Backend.stop()` stops them.

## Usage

`main.py` starts the modules in this order (see `docs/backend/Lifecycle.md`):

```python
storage.init_storage()
db.init_db()
# ... the instance lock and db.startup_recovery() ...
engine = AiEngine()
backend = Backend(engine)          # It checks the size of the vectors

app.include_router(backend.create_router())     # The HTTP route
register_pages(backend.upload_service)          # The UI
app.on_startup(backend.start)                   # The worker and the maintenance thread
app.on_shutdown(lambda: stop_components(components))
# stop_components stops the backend, saves the vector index, closes the database and releases the lock
```

## Errors

| Error | Meaning |
|-------|---------|
| `BackendError` | Base class of the errors below |
| `BackendConfigError` | `config.toml` or the token variable is not correct |
| `QueueFullError` | Too many images wait for the worker |
| `InstanceRunningError` | Another copy of the application already uses the data |

The upload step also passes on the errors of the `storage` and `db` modules.
The route and the UI turn each of them into a fixed message without internal details.
