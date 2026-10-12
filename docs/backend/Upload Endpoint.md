# Upload Endpoint

## Purpose

The upload endpoint accepts the images from the wearable, one at a time.
It checks each image, stores it without metadata, and puts it in the queue for the worker.
The same upload step also serves the upload page of the UI.

The endpoint does not analyse the image. The worker does that later (see `docs/backend/Worker.md`).

## Where it fits

```
wearable or script ──HTTP──▶  POST /api/images ──┐
                                                   ├──▶ IngestService.submit ──▶ storage + db
UI upload page ──▶ UiUploadService (same process) ─┘
```

The UI does not use HTTP. It calls the upload step in the same process, through `ui/ports.py`.
This is why the browser upload and the wearable upload give the same results.

## The upload step

`IngestService.submit(data)` does these steps in this order:

1. **Queue check.** It counts the images with the status `pending`.
   If the number is `ingest.max_pending` or more, it raises `QueueFullError`.
   The worker of a slow device (the Raspberry Pi) cannot clear a long queue, so the server refuses new images early.
   A duplicate is also refused while the queue is full, because this check comes first.
2. **Prepare.** `storage.prepare_image` checks the data and makes a JPEG without metadata (see `docs/storage/Storage Module.md`).
3. **Duplicate check.** `db.find_image_by_sha256`. If the hash is known, the step returns `DUPLICATE` with the id of the existing image. Nothing is written.
4. **Save.** `storage.save_image` writes the file and gives a relative path.
5. **Add the row.** `db.add_image` adds the row with the status `pending`.
   If this fails, the step deletes the file again, so no file stays without a row.

If two uploads of the same picture arrive at the same time, both can pass step 3.
The unique hash in the database lets only one row in. The second upload deletes its own file and returns `DUPLICATE`.

6. **Wake the worker.** For a new image, the step calls the `on_stored` hook (the worker's `wake`).
   An error in the hook is logged. It does not fail an upload that is already stored.

The step returns `STORED` or `DUPLICATE` with the id of the image. Other results are errors.

## HTTP route

`POST /api/images`. The body of the request is the image. The `Content-Type` header is not used.
The storage module finds the format (JPEG, PNG or WebP) from the content.

### Answers

All answers are JSON.

| Code | Body | When |
|------|------|------|
| 201 | `{"status": "stored", "image_id": 1}` | A new image is stored |
| 200 | `{"status": "duplicate", "image_id": 1}` | The same image was stored before |
| 400 | `{"error": "..."}` | The body is empty, or `Content-Length` is not valid |
| 401 | `{"error": "..."}` and the header `WWW-Authenticate: Bearer` | The token is missing or wrong |
| 413 | `{"error": "..."}` | The upload has too many bytes, or the image has too many pixels |
| 415 | `{"error": "..."}` | The data is not a usable JPEG, PNG or WebP image |
| 503 | `{"error": "..."}` and the header `Retry-After: 30` | The queue is full |
| 507 | `{"error": "..."}` | The disk is full, or has less free space than `limits.min_free_bytes` |
| 500 | `{"error": "..."}` | Another error |

The error texts are fixed. They never show a path, an SQL text or an exception text.

### Examples

```
curl -i -X POST --data-binary @photo.jpg http://localhost:8080/api/images
curl -i -X POST -H "Authorization: Bearer $SAARTHI_API_TOKEN" --data-binary @photo.jpg http://localhost:8080/api/images
```

A client should treat 200 and 201 as success. After a 503, it should wait for the time in `Retry-After` and send the same image again.
After a 413, 415 or 400, the same image will never work, so the client should not send it again.

### Size limit

The largest body is `limits.max_bytes` in `storage/config.toml` (15 MB by default).
The route checks the `Content-Length` header first. If it is larger than the limit, the route refuses at once and does not read the body.
If the client sends no `Content-Length` (a chunked body), the route reads the body in chunks and stops when it passes the limit.
A big upload therefore never fills the memory.

### Access token

- If the environment variable `SAARTHI_API_TOKEN` is set, every request needs the header `Authorization: Bearer <token>`.
  The scheme name is not case-sensitive.
- If the variable is not set, the route is open on the local network.
- A variable that is set but empty stops the start of the backend. An empty value is a mistake, and the route must not open by accident.
- The route compares the token in constant time. It checks the token before it reads the body, so a request without a token costs the server almost nothing.
- The name of the variable is `api.token_env_var` in `backend/config.toml`.
- The token is never written to the log.

The owner chose an optional token. The device is used on a local network only, so the token is off by default.
Switch it on with the environment variable. You do not need a code change.

### Threads

Decoding an image, writing a file and using the database take time.
The route runs the upload step in the thread pool, so the server can answer other requests in the meantime.
The UI adapter does the same with `asyncio.to_thread`. SQLite works in these threads: each thread has its own connection.

## UI adapter

`UiUploadService.submit_image(filename, content_type, data)` implements `ui.ports.UploadService`.
It calls the upload step and turns the result into an outcome for the upload page.
It does not use or store the file name or the content type of the browser.

| Result of the upload step | Outcome for the UI | Text |
|---------------------------|--------------------|------|
| New image | `stored` | (none) |
| Duplicate | `duplicate` | This image is already stored. |
| Not a usable image | `rejected` | This file is not a usable image. |
| Too large | `rejected` | This file is too large. |
| Queue full | `failed` | The server is busy. Try again later. |
| Disk full | `failed` | The disk of the server is full. |
| Any other error | `failed` | The server could not store the image. |

A `failed` row shows the button **Try again** on the page.

## Configuration

`backend/config.toml`:

| Key | Default | Description |
|-----|---------|-------------|
| `ingest.max_pending` | 500 | Refuse new uploads when this many images wait. At about 10 seconds for each image, 500 images are about 80 minutes of work. Set the value for your device |
| `api.token_env_var` | `SAARTHI_API_TOKEN` | The name of the environment variable that holds the token |

## Security and privacy

- The route reads the body in chunks and stops at the limit.
- The route does not trust the content type. The storage module checks the content.
- The image is saved without metadata, with a random name.
- The answers contain no internal details. The log contains counts and ids, not image content and not the token.
- The route has no rate limit. The size limit, the queue limit and the free-space limit protect the device.
- A client that sends its body very slowly can hold a connection open. The timeouts of the web server are the protection.

## Not in the endpoint

- The endpoint shows no status of the queue. The number of `pending` images is in the database only.
- The start-up and the shutdown are in `main.py` (see `docs/backend/Lifecycle.md`).
  The worker and the maintenance jobs are in `docs/backend/Worker.md` and `docs/backend/Maintenance.md`.

## Tests

| File | What it tests |
|------|---------------|
| `tests/test_backend_settings.py` | The config file |
| `tests/test_backend_ingest.py` | The upload step with the real storage and database: stored, duplicate, refused data, full queue, rollback, the race |
| `tests/test_backend_http.py` | The route: every answer, the size limit with and without `Content-Length`, the token |
| `tests/test_backend_api.py` | The UI adapter, the `Backend` class, and the UI queue with the real backend |
