# Backend

This is the backend module.
It controls the flow of the application. It connects the other modules in the right order.

The backend has four jobs:

1. The upload step. It accepts one image, checks it, stores it, and puts it in the queue.
2. The worker. It takes the photos and the images from the queue and runs the ai stages on them.
3. Enrollment. A carer adds the objects that the owner wants to find, with photos. The backend calibrates each object.
4. The memory. It matches the regions of a frame with the objects, and keeps the sightings and the last-seen records (see `docs/backend/Memory.md`).
5. The search. It turns the words of a user into an object, and answers where the object was seen last (see `docs/backend/Search.md`).
6. The maintenance jobs. They delete old images and orphan files, report missing files, and save the vector index.

`main.py` starts the application and stops it. It calls the backend after all other modules are ready.

For the full design, see `docs/backend/Upload Endpoint.md`, `docs/backend/Enrollment.md`, `docs/backend/Worker.md`,
`docs/backend/Memory.md`, `docs/backend/Maintenance.md` and `docs/backend/Lifecycle.md`.

## Files

| File | Purpose |
|------|---------|
| `api.py` | The public interface: the `Backend` class |
| `ingest.py` | `IngestService.submit`: the upload step. It uses `storage` and `db` |
| `http.py` | The HTTP route `POST /api/images` |
| `http_objects.py` | The HTTP routes of the gallery, `/api/objects` |
| `http_search.py` | The HTTP routes of the search: `POST /api/search`, `GET /api/objects/{id}/last-seen` and its picture |
| `http_common.py` | What both route files share: the token check, the body reader, the error answers |
| `enrollment.py` | `EnrollmentService`: objects, reference photos, the carer's confirmation, the calibration |
| `ui_adapter.py` | Connects the UI to the upload step, to enrollment and to the search (`ui.ports.UploadService`, `EnrollmentService`, `FindService`) |
| `memory.py` | `MemoryService`: matches the regions of a frame, writes the sightings, keeps the last-seen records |
| `verification.py` | `ReferenceVerifier`: the check of a sighting against the reference photos of an object (stage I5). The memory and the search use it |
| `search.py` | `SearchService`: answers "where is my object?" from the last-seen record, then from the archive of frames |
| `query.py` | `QueryResolver`: turns the words of a user into an enrolled object, or a short list to choose from |
| `describe.py` | `DescribeService`: makes the sentence for the voice (stage Q2). It chooses the room from neighbouring frames, the surface from the box, and builds the text |
| `voice.py` | `VoiceService`: checks a WAV recording and turns it into text with the speech recognizer (stage Q0) |
| `worker.py` | `Worker`: runs the ai stages on `pending` photos and images, in a background thread |
| `maintenance.py` | `Maintenance`: retention, orphan and missing files, the vector checkpoint |
| `instance_lock.py` | `InstanceLock`: only one copy of the application may use the data |
| `settings.py` | Reads `config.toml` |
| `errors.py` | The error classes |
| `config.toml` | The queue limit, the token variable, the worker, the enrollment limits, the server and the maintenance settings |

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

## Gallery routes

The routes under `/api/objects` create objects, add reference photos, show the crop of a photo,
and confirm, retry or remove a photo. They use the same optional token as the upload route.
The list of routes and codes is in `docs/backend/Enrollment.md`.

## Search routes

| Route | Does |
|-------|------|
| `POST /api/search` | The body is JSON: `{"query": "the blue thing I drink from"}`. The answer has `status` (`matched`, `ambiguous` or `none`), the `candidates` and, for `matched`, the `answer` |
| `GET /api/objects/{id}/last-seen` | Where and when the object was seen last |
| `GET /api/objects/{id}/last-seen/picture` | A JPEG of the answer. `?view=crop` (the default) or `frame`, and `?max_side=` from 32 to 1024 |

They use the same optional token and the same fixed messages as the other routes. An object that is a draft or a look-alike gives 409.
The pictures and the answers have `Cache-Control: no-store`. The words of a user are never logged.

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

| Key (table `[enrollment]`) | Meaning |
|----------------------------|---------|
| `enrollment.max_photos_per_object` | The most reference photos for one object |
| `enrollment.max_objects` | The most objects in the gallery |

| Key (table `[server]`) | Meaning |
|------------------------|---------|
| `server.host` | The address of the web server. `0.0.0.0` accepts the local network. `127.0.0.1` accepts only this computer |
| `server.port` | The port of the web server |
| `server.worker_stop_seconds` | At shutdown, the time that the server waits for the worker to finish its image |
| `server.tls_certfile`, `server.tls_keyfile` | Paths of the certificate and the key. Both set: HTTPS. Both empty (default): HTTP. See `docs/backend/HTTPS.md` |

| Key (table `[maintenance]`) | Meaning |
|-----------------------------|---------|
| `maintenance.interval_minutes` | The time between two maintenance runs |
| `maintenance.orphan_delete_limit` | The most orphan files that one run deletes. 0 never deletes one |
| `maintenance.retention_batch` | The number of expired images that are deleted in one group |

| Key (table `[memory]`) | Meaning |
|------------------------|---------|
| `memory.hysteresis_window_seconds` | A weak sighting counts the sightings of the object in this time window |
| `memory.hysteresis_hits` | The number of frames with a sighting that a weak sighting needs. A strong sighting needs none |
| `memory.verify_weak` | Check a weak sighting against the reference photos of the object (stage I5). A sighting that passes counts as strong. The check never rejects a sighting |
| `memory.verify_max_references` | The check compares with at most this many reference photos, the most similar first |

| Key (table `[search]`) | Meaning |
|------------------------|---------|
| `search.stale_after_hours` | A last-seen record is fresh for this time. After it, the search also looks in the archive |
| `search.window_minutes` | The archive is searched in windows of this size, newest first |
| `search.max_scan_hours` | The search does not look further back than this |
| `search.max_regions_per_window` | The most regions that one window holds |
| `search.max_frames` | The search compares at most this number of frames. The answer says if it stopped early |
| `search.verify_weak` | Check a weak hit in the archive against the reference photos (stage I5). A hit that passes counts as strong. The check never rejects a hit |
| `search.verify_max_references` | The check compares with at most this many reference photos, the most similar first |
| `search.crop_padding` | The picture of an answer shows the object with this much space around it (a part of the size of its box) |

| Key (table `[query]`) | Meaning |
|-----------------------|---------|
| `query.matched_min_score` | The best object is sure if its score is at least this value |
| `query.matched_min_gap` | ...and at least this value above the second best |
| `query.suggest_min_score` | Below `matched_min_score`, the user chooses between the objects with at least this score. Below it, no object fits |
| `query.max_candidates` | The most objects that the user chooses from |
| `query.max_chars` | A question with more characters is refused |
| `describe.neighbour_window_seconds` | Frames this close in time to the answer help to choose the room |
| `describe.neighbour_max_frames` | The most neighbouring frames that are used |
| `describe.room_min_frames` | The room is named only if at least this many frames have scores |
| `describe.room_agree_fraction` | ...and at least this part of them have the room as the best room |
| `describe.room_min_score` | ...and the average score of the room is at least this value |
| `describe.surface_min_score` | The surface is named only if its score is at least this value |
| `describe.time_zone` | The time zone for the part of the day (IANA name). Empty: the time zone of the computer |
| `describe.day_part_min_hours` | A sighting this old or older gets a part of the day ("yesterday evening") |
| `voice.min_seconds` | A recording shorter than this is refused |
| `voice.silence_rms` | A recording with a sound level below this value is silent. It gets no text |

The largest upload is `limits.max_bytes` in `storage/config.toml`.

## Worker

`Worker.process_next()` takes the oldest `pending` reference photo first (see `docs/backend/Enrollment.md`).
If there is none, it takes the oldest `pending` image and does these steps:
read and decode the file, check the quality, reject a blurry image or a near duplicate, detect candidates,
make the vectors, store the regions and the vectors, match the regions with the objects (the memory), and mark the image `done`.
The detector looks for the phrases of the `ready` objects of the gallery. Without a ready object, it uses the placeholder phrases of `ai/config.toml`.
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

app.include_router(backend.create_router())     # The HTTP routes
register_pages(backend.upload_service, backend.enrollment_service, backend.find_service)  # The UI
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
| `InvalidInputError` | A name or a phrase of an object is not valid |
| `LimitReachedError` | The gallery has the largest number of objects or photos |
| `NotFoundError` | An object or a photo does not exist |
| `InvalidQueryError` | The words of a user are empty, too long or not printable |
| `ObjectNotReadyError` | A search asks for an object that is a draft, or a look-alike |

The upload step also passes on the errors of the `storage` and `db` modules.
The route and the UI turn each of them into a fixed message without internal details.
