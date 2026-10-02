# Application Lifecycle

## Purpose

`main.py` owns the life of the application: how it starts, how it runs, and how it stops.
The other modules do not start themselves. They give helpers, and `main.py` calls them in the right order.

Run the application with:

```
uv run python main.py
```

The server listens on `server.host` and `server.port` of `backend/config.toml` (`0.0.0.0` and `8080` by default).

| Address | What it is |
|---------|------------|
| `/` | A redirect to `/upload` |
| `/upload` | The upload page for people |
| `/find` | The page where a person asks where an object is (see `docs/ui/Find Page.md`) |
| `/enroll` | The page where a carer adds the objects (see `docs/ui/Enrollment Page.md`) |
| `POST /api/images` | The upload route for the wearable and for scripts (see `docs/backend/Upload Endpoint.md`) |
| `/api/objects` | The routes of the gallery (see `docs/backend/Enrollment.md`) |
| `POST /api/search` and `GET /api/objects/{id}/last-seen` | The routes of the search (see `docs/backend/Search.md`) |

## Start-up order

Each step needs the step before it. If a step fails, the application prints one line and ends with the exit code 1.
The server never starts.

| Step | What happens | Why it is in this place |
|------|--------------|-------------------------|
| 1. Storage | `storage.init_storage()`: read the config, prepare the root folder, remove old temporary files | The upload step and the worker need the folder |
| 2. Database | `db.init_db()`: open SQLite, create the tables, open the vector index | Everything else reads the database |
| 3. Instance lock | Take an exclusive lock on `db/data/saarthi.lock` | Only one copy may use the data. A second copy stops here (see below) |
| 4. Recovery | `db.startup_recovery()`: repair what a stop or a crash left behind | It changes data, so the lock comes first |
| 5. Ai models | `AiEngine()`: load the detector, the embedder, the query encoder and the verifier | The first start downloads about 925 MB (the detector 690 MB, the embedder 90 MB, the query encoder 91 MB, the verifier 55 MB). Later starts take about 7 seconds |
| 6. Backend | `Backend(engine)`: the upload step, the worker and the maintenance jobs | It checks that the size of the model vectors equals `vectors.dimension` |
| 7. Web server | The upload page and the upload route start | The worker and the maintenance thread start with the server |

The threads (worker and maintenance) start when the web server starts, not before.
In this way, nothing processes an image before the server can answer.

## Shutdown order

Press Ctrl+C, or send the signal `SIGTERM` (a service manager does this). The application does these steps:

1. The maintenance thread stops. A run that is in progress finishes first.
2. The worker stops. It finishes the image that it holds. This takes up to `server.worker_stop_seconds` (120 by default).
3. The vector index is saved to disk.
4. The database connection is closed.
5. The instance lock is released.

A clean shutdown loses no vectors. Both signals were tested with the real application.

## One copy only

Two copies on the same data damage each other. This was found in a real test:

- The recovery of the second copy reset the image that the first copy was processing. The first copy then failed on that image and processed it again.
- The maintenance of the second copy saved its own empty vector index over the file of the first copy.

The instance lock prevents this. The application takes an exclusive lock (`flock`) on `db/data/saarthi.lock`
after the database opens and before the recovery.

- A second copy stops at once, before it loads the models, with the exit code 1 and this message:
  `Another copy of Saarthi is running (process 40054). It holds .../db/data/saarthi.lock`.
- The operating system frees the lock when the process ends, also after a crash or `kill -9`.
  A lock file that stays behind causes no problem.
- The lock is for Linux and macOS. The Raspberry Pi runs Linux.

## After a crash

A crash is a stop without a clean shutdown (a power loss, `kill -9`).

- An image that was `processing` stays in that status. The recovery at the next start sets it back to `pending`.
- The vector index on disk has the vectors of the last save. Newer vectors are lost.
  The index is saved at a clean shutdown, once an hour (the maintenance job), and after every
  `vectors.checkpoint_every` new vectors (200 by default).
  The recovery finds the regions that lost their vectors, and sets their images back to `pending`.
  The worker processes them again.

A real test shows this. The application was stopped with `kill -9` while it processed an image:

| | Before the kill | After the restart |
|--|-----------------|-------------------|
| Image in `processing` | 1 | 0 (set back to `pending`, then `done`) |
| Vectors on disk | 11 | 31 |
| Embedded regions in the database | 21 | 31 |

The recovery cleared the 10 marks without a vector and set that image back to `pending` too.
It reported: `1 interrupted image(s), 1 requeued image(s), 10 missing vector(s) cleared`.
After the work, the index and the database had the same 31 vectors with the same ids.

The cost of a crash is the work of the lost images, about 8 seconds for each image on the development machine.
Lower `vectors.checkpoint_every` in `db/config.toml` if a crash is likely.

## Logs

Each module writes its own lines in the form `{module}-{time}: {message}`.
A normal start shows these lines (shortened):

```
main-...: Starting storage
db.api-...: Start-up recovery: 0 interrupted image(s), 0 requeued image(s), 0 orphan vector(s) removed, 0 missing vector(s) cleared
main-...: Loading the ai models (the first start downloads them)
ai.api-...: AI engine ready: detector grounding-dino-tiny@a2bb814d, embedder dinov2-small@ed25f3a3
backend.api-...: Backend ready. Upload access token: not required
main-...: Serving on http://0.0.0.0:8080 (upload page: /upload)
backend.worker-...: Worker started
backend.maintenance-...: Maintenance started. It runs every 60 minute(s)
```

The line about the token tells you if the upload route is open (see `docs/backend/Upload Endpoint.md`).

## What the first real run showed

The application ran with the real models, on the development machine (a CPU, no GPU), with the test photo.

- The start took about 8 seconds, of which about 7.5 seconds were the models.
- The photo was **rejected** as "too blurry". Its blur score is 94.2, and the minimum is 100.
  The photo looks normal. The threshold needs real frames from the wearable.
- With `blur_min = 0`, a copy of the photo was `done` in about 8.4 seconds (the detector 7.6 seconds, the vectors 0.7 seconds), with 11 regions.
- The same photo saved again with another JPEG quality had a blur score of 117.1. The score depends on the JPEG compression.
- A near duplicate of an image that is `done` was rejected in milliseconds.
- The same photo sent twice gave the answer 200 (duplicate) the second time.

## Known limits

- The speed on the Raspberry Pi is not measured. The application was run only on the development machine.
- The web server shows no health or status page. The number of `pending` images is the only sign of the load.
- The application does not start itself at boot. Use a service manager (for example `systemd`) for this.
- The upload route is open on the local network unless you set the token.
- The shutdown can take up to `server.worker_stop_seconds`, because the worker finishes its image.

## Tests

| File | What it tests |
|------|---------------|
| `tests/test_main.py` | The start order, a wrong vector size, `attach`, an upload that is analysed with the vectors saved at shutdown, the exit code, the pages, the lock |
| `tests/test_backend_instance_lock.py` | The lock, including a lock held by another process and a crash that frees it |
