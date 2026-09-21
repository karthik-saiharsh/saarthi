# Worker

## Purpose

The worker takes the images that wait in the queue and runs the ai stages on them.
The upload step puts an image in the queue with the status `pending`. The worker takes it out.
When the worker is finished, the image is `done`, `rejected` or `failed`.

The worker does not make decisions about the models. It calls the helpers of the `ai`, `storage` and `db` modules in a fixed order.

## Where it fits

```
upload ──▶ db: image `pending` ──▶ Worker ──▶ db: `done` (regions and vectors) | `rejected` | `failed`
   │                                  ▲
   └─── wake() ───────────────────────┘
```

- The upload step calls `wake()` after it stores a new image (see `IngestService(on_stored=...)`).
- The worker also checks the queue every `worker.idle_wait_seconds`, as a safety net.

## One thread in the same process

The worker runs as one background thread of the application. It does not run in a separate process.
The `db` module keeps the vector index in memory. A second process would have a second copy of the index, and the two copies would differ.

The worker processes one image at a time. The models use the CPU, so more threads would not make it faster.
The db connection is per thread. The worker closes its connection when the thread ends.

## Steps for one image

`Worker.process_next()` takes the oldest `pending` image and does these steps:

| Step | Helper | Result |
|------|--------|--------|
| 1. Claim | `db.claim_next_pending` | The image is `processing`. No other worker can take it |
| 2. Read and decode | `storage.read_image`, `engine.open_image` | An RGB image |
| 3. Quality | `engine.assess_quality`, `db.set_image_analysis` | The blur score and the hash are saved |
| 4. Blur gate | `engine.is_blurry` | If true: the image is `rejected` with the reason `too blurry` |
| 5. Duplicate gate | `db.list_recent_phashes`, `engine.is_near_duplicate` | If true: the image is `rejected` with the reason `near duplicate of a recent image` |
| 6. Detect | `engine.detect_candidates` | A list of candidates |
| 7. Embed | `engine.embed_regions` | One vector for each candidate |
| 8. Store | `db.add_regions`, `db.store_region_vectors` | The regions and the vectors |
| 9. Finish | `db.mark_image_done` | The image is `done` |

An image with no candidates is also `done`, with no regions.

### Why the vectors come before the regions

Step 7 comes before step 8. The model work is the part that can fail. If it fails, no region row exists yet.
The image has no half-written regions.
After step 8, only a database error can still fail. The `store_region_vectors` helper does both of its steps or none.

### Near duplicates

The worker compares the hash of the image with the hashes of the `done` images that were captured in the last
`worker.duplicate_window_minutes`. It compares with at most `worker.duplicate_max_compared` images.
Images that are `rejected` or `failed` are not compared.
The number of bits that count as a near duplicate is `quality.phash_max_distance` in `ai/config.toml`.

### Generic pass

The generic pass ("an object") costs one more detector run. The worker runs it on every Nth image that reaches the detector.
N is `worker.generic_pass_every`. The value 0 turns the pass off.
The counter starts at 0 each time the application starts. It does not count rejected images.

## Failures

- Any error in one image marks the image `failed`, and the worker goes on with the next image.
- The reason in `status_detail` is only the name of the error class, for example `InferenceError` or `StorageNotFoundError`.
  An error message can hold a path or other data, so the worker never stores it. The full error is in the log.
- A failed image is not tried again. A crash is different: the start-up recovery sets an image that was `processing` back to `pending`.
- If a database error happens after step 8 starts, the image can keep regions without vectors. The image is `failed`, so the
  start-up recovery does not touch it. Such regions are not in the vector index, so searches do not see them.
- If the worker cannot mark an image as failed, it writes an error to the log. The image stays `processing` until the next start.
- An error in the loop itself (for example, a busy database when the worker claims an image) is logged.
  The worker waits `worker.error_pause_seconds` and goes on. The thread does not stop.

## Start and stop

```python
worker = Worker(config, engine)
ingest = IngestService(config, on_stored=worker.wake)
worker.start()
...
worker.stop(timeout=60)
```

- `start()` starts the thread. A second call raises `RuntimeError`.
- `stop()` asks the thread to stop. The worker finishes the image that it holds. This can take as long as one image:
  about 10 seconds on the development machine, more on the Raspberry Pi. When the worker is idle, `stop()` returns at once.
- If the application is killed in the middle of an image, the start-up recovery repairs the data at the next start.
- `process_next()` can be called without the thread. The tests do this.

## Configuration

`backend/config.toml`, table `[worker]`:

| Key | Default | Description |
|-----|---------|-------------|
| `idle_wait_seconds` | 5 | How long the worker waits for a wake-up signal when no image waits |
| `generic_pass_every` | 10 | Run the generic pass on every Nth analysed image. 0 never runs it |
| `duplicate_window_minutes` | 60 | Compare with the images that were captured in this time |
| `duplicate_max_compared` | 200 | Compare with at most this number of images |
| `error_pause_seconds` | 5 | The pause after an error in the loop |

The thresholds of the stages are in `ai/config.toml`. A value of `quality.blur_min = 0` turns the blur gate off.
This is useful until the threshold is set with real frames.

## Logging

The log format is `{module}-{time}: {message}`.

| Level | Event |
|-------|-------|
| INFO | The worker starts and stops. An image is `done` (with the number of regions). The models report counts and times |
| ERROR | An image failed (the error class only). An image cannot be marked as failed |
| Exception with trace | An unexpected error in an image or in the loop |

The log does not show image content or file names.

## Known limits

- The speed is about 10 seconds for each image on the development machine. A frame that arrives faster than this waits in the queue.
  The upload step refuses new images when `ingest.max_pending` images wait. The speed on the Raspberry Pi is not measured.
- The blur threshold is not set with real frames. The start value of 100 rejects the test photo.
- The worker does not retry a failed image. A helper to retry failed images does not exist.
- The worker gives no progress report. The number of `pending` images is the only sign of the load.
- The `Backend` class does not create the worker yet. `main.py` must create it with the ai engine and connect `wake` to the upload step.
- The vocabulary of the detector is a placeholder (see `ai/docs/AI Ingestion Module.md`).

## Tests

`tests/test_backend_worker.py` uses the real storage, database and index. The ai models are fakes.
The quality gate, the image decoding and the post-processing are the real code.

| Test | What it checks |
|------|----------------|
| An image is analysed | `done`, regions with vectors, the blur score and the hash |
| No candidates, nothing to do | `done` with no regions, and `process_next` returns `False` |
| The blur gate | A flat image is `rejected` and the detector is not called |
| The duplicate gate | A near-duplicate is `rejected`. Different pictures are not |
| The generic pass | The pattern for N = 2, 1 and 0 |
| Failures | A detector error, an embedder error (no regions left), a missing file, an unexpected error. The next image still works |
| The thread | A wake-up works with a long wait. Stop is fast. No double start. The loop survives an error |

The worker has not yet run with the real models. The first full run is part of the `main.py` step.
