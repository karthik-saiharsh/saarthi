# Ingestion Database Schema

## Purpose

This document describes the database for the image ingestion pipeline.
The pipeline runs on a local Raspberry Pi server.
A client sends images one at a time.
The server saves each image on disk without its metadata (EXIF).
The server then records its own metadata in a SQLite database.

This design covers the ingestion data only.
The object gallery and the memory (the sighting log and the last-seen table) have their own sections below.

## Design rules

- The server stores the vectors in a FAISS index, not in SQLite.
- The db module owns both stores. The `backend` module uses `db/api.py` only.
- `Region.id` is the vector id in the FAISS index. This is the only link between the two stores.
- All timestamps are UTC.
- `captured_at` is the time at which the server received the image. It is not the time in the image metadata.
- The database refuses wrong values. `CHECK` constraints limit `status` and `source`, and they check the image size and the box coordinates.

## Table `image`

One row for each uploaded image.

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer, key | Image id |
| `file_path` | text, unique | Path of the file, relative to the storage root. Never absolute |
| `sha256` | text(64), unique | Hash of the stored file. It rejects exact duplicate uploads |
| `phash` | text(16), index, null | Perceptual hash for near-duplicate checks. Null until computed |
| `width` | integer | Width in pixels. Must be more than 0 |
| `height` | integer | Height in pixels. Must be more than 0 |
| `captured_at` | datetime, index | Time at which the server received the image |
| `room` | text, null | Name of the place |
| `blur_score` | float, null | Laplacian variance. Null until computed |
| `status` | text, index | Processing state. See below |
| `status_detail` | text, null | Reason for a reject or text of an error. It must not contain image content or personal data |
| `processed_at` | datetime, null | Time at which processing ended |

### Image status

| Value | Meaning |
|-------|---------|
| `pending` | The image is on disk. Nobody has analysed it yet |
| `processing` | A worker analyses the image |
| `done` | The regions and the vectors are written |
| `rejected` | A quality gate dropped the image (for example blur or duplicate) |
| `failed` | An error stopped the processing |

## Table `region`

One row for each detected box in an image.

| Column | Type | Description |
|--------|------|-------------|
| `id` | integer, key | Region id. It is also the FAISS vector id |
| `image` | foreign key, index | Parent image. The database deletes the region when it deletes the image |
| `x0`, `y0` | integer | Top-left corner in pixels. Must be 0 or more |
| `x1`, `y1` | integer | Bottom-right corner in pixels. `x1` must be more than `x0`. `y1` must be more than `y0` |
| `det_score` | float | Detector score. Use it for analysis only. Do not use it to decide identity |
| `source` | text | Detector pass that made the box. See below |
| `phrases` | JSON list | Detector labels that remain after the merge of duplicate boxes |
| `in_hand` | boolean | True when a hand covers the object |
| `quality_ok` | boolean | False when the crop failed the size, blur or edge check, or lies inside a screen, a picture frame or a mirror |
| `embedding_model` | text, index, null | Name of the model that made the vector. Null means the vector is not in the FAISS index |

### Region source

| Value | Meaning |
|-------|---------|
| `full` | Full-frame pass |
| `tile` | Tiled pass for small objects |
| `generic` | Generic "an object" pass |

## Gallery tables

The gallery holds the objects that the owner wants to find (see `docs/backend/Enrollment.md`).
The tables are `enrolled_object`, `reference_photo` and `object_view`.

### Table `enrolled_object`

| Column | Meaning |
|--------|---------|
| `name` | The name that the carer gave. Unique, at most 80 characters |
| `category_phrase` | The words for the detector, for example `a bottle` |
| `synonyms` | JSON list of more phrases for the same kind of object |
| `tracked` | `false` means a look-alike. The system never reports it |
| `status` | `draft` or `ready` |
| `tau`, `tau_strong`, `margin_min` | The calibration. Null until the object is `ready` |
| `calibration_model` | The model whose vectors the calibration used |
| `calibration_note` | A short text for the carer: what is missing, or a warning |
| `created_at`, `updated_at` | Naive UTC times |

### Table `reference_photo`

| Column | Meaning |
|--------|---------|
| `enrolled_object` | The object. Delete cascades to the photo |
| `file_path` | Relative to the storage root. Unique |
| `sha256` | Hash of the stored file. Unique together with the object |
| `width`, `height` | Size in pixels |
| `status` | `pending`, `processing`, `located`, `confirmed` or `failed` |
| `status_detail` | The reason of a failure. No personal data |
| `x0`, `y0`, `x1`, `y1`, `det_score` | The box of the object. Null until the object is located |

### Table `object_view`

One row for each vector of a photo.

| Column | Meaning |
|--------|---------|
| `enrolled_object`, `photo` | The owners. Delete cascades |
| `vector` | `float32` numbers of unit length, as bytes. The length is `vectors.dimension` |
| `embedding_model` | The model that made the vector. Vectors of different models cannot be compared |
| `augmentation` | How the view was made, for example `original` |

The gallery vectors are in SQLite, not in FAISS. The gallery is small (tens of objects, tens of views each).

## Scene table

The table `frame_scene` is a cache for the description of a place (see `docs/backend/Voice.md`).
The scene model makes the room scores of a frame when a description needs them. The helpers keep the scores, so a frame is scored one time.
The helpers are in `db/scenes.py`. `db/api.py` gives them as `db.scenes`.

| Column | Meaning |
|--------|---------|
| `image` | The frame. It is the key. The row goes with the frame (cascade), so retention removes it |
| `model` | The model that made the scores. A row of another model is not used |
| `rooms` | JSON: the probability of each room, for example `{"a bedroom": 0.7, "a corridor": 0.2}` |
| `computed_at` | Naive UTC time |

| Helper | Task |
|--------|------|
| `get_room_scores(image_ids, model)` | The saved scores of these frames. A frame without scores, or with scores of another model, is not in the result |
| `set_room_scores(image_id, model, rooms)` | Save the scores of a frame. Scores that exist are replaced. The values must be finite, not negative, and add up to 1 (a difference of 0.01 is allowed). `InvalidRecordError` for a wrong value or an unknown frame |
| `list_neighbour_images(image_id, window_seconds, limit)` | The `done` frames that were captured at most `window_seconds` before or after the frame, the nearest first. The frame itself is not in the list. An unknown frame gives an empty list |

## Memory tables

The tables `sighting` and `last_seen` hold what the system saw of the enrolled objects.
The rule that changes them is in `docs/backend/Memory.md`. The helpers are in `db/sightings.py`. `db/api.py` gives them as `db.sightings`.

| Table | Meaning |
|-------|---------|
| `sighting` | One row for each accepted match. `enrolled_object`, `image`, `region` (all cascade), `seen_at` (the time of the frame), `similarity`, `margin`, `strong` (true for a strong sighting and for a weak sighting that passed the check), `in_hand`. Index on (`enrolled_object`, `seen_at`) |
| `last_seen` | One row for each object (the key is `enrolled_object`, cascade). `image` is `RESTRICT`, so the frame stays. `x0`, `y0`, `x1`, `y1` are a copy of the box. Also `seen_at`, `similarity`, `margin`, `strong`, `in_hand`, `updated_at` |

| Helper | Task |
|--------|------|
| `add_sighting` | Add a row to the log. `InvalidRecordError` if the object, the image or the region does not exist |
| `count_recent(object_id, since, until)` | The number of frames with a sighting in the window (both ends included). A frame counts one time |
| `get_last_seen`, `list_last_seen` | Read the records |
| `set_last_seen` | Make a sighting the record. An older sighting does not replace a newer record. Returns True if the record changed |

`list_images_older_than` does not list an image that `last_seen` uses. `delete_images` refuses such an image with `InvalidStateError`.

### Gallery helpers

`db/api.py` gives them as `db.gallery`. The `backend` module uses them.

| Helper | Task |
|--------|------|
| `create_object`, `get_object`, `list_objects`, `count_objects`, `delete_object` | Objects. `delete_object` returns the file paths for the storage module |
| `add_reference_photo`, `get_photo`, `list_photos`, `count_photos_by_status` | Photos |
| `claim_next_reference_photo` | The worker takes the oldest `pending` photo. It becomes `processing` |
| `store_photo_views`, `mark_photo_failed` | The result of the worker: the box and the views, or a failure |
| `confirm_photo`, `remove_photo`, `retry_photo` | The steps of the carer. A photo in `processing` cannot be removed |
| `load_views`, `load_other_views` | The views of an object, and the views of all other objects (the negatives) |
| `set_calibration`, `clear_calibration` | Save the thresholds, or go back to `draft` with a note |
| `gallery_phrases` | The phrases of all `ready` objects, for the detector |
| `list_photo_paths`, `reset_processing_photos` | For the file checks and for the start-up recovery |

`list_all_file_paths()` in `db/api.py` returns the paths of the image files and of the reference photos.

## Processing flow

The `backend` module controls this flow. The db module gives the helper for each step.

| Step | Action | Helper |
|------|--------|--------|
| 1 | Check for a duplicate upload | `find_image_by_sha256` |
| 2 | Save the file, then insert the image with the status `pending` | `add_image` |
| 3 | A worker takes the oldest `pending` image. The status becomes `processing` | `claim_next_pending` |
| 4 | The worker saves the quality-gate values | `set_image_analysis` |
| 5 | The worker inserts the regions. This gives the region ids | `add_regions` |
| 6 | The worker adds the vectors to the FAISS index, with the region ids as vector ids, and sets `embedding_model` on each region | `store_region_vectors` |
| 7 | The worker sets the final status | `mark_image_done`, `mark_image_rejected` or `mark_image_failed` |

The `ai` module makes the vectors. The db module only stores them.
If the server stops during step 6 or 7, the start-up recovery repairs the data (see below).

## Query helpers

All helpers are in `db/api.py`. They return `Image` and `Region` objects.
Each helper checks its parameters with `assert`.
Each helper uses peewee expressions only. No helper builds SQL from text.

| Helper | Task | Result |
|--------|------|--------|
| `add_image(file_path, sha256, width, height, room=None)` | Insert an image with the status `pending`. `captured_at` is the time of the call. The path must be relative and must not contain `..` | `Image`. Raises `DuplicateImageError` |
| `find_image_by_sha256(sha256)` | Find an image by hash | `Image` or `None` |
| `list_recent_phashes(since=None, limit=200)` | List the perceptual hashes of `done` images, newest first. SQLite cannot compare hashes by Hamming distance, so the caller compares the list (`AiEngine.is_near_duplicate`) | List of text |
| `get_image(image_id)` | Read one image | `Image` or `None` |
| `claim_next_pending()` | In one write transaction, set the oldest `pending` image to `processing`. Two workers cannot get the same image | `Image` or `None` |
| `set_image_analysis(image_id, phash, blur_score)` | Save the quality-gate values | `Image` |
| `add_regions(image_id, regions)` | Insert all regions in one transaction. An empty list is allowed | List of `Region` |
| `get_regions(image_id)` | Read the regions, ordered by id | List of `Region` |
| `mark_regions_embedded(region_ids, embedding_model)` | Record that the vectors are in FAISS. The helper changes all rows or none. The `backend` module uses `store_region_vectors` instead | Number of rows changed |
| `mark_image_done(image_id)` | Set the status `done` | `Image` |
| `mark_image_rejected(image_id, reason)` | Set the status `rejected` | `Image` |
| `mark_image_failed(image_id, error)` | Set the status `failed` | `Image` |
| `list_embedded_region_ids()` | List the ids of the regions that have a vector | List of integers |
| `list_searchable_regions(since, until, embedding_model, limit=5000)` | List the regions that a search can compare, newest frame first. A frame is in the window if `since` < `captured_at` <= `until`. Only regions of `done` frames with `quality_ok` and a vector of `embedding_model` are listed. Each region has its frame in `region.image` | List of `Region` |
| `list_images_older_than(cutoff, limit=100)` | List the oldest `done`, `rejected` and `failed` images that were captured before `cutoff`. An image that waits or is in use is never listed | List of `Image` |
| `delete_images(image_ids)` | Delete the images and their regions (by cascade), then remove their vectors from the index. Unknown ids are ignored. If one image is `processing`, the helper raises `InvalidStateError` and deletes nothing | The `file_path` of each deleted image |
| `list_image_paths()` | List the `file_path` of every image, sorted | List of text |
| `count_images_by_status(status)` | Count the images with one status. The upload step uses it to limit the queue of `pending` images | Integer |
| `close_connection()` | Close the database connection of the current thread. A thread that uses the database for a long time calls it when it ends | `None` |
| `recover_interrupted()` | Reset interrupted images. It changes the tables only. The application uses `startup_recovery()` | `RecoveryReport` |
| `startup_recovery()` | Repair the tables and the index after a stop (see below) | `StartupReport` |

`add_regions` takes `NewRegion` objects. A `NewRegion` checks its own box:
`x0` must be 0 or more and less than `x1`, and `y0` must be 0 or more and less than `y1`.

These helpers work only on an image with the status `processing`:
`set_image_analysis`, `add_regions`, `mark_image_done`, `mark_image_rejected` and `mark_image_failed`.
Each of them changes the row with one `UPDATE` that tests the status.
If the status is not `processing`, the helper raises `InvalidStateError`.

The helpers `mark_image_rejected` and `mark_image_failed` keep the text in `status_detail`.
The helper cuts the text to 500 characters.
Do not put image content or personal data in this text.

### Errors

| Error | Meaning |
|-------|---------|
| `ConfigError` | `config.toml` is missing or not correct |
| `DbError` | Base class of the errors below. The helpers also raise it for other database errors |
| `VectorIndexError` | The vector index refuses the call (for example, an id is already in the index, or the file cannot be read) |
| `DuplicateImageError` | `file_path` or `sha256` is already in the database |
| `InvalidStateError` | The image does not exist or has the wrong status. Also used when a region id does not exist |
| `InvalidRecordError` | The database refuses a value |

### Retention and delete order

`delete_images` deletes the rows first. The vectors are removed after the transaction.
The caller deletes the files last (`storage.delete_images`).
A stop in between leaves orphan files or orphan vectors. The orphan check of the storage module
and `startup_recovery` repair them. The opposite order would leave rows that point to files that do not exist.

## Vector index

The vectors are in a FAISS index. The index file path is `paths.faiss_index` in `db/config.toml`.

- The index is an `IndexIDMap2` over `IndexFlatIP`. The vector id is `Region.id`.
- All vectors have unit length. The inner product of two vectors is their cosine similarity.
- The index search is exact. No training step is needed.
- One lock protects the index, so workers in different threads can use it.
- The index is held in memory. The db module writes it to disk with a temp file and then replaces the old file, so a failed write leaves the old file whole.

### Vector helpers

| Helper | Task | Result |
|--------|------|--------|
| `store_region_vectors(region_ids, vectors, embedding_model)` | Add the vectors to the index and mark the regions as embedded. If the database step fails, the helper removes the vectors again | `None`. Raises `VectorIndexError` or `DbError` |
| `remove_region_vectors(region_ids)` | Remove vectors from the index and clear the marks. Unknown ids are ignored | Number of vectors removed |
| `clear_regions_embedded(region_ids)` | Clear the `embedding_model` of regions. Unknown ids are ignored | Number of rows changed |
| `checkpoint_vectors()` | Write the index to disk now | `None`. Raises `VectorIndexError` |
| `vector_count()` | Count the vectors | Integer |
| `list_vector_ids()` | List the region ids in the index | Sorted list of integers |
| `get_region_vectors(region_ids)` | Read the stored vectors of regions, in the order of the ids. Raises `VectorIndexError` if a region has no vector | `float32` array with the shape (number of ids, dimension) |

`vectors` must be a `float32` array with the shape `(number of ids, dimension)`.
Each row must have a length of 1.0 (a tolerance of 0.001 is allowed).
The helper refuses a value that is not finite, a repeated id, and an id that is already in the index.
When the helper refuses a call, it changes nothing.

### Checkpoints

The index is saved to disk in these cases:
- After `vectors.checkpoint_every` added or removed vectors. This happens inside `store_region_vectors` and `remove_region_vectors`, after the database step is complete.
- When the `backend` module calls `checkpoint_vectors()` at shutdown.

A save error is written to the log and does not stop the ingestion. The next change tries again.
A crash can lose the vectors that the file does not have yet. The start-up recovery repairs this.
In a real test, a `kill -9` lost 10 vectors. The recovery cleared the marks, set the image back to `pending`, and the worker made the vectors again.

## Start-up recovery

The recovery changes data. Only one copy of the application may run on the data.
The application takes an instance lock before the recovery (see `docs/backend/Lifecycle.md`).

The `backend` module calls `startup_recovery()` one time at start.
It must call it after `init_db()` and before it starts the workers.

The helper does these steps:

1. It finds the images with the status `processing`. A stop happened while a worker held them.
   It deletes their regions and vectors, and sets the images to `pending`. The retry makes new regions, so duplicates cannot occur.
2. It removes each vector that has no embedded region in the tables (an orphan vector).
   This happens when the server stops after the index change and before the database change.
3. It clears each mark in the tables that has no vector in the index (a missing vector).
   This happens when the server stops before the index is saved.
4. It finds each `done` image that has a region without a vector (a gap).
   It deletes the regions and vectors of this image and sets it to `pending`. The workers process the image again.

The table changes run in write transactions and come before the index changes.
If the server stops during the recovery, the next recovery finishes the work.
You can run the recovery more than one time. A second run finds nothing to repair.

The helper returns a `StartupReport`:

| Field | Meaning |
|-------|---------|
| `interrupted_image_ids` | Images that were `processing` and are `pending` again |
| `requeued_image_ids` | `done` images with a gap that are `pending` again |
| `interrupted_photo_ids` | Reference photos that were `processing` and are `pending` again |
| `orphan_vectors_removed` | Number of orphan vectors that the helper removed |
| `missing_vectors_cleared` | Number of marks that the helper cleared |

The `backend` module writes the report to the log. It does not need to take other action.

**Rule for the caller:** every region of a `done` image must have a vector.
If the `backend` module inserts a region and does not make a vector for it,
the recovery sets its image to `pending` at every start.

`recover_interrupted()` does step 1 only, and only changes the tables.
Use `startup_recovery()` in the application.

## Configuration

The file `db/config.toml` holds the paths of the SQLite file and the FAISS index.
The paths are relative to the `db/` folder.
The loader refuses absolute paths and paths that leave the `db/` folder.

| Key | Default | Description |
|-----|---------|-------------|
| `paths.sqlite` | `data/saarthi.db` | SQLite file |
| `paths.faiss_index` | `data/regions.faiss` | FAISS index file |
| `vectors.dimension` | `384` | Length of one vector. It must match the embedding model |
| `vectors.checkpoint_every` | `200` | Save the index after this number of added or removed vectors |

Both `vectors` keys must be positive integers.
If the index file has a different dimension than `vectors.dimension`, `init_db()` raises `VectorIndexError`.

`init_db()` reads the file, opens the database, sets these SQLite options, creates the missing tables, and opens the vector index:

| Option | Value | Reason |
|--------|-------|--------|
| `foreign_keys` | on | Enforce the link between region and image |
| `journal_mode` | `wal` | Readers do not block the writer |
| `busy_timeout` | 5000 ms | Wait for a lock before an error |
| `synchronous` | `normal` | Good speed with WAL |

## Logging

The module logs in this format: `{module}-{time}: {message}`.

| Level | Event |
|-------|-------|
| DEBUG | An image or a group of regions is added, vectors are stored |
| INFO | The config is loaded, the database and the index are open, the tables are ready, an image is claimed, an image gets a final status, the index is saved, vectors are removed, the recovery summary |
| WARNING | The start-up recovery repaired the tables or the index, a temp file cannot be removed |
| ERROR | The config is not valid, the database or the index cannot open, the index cannot be saved, or a database call fails |

The default log level is INFO. DEBUG lines are not shown.

The logs do not contain image content or personal data.
