# DB

This is the db module.
It stores the metadata of the captured images in a SQLite database.
It stores the embedding vectors of the detected regions in a FAISS index.
It also stores the gallery: the objects that the owner wants to find, their reference photos, and their vectors.
The module uses the `peewee` ORM.

The `Region.id` value is the vector id in the FAISS index.
The module keeps the tables and the index in step.

For the full design, see `docs/db/Ingestion Database Schema.md`.

## Files

| File | Purpose |
|------|---------|
| `schema.py` | The tables: `Image`, `Region`, `EnrolledObject`, `ReferencePhoto`, `ObjectView` |
| `gallery.py` | The helpers of the gallery. `api.py` gives them as `db.gallery` |
| `support.py` | The error translation that `api.py` and `gallery.py` share |
| `api.py` | The public interface: config loader, `init_db`, the query helpers, the vector helpers and the start-up recovery |
| `vector_index.py` | The FAISS index. It has no database code |
| `errors.py` | The error classes |
| `config.toml` | Paths and vector settings |

## Rules

- This module does not control the flow of the application. The `backend` module does that.
- This module gives small helpers. Each helper does one storage job.
- The `backend` module does not open the SQLite file or the FAISS file. It uses `db/api.py` only.
- This module does not read or write image files. The `storage` module does that.
- This module does not make embeddings. The `ai` module does that.

## Tables

- `image`: one row for each uploaded image.
- `region`: one row for each detected box in an image.
- `enrolled_object`: one row for each object of the gallery, with its calibration.
- `reference_photo`: one row for each photo that a carer gave for an object.
- `object_view`: one vector for each view of a reference photo. These vectors are in SQLite, not in FAISS, because the gallery is small.

## Configuration

Edit `config.toml` to change where the data is stored and how the index is saved.
All paths are relative to the `db/` folder.
Absolute paths and paths that leave the `db/` folder are not allowed.

```toml
[paths]
sqlite = "data/saarthi.db"
faiss_index = "data/regions.faiss"

[vectors]
dimension = 384
checkpoint_every = 200
```

| Key | Meaning |
|-----|---------|
| `vectors.dimension` | Length of one vector. It must match the embedding model |
| `vectors.checkpoint_every` | The index is saved after this number of added or removed vectors |

The index is also saved when the application stops (`checkpoint_vectors()`).
A crash can lose at most `checkpoint_every` vectors. The start-up recovery repairs the loss.

## Usage

```python
from db.api import init_db, startup_recovery, checkpoint_vectors

database, config = init_db()
report = startup_recovery()
# ... run the application ...
checkpoint_vectors()  # When the application stops
```

`init_db` opens the database and the vector index, and creates the missing tables.
You can call it more than one time.

## Query helpers

All helpers are in `api.py`. The helpers return `Image` and `Region` objects.

| Helper | Task |
|--------|------|
| `add_image` | Insert an image with the status `pending` |
| `find_image_by_sha256` | Find an image with the same hash (duplicate upload) |
| `list_recent_phashes` | List the perceptual hashes of `done` images, newest first (near-duplicate check) |
| `get_image` | Read one image |
| `claim_next_pending` | Set the oldest `pending` image to `processing` |
| `set_image_analysis` | Save `phash` and `blur_score` of a `processing` image |
| `add_regions` | Insert the regions of a `processing` image |
| `get_regions` | Read the regions of an image |
| `mark_image_done` | Set a `processing` image to `done` |
| `mark_image_rejected` | Set a `processing` image to `rejected`, with a reason |
| `mark_image_failed` | Set a `processing` image to `failed`, with an error text |
| `list_images_older_than` | List `done`, `rejected` and `failed` images captured before a time (retention) |
| `delete_images` | Delete images, their regions and their vectors. Returns the file paths for the storage module |
| `list_image_paths` | List the file path of every image (orphan and missing file checks) |
| `count_images_by_status` | Count the images with one status (the upload step limits the queue with it) |
| `close_connection` | Close the database connection of the current thread. A long-running thread calls it when it ends |

A helper that changes the status of an image works only from the correct status.
If the status is not correct, the helper raises `InvalidStateError`.

## Vector helpers

| Helper | Task |
|--------|------|
| `store_region_vectors` | Add the vectors of regions to the index and mark the regions as embedded. It does both or none |
| `remove_region_vectors` | Remove vectors from the index and clear the marks |
| `checkpoint_vectors` | Write the index to disk now. Call it when the application stops |
| `vector_count` | The number of vectors in the index |
| `list_vector_ids` | The region ids that have a vector in the index |

`store_region_vectors` needs a `float32` array with one row of unit length for each region id.

Low-level helpers that you normally do not need:
`mark_regions_embedded`, `clear_regions_embedded`, `list_embedded_region_ids`, `recover_interrupted`.

## Start-up recovery

Call `startup_recovery()` one time after `init_db()` and before the workers start.
It repairs the tables and the index after a stop, and returns a `StartupReport`.
The backend writes the report to the log. No other action is needed.

The application makes sure that only one copy runs on the data (`backend/instance_lock.py`).
The recovery changes data, so it must never run beside another copy.

**Rule:** every region of a `done` image must have a vector.
If a region has no vector, the recovery sets its image back to `pending`.

## Errors

| Error | Meaning |
|-------|---------|
| `ConfigError` | `config.toml` is missing or not correct |
| `DbError` | Base class of the errors below. Also used for other database errors |
| `DuplicateImageError` | `file_path` or `sha256` is already in the database |
| `InvalidStateError` | The image does not exist or has the wrong status |
| `InvalidRecordError` | The database refuses a value |
| `VectorIndexError` | The vector index refuses the call (for example, an id is already in the index) |
