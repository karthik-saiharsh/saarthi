# Maintenance Jobs

## Purpose

The maintenance jobs keep the data healthy while the application runs.
They run in one background thread. The first run starts when the application starts.
After this, they run every `maintenance.interval_minutes` (60 by default).

`Maintenance.run_once()` does all jobs one time. It returns a `MaintenanceReport`.
An error in one job is written to the log. It does not stop the other jobs.

## The jobs

They run in this order.

### 1. Retention

Delete images that are too old.

- The age limit is `retention.days` in `storage/config.toml`. The value 0 (the default) keeps all images, and the job does nothing.
- The job lists the images that were captured before the limit. It lists only `done`, `rejected` and `failed` images.
  It skips an image that a last-seen record uses.
  An image that waits for the worker or is in use is never deleted.
- It deletes the images in groups of `maintenance.retention_batch` (100 by default).
- The order of the deletes is on purpose:
  1. `db.delete_images` deletes the rows, the regions and the vectors.
  2. `storage.delete_images` deletes the files.

  If the application stops in between, only orphan files remain, and the next orphan job removes them.
  The opposite order could leave rows that point to files that do not exist.

### 2. Orphan files

Delete files that no database row owns.

- A file is an orphan if its path is not in `db.list_all_file_paths()`. This list holds the files of the images and of the reference photos.
  The reference photos must be in the list, or the job would delete them.
- A file that is younger than `cleanup.orphan_grace_seconds` (3600 by default, in `storage/config.toml`) is not an orphan yet.
  It can be an upload that is still being written.
- The job deletes at most `maintenance.orphan_delete_limit` files in one run (100 by default). A larger number is only reported,
  with a warning to check that the database is the right one.
  This protects the images. If the database is empty or wrong, every image file would look like an orphan.
- The value 0 never deletes an orphan file. The job only reports it.

### 3. Missing files

Report the rows whose file does not exist.

- The job writes a warning with the number and the first five paths.
- It changes nothing. A person must decide what to do with these images.

### 4. Vector checkpoint

Save the vector index to disk (`db.checkpoint_vectors()`).

- This is a save by time. The index is also saved after every `vectors.checkpoint_every` new vectors and at shutdown.
- A save error is written to the log. The report shows `checkpoint_saved = False`.

## The report

| Field | Meaning |
|-------|---------|
| `expired_images_deleted` | Images that the retention job deleted |
| `orphan_files_found` | Orphan files that were found |
| `orphan_files_deleted` | Orphan files that were deleted (up to the limit) |
| `missing_files` | Rows without a file |
| `checkpoint_saved` | True if the vector index was saved |

A run writes one line to the log:

```
backend.maintenance-...: Maintenance: 0 expired image(s) deleted, 0 orphan file(s) found and 0 deleted, 0 missing file(s), vectors saved: True
```

## Configuration

`backend/config.toml`, table `[maintenance]`:

| Key | Default | Description |
|-----|---------|-------------|
| `interval_minutes` | 60 | The time between two runs |
| `orphan_delete_limit` | 100 | The most orphan files that one run deletes. 0 never deletes one |
| `retention_batch` | 100 | The number of expired images that are deleted in one group |

Retention itself is set in `storage/config.toml` (`retention.days`). The orphan grace time is `cleanup.orphan_grace_seconds` in the same file.

## Start and stop

`Backend.start()` starts the maintenance thread together with the worker.
`Backend.stop()` stops it first, then the worker. A run that is in progress finishes before the thread stops.
The thread closes its database connection when it ends.

## Known limits

- The job runs in the same process as the worker. A run of retention for many images takes time.
  The worker and the upload step keep working during the run.
- Retention never deletes an image that a last-seen record uses (see `docs/backend/Memory.md`). The answer shows a crop of that frame.
  So one full frame stays on disk for each object that the system has seen. The frame can expire when a newer sighting replaces the record.
- A missing file is only reported. There is no helper that repairs the row.

## Tests

`tests/test_backend_maintenance.py` uses the real storage and database.
The retention tests use a retention time of 9000 days and images from the year 2000, so only the test images are old enough to be deleted.

| Test | What it checks |
|------|----------------|
| Retention off | Nothing is deleted with `retention.days = 0` |
| Retention on | Old finished images go with their files and vectors. A recent image and an old `pending` image stay |
| Groups | Five images are deleted with a group size of 2 |
| A failure | An error in retention does not stop the orphan job and the checkpoint |
| Orphan files | Old orphans are deleted. New orphans and known files stay |
| Orphan limit | The limit stops a mass delete. The value 0 deletes nothing |
| Missing files | A missing file is reported and nothing changes |
| Checkpoint | The index is saved. A save error does not stop the run |
| The thread | It runs at once and stops fast |
