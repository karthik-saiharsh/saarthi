# Storage

This is the storage module.
It keeps the images from the user's wearable as files on disk.
It removes all metadata from an image before it saves the image.

For the full design, see `docs/storage/Storage Module.md`.

## Files

| File | Purpose |
|------|---------|
| `api.py` | The public interface: `init_storage` and the helpers |
| `imaging.py` | `prepare_image`: check, orient, remove metadata, encode, hash. No disk access |
| `files.py` | Safe paths, atomic write, read, delete, list, orphan and missing checks |
| `settings.py` | Reads `config.toml` |
| `errors.py` | The error classes |
| `config.toml` | Root folder, limits, JPEG quality, retention, cleanup time |

## Rules

- This module does not control the flow of the application. The `backend` module does that.
- This module gives small helpers. Each helper does one storage job.
- The `backend` module uses `storage/api.py` only.
- This module does not know the database. The `backend` module gives it the paths that the database holds.
- This module does not look at what is in an image. The `ai` module does that.

## What a stored image is

- Every image is a JPEG (quality 95) in RGB.
- The orientation of the camera is applied to the pixels.
- The file has no metadata: no EXIF, no GPS, no XMP, no ICC profile, no comment.
- The file name is random. The path has the form `YYYY/MM/DD/<32 hex characters>.jpg`.
- The path that the module gives out is relative to the storage root.

## Configuration

Edit `config.toml`.

| Key | Meaning |
|-----|---------|
| `location.root` | Folder for the files. A relative path starts in `storage/` and must stay inside it. An absolute path is also allowed, for example a mounted SSD |
| `limits.max_bytes` | Largest upload, in bytes |
| `limits.max_pixels` | Largest image, in pixels. Keep it equal to `[image] max_pixels` in `ai/config.toml` |
| `limits.min_free_bytes` | New images are refused if the disk would have less free space |
| `image.jpeg_quality` | Quality of the saved JPEG, from 1 to 95 |
| `retention.days` | Images older than this can be deleted. 0 keeps all images. Today 1 (a rolling buffer of 24 hours) |
| `cleanup.orphan_grace_seconds` | A file that the database does not know is an orphan only after this time |

## Helpers

| Helper | Task |
|--------|------|
| `init_storage` | Read the config, prepare the root folder, remove stale temporary files |
| `current_config` | Return the config that `init_storage` loaded (limits, retention days) |
| `prepare_image` | Check an upload and make the JPEG without metadata. Returns the bytes, the SHA-256 hash, the width and the height |
| `save_image` | Write a prepared image to disk. Returns the relative path |
| `read_image` | Return the bytes of a stored image |
| `read_crop` | Return a JPEG of a box of a stored image, shrunk to `max_side`. The carer sees it to confirm a reference photo |
| `delete_images` | Delete stored images. Returns a report of deleted, missing and failed paths |
| `list_stored_paths` | List all stored paths |
| `find_orphan_files` | Find files that the database does not know |
| `find_missing_files` | Find database paths that have no file |
| `free_space_bytes` | Free space on the disk of the root |

## Usage

```python
import storage.api as storage

storage.init_storage()
prepared = storage.prepare_image(upload_bytes)
# Ask the database for prepared.sha256 here. Stop if it is a duplicate.
path = storage.save_image(prepared)
# Add path, prepared.sha256, prepared.width and prepared.height to the database.
```

## Errors

| Error | Meaning |
|-------|---------|
| `StorageError` | Base class of all errors below |
| `StorageConfigError` | `config.toml` is not correct, or `init_storage` was not called |
| `UnsupportedImageError` | The data is not a JPEG, PNG or WebP image, or it is animated, 16-bit or damaged |
| `ImageTooLargeError` | The image has too many bytes or too many pixels |
| `StorageFullError` | The disk has too little free space |
| `StoragePathError` | A path is not a safe storage path |
| `StorageNotFoundError` | The file does not exist |
| `StorageIOError` | The disk refuses a read, a write or a delete |
| `InvalidCropError` | The box of `read_crop` is not inside the image |
