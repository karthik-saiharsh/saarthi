# Storage Module

Oct 25, 2026 · @KS

## Purpose

The storage module keeps the images from the user's wearable as files on the disk of the server.
The images show the home of a person who may have dementia. They are private.
The module therefore removes all metadata from each image before it saves the image.
The module also makes sure that no path can lead out of the storage folder.

The module does not control the flow of the application. The `backend` module does that.
The module does not know the database.

## Where it fits

| Step | Module | Helper |
|------|--------|--------|
| Upload: check the file and remove the metadata | storage | `prepare_image` |
| Upload: look for a duplicate | db | `find_image_by_sha256` |
| Upload: write the file | storage | `save_image` |
| Upload: add the row | db | `add_image` |
| Worker: read the file | storage | `read_image` |
| Worker: analyse the image | ai | `open_image`, `AiEngine` |
| Retention: list old images, delete rows and vectors | db | `list_images_older_than`, `delete_images` |
| Retention: delete the files | storage | `delete_images` |
| Start: find orphan and missing files | storage and db | `find_orphan_files`, `find_missing_files`, `list_image_paths` |

## What a stored image is

- The format is JPEG, quality 95, RGB.
- The orientation of the camera is applied to the pixels. The file has no orientation tag.
  The `ai` module reads the pixel grid of the file. The boxes in the database use the same grid.
- The file has no metadata. The module builds a new image from the raw pixels, so nothing can be copied from the source.
  A test reads the segments of the saved file. It finds no EXIF, GPS, XMP, ICC profile or comment.
- The file name is a random identifier. It never contains a name from the user.
- The path has the form `YYYY/MM/DD/<32 hex characters>.jpg`. The date is the UTC date of the server clock.
- The same input always gives the same bytes. An exact duplicate upload has the same SHA-256 hash.
  Pictures that differ only in metadata also have the same hash.

## prepare_image

`prepare_image(data)` does not use the disk. It does these steps:

1. It refuses empty data and data that is larger than `limits.max_bytes`.
2. It opens the image and finds the format from the content of the data.
   It does not use the file name or the content type of the browser.
   Only JPEG, PNG and WebP are accepted.
3. It refuses an animated image, a 16-bit image, and an image with more pixels than `limits.max_pixels`.
   The pixel check happens before the full decode. This stops a decompression bomb.
4. It decodes the whole image. A truncated or damaged file is refused.
5. It applies the EXIF orientation. It converts the image to RGB. A transparent area becomes white.
6. It builds a new image from the raw pixels and saves it as a JPEG.
7. It returns a `PreparedImage` with the bytes, the SHA-256 hash, the width and the height.
   The width and the height are the values of the saved image.

## Writing and reading

- Before it writes, the module checks the free space. It refuses the write if the disk would have less than `limits.min_free_bytes` free.
- The module writes to a temporary file with the mode 0600. It calls `fsync`. Then it renames the file to its final name in one step.
  A reader never sees a half-written file. A stop of the server leaves at most a `.tmp` file.
- The folders for the year, the month and the day have the mode 0700.
- Every helper that takes a path accepts only the exact form `YYYY/MM/DD/<32 hex characters>.jpg`.
  This rule refuses `..`, absolute paths, backslashes, other suffixes and extra characters.
  After this, the module resolves the real path. If it is not inside the storage root, the module refuses it.
  This stops a symbolic link that leads out of the root. The error is `StoragePathError`.
- `delete_images` checks all paths first. One bad path stops the whole call before any file is deleted.
  After a delete, the module removes the empty day, month and year folders.

## Orphan and missing files

A stop at the wrong time can leave a file without a row, or a row without a file.

- An **orphan** is a file that the database does not know.
  `find_orphan_files(known_paths)` lists the orphans. The backend gives it the paths from `db.list_image_paths()`.
  A file that is younger than `cleanup.orphan_grace_seconds` is not an orphan yet.
  It can be an upload that has no row yet.
- A **missing file** is a path in the database that has no file.
  `find_missing_files(known_paths)` lists them. The backend decides what to do.
- `init_storage` removes temporary files that are older than the grace time.

## Retention

The database knows the age of an image. The database, the index and the storage each delete their own part.
The `backend` module runs these steps:

1. `cutoff = now - retention_days`. A value of 0 for `retention.days` means that nothing is deleted.
2. `images = db.list_images_older_than(cutoff)`. Only `done`, `rejected` and `failed` images are listed.
3. `paths = db.delete_images(ids)`. This deletes the rows, the regions and the vectors.
4. `storage.delete_images(paths)`. This deletes the files.

The order is on purpose. The rows go first and the files go last.
If the server stops in between, only orphan files remain, and the orphan check removes them.
The other order could leave rows that point to files that do not exist.

## Configuration

The file `storage/config.toml` holds the settings.

| Key | Default | Description |
|-----|---------|-------------|
| `location.root` | `uploads` | Folder for the files. A relative path starts in `storage/` and must stay inside it. An absolute path is also allowed |
| `limits.max_bytes` | 15728640 | Largest upload, in bytes. The same as the limit of the UI |
| `limits.max_pixels` | 40000000 | Largest image, in pixels. Keep it equal to the value in `ai/config.toml` |
| `limits.min_free_bytes` | 1073741824 | New images are refused if less than this is free |
| `image.jpeg_quality` | 95 | Quality of the saved JPEG. The value must be from 1 to 95 |
| `retention.days` | 0 | Delete images older than this number of days. 0 keeps all images |
| `cleanup.orphan_grace_seconds` | 3600 | A file is an orphan only after this time |

`load_config` checks every value and names the key of a wrong value. It raises `StorageConfigError`.
The default root `storage/uploads/` is in `.gitignore`.

## Errors

All errors derive from `StorageError`.

| Error | Meaning |
|-------|---------|
| `StorageConfigError` | `config.toml` is not correct, or `init_storage` was not called |
| `UnsupportedImageError` | The data is not a usable JPEG, PNG or WebP image |
| `ImageTooLargeError` | The image has too many bytes or too many pixels |
| `StorageFullError` | The disk is full, or it has less free space than `limits.min_free_bytes` |
| `StoragePathError` | A path is not a safe storage path |
| `StorageNotFoundError` | The file does not exist |
| `StorageIOError` | The disk refuses a read, a write or a delete |

## Security and privacy

- All metadata is removed. This includes the GPS position and the camera maker.
- The file format comes from the content. Animated and 16-bit images are refused.
- The size and the pixel count are checked before the image is decoded.
- File names are random. Paths are checked twice. A link cannot lead out of the root.
- Files have the mode 0600. The folders that the module makes have the mode 0700.
- The log shows relative paths, sizes and counts. It does not show image content or original file names.
- The module does **not** encrypt the files. An image shows the inside of a home.
  The owner decided to delay disk encryption. The reasons are:
  the server is used on a local network (a small office or home network), it is never connected directly to the internet,
  it stays on the premises, and it has one user.
  Look at this decision again if one of these facts changes. Disk encryption of the drive of the storage root,
  for example LUKS on the Raspberry Pi, is the planned way to add it.

## Known limits

- The JPEG is encoded again. The pixels change a little. On the test photo, the number of detected candidates changed from 13 to 14,
  and 12 of the 13 boxes were still the same (IoU of 0.8 or more). The blur score did not change.
- A colour profile is removed. A wide-gamut image can look different after the conversion to RGB. This does not matter for detection.
- A 16-bit image is refused.
- The date folder uses the clock of the server. A wrong clock gives wrong folders, but no wrong data.
- The module does not delete anything by itself. The `backend` module must run the retention steps.

## First measurements

On a Core Ultra 9 CPU (no GPU), the preparation of a 968 x 728 photo takes about 33 ms.
A 12 megapixel photo of noise (the worst case for JPEG, 10.7 MB) takes about 0.9 s.
The Raspberry Pi will be slower. These times are small compared with the 10 s of the detector.

## Tests

| File | What it tests |
|------|---------------|
| `tests/test_storage_imaging.py` | The removal of all metadata, the orientation, the formats, the limits, the hash |
| `tests/test_storage_files.py` | Paths, links, atomic write, full disk, delete, orphan and missing files, temporary files |
| `tests/test_storage_api.py` | The public helpers on the real root folder |
| `tests/test_storage_settings.py` | The config file |
| `tests/test_ingestion_chain.py` (slow) | One photo from the upload to the retention, with storage, db, index and models |
