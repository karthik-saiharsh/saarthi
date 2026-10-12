# Enrollment

## Purpose

Enrollment builds the gallery. The gallery holds the objects that the owner wants to find.
A carer gives each object a name and a category phrase, for example "a bottle".
The carer adds photos of the object. The system finds the object in each photo and makes vectors.
It then calculates a threshold for the object.
These are the stages E1 to E4 of `docs/guide/DINO-Based Object Recall Pipeline.md`.

The gallery is not the archive. The archive holds the regions of the frames from the wearable.
The gallery holds the objects that the system looks for.
The ingestion uses the gallery only for the detector phrases (see "Link to ingestion").
Matching and the last-seen record are in `docs/backend/Memory.md`. The search is in `docs/backend/Search.md`.
The search uses the names and the phrases of the objects to find the object that a user means.
The reference photos are also used by the verification (I5), which checks a weak sighting (see `docs/backend/Verification.md`).
The verification needs the confirmed photos and their files.

## Where it fits

```
carer ──▶ UI /enroll or HTTP /api/objects ──▶ EnrollmentService ──▶ db: object, photo `pending`
                                                      │                        │
                                                      └─── wake() ──▶ Worker ──┘
                                                                         │
                         db: photo `located` (box and views) or `failed` ◀┘
carer: Accept ──▶ photo `confirmed` ──▶ calibration ──▶ object `ready`
```

- `backend/enrollment.py` is the `EnrollmentService`. It holds the steps that need more than one module.
- `backend/worker.py` does the model work for a photo.
- `backend/http_objects.py` has the HTTP routes. `backend/ui_adapter.py` has the adapter for the UI.
- All methods of the service block. Call them from a thread.

## The steps

| Stage | Who | What happens |
|-------|-----|--------------|
| E1 Collect | The carer | Creates the object and adds photos. The service checks the photo, removes the metadata, and stores it |
| E2 Locate | The worker | Finds the object in the photo with the detector. It uses the category phrase and the synonyms |
| E3 Views | The worker | Makes 4 vectors for the box (see "Views") |
| Confirm | The carer | Looks at the crop of each photo. Accepts it, or removes the photo |
| E4 Calibrate | The service | Calculates the thresholds from the accepted photos (see "Calibration") |

A photo has one of these states:

| State | Meaning |
|-------|---------|
| `pending` | Stored. The worker did not look at it yet |
| `processing` | The worker looks for the object |
| `located` | The worker found the object. The carer must check the crop |
| `confirmed` | The carer accepted the crop. The views count for the object |
| `failed` | The object was not found, or an error occurred |

An object is `draft` or `ready`. An object is `ready` when it has at least 3 confirmed photos.
Then it has `tau` and `tau_strong`.

## Why the carer confirms the crop

An open-vocabulary detector returns a box also for an object that is not in the photo.
In a test, the phrase "a violin" gave a box with the score 0.44. The phrase "a phone" gave 0.69 for a real phone.
The score alone cannot tell the two cases apart. So the carer looks at each crop.
A photo where the detector finds nothing is `failed`. Its reason is the fixed text `object not found`.

## Photo steps in the worker

`Worker.process_next()` takes a reference photo before it takes an image. So the carer does not wait behind the frames.
For one photo the worker does these steps:

1. Claim the oldest `pending` photo. It becomes `processing`.
2. Read and decode the file.
3. `engine.locate_object(image, category_phrase, synonyms)`. This is one detector pass with the phrases of the object.
   It does not add the hand and depiction phrases (see "Known limits").
   The best box is the result.
4. If there is no box: the photo is `failed` with the reason `object not found`.
5. `engine.embed_views(image, box)` makes the views.
6. Store the box and the views. The photo is `located`.

An error marks only this photo `failed`. The reason is the name of the error class.
At the start, a photo that was `processing` goes back to `pending` (the start-up recovery).

## Views

Each photo gives 4 vectors. The list is `[enrollment].augmentations` in `ai/config.toml`.

| View | How it is made |
|------|----------------|
| `original` | The crop of the box |
| `box_smaller` | The box 10 % smaller |
| `box_larger` | The box 10 % larger |
| `brighter` | The crop with 15 % more brightness |

The three box views use one model run. The vectors are in the table `object_view`, as `float32` bytes.
They are not in the FAISS index, because the gallery is small.

## Calibration

The functions are in `ai/calibration.py`. They use numpy only. The service calls them for each object.

1. `own` is the list of the cosine similarities between the views of different photos of the object.
   Views of the same photo are too alike, so they are not used.
2. `tau = mean(own) - 0.5 * std(own)`, but at least `tau_min` (0.45). With 3 or 4 photos, the formula alone is not stable.
   In a test with 14 photos, it gave values from 0.18 to 0.54, and a low value let many wrong boxes pass.
3. The negatives are the views of all other objects. Let `neg_max` be the highest similarity between an own view and a negative.
   Then `tau = max(tau, neg_max + margin_min)`.
4. `tau_strong = tau + 0.10`. The values are not larger than 1.0.
5. If `neg_max + margin_min` is above the mean of `own`, the object is still `ready`. It has a note: it is too similar to another object.
6. If the mean of `own` is below `tau_min`, the object is still `ready`. It has a note: the photos look very different from each other.

The values are in `[enrollment]` of `ai/config.toml`: `tau_std_factor`, `tau_min`, `tau_strong_offset`, `margin_min` (0.03),
`margin_strong` (0.08) and `min_photos` (3). The reasons for `tau_min` and `tau_strong_offset` are in `benchmarks/calibration/README.md`.

Every change of a photo or an object calibrates all objects again. The negatives of an object change when another object changes.

## Look-alikes

A look-alike is an object that looks like one of the owner's objects, for example the bottle of a family member.
The carer adds it as a normal object and sets `tracked = false`.
It is a negative for the other objects, and the system never reports it.
The detector still looks for it (see "Link to ingestion").

## Words from the carer

The name, the category phrase and the synonyms go into the prompt of the detector. The service checks them.

- A name has 1 to 80 printable characters.
- A phrase has lowercase letters, digits, spaces, apostrophes and hyphens. It has at most 6 words and 40 characters.
  A period is not allowed, because it ends a phrase in the prompt. Capital letters become lowercase.
- An object has at most 5 synonyms. A synonym equal to the category phrase or to another synonym is dropped.
- The service gives the error `InvalidInputError` for a wrong value.

The log shows ids and counts. It never shows a name or a phrase.

## Limits

These settings are in the table `[enrollment]` of `backend/config.toml`.

| Key | Meaning |
|-----|---------|
| `max_photos_per_object` | The most photos for one object (20). The error is `LimitReachedError` |
| `max_objects` | The most objects in the gallery (50) |

The upload limits of `storage/config.toml` apply to the photos too.

## Link to ingestion

For each frame, the worker calls `db.gallery.gallery_phrases()`.
The result is the category phrase and the synonyms of all `ready` objects, also the look-alikes. Each phrase is in the list one time.
The worker gives the list to the detector.
If no object is `ready`, the worker uses `detection.default_phrases` of `ai/config.toml`.
`detection.small_phrases` is still a list in the config. A gallery phrase gets the tiled pass only if it is in this list.

## HTTP routes

The routes use the same optional token, the same body limit and the same fixed messages as the upload route.
The token is checked before the body is read. An id in a path must be a positive number. Another value gives 404.

| Route | Action |
|-------|--------|
| `POST /api/objects` | Create an object. The body is JSON: `name`, `category`, `synonyms` (optional), `tracked` (optional). The body has at most 16 KiB. The answer is 201 |
| `GET /api/objects` | List the objects |
| `GET /api/objects/{id}` | One object with its photos |
| `DELETE /api/objects/{id}` | Delete the object, its photos and its files (204) |
| `POST /api/objects/{id}/photos` | Add a photo. The body is the image. The answer is 201 |
| `GET /api/objects/{id}/photos/{photo}/crop?max_side=512` | A JPEG of the crop. `max_side` is from 32 to 1024 |
| `POST /api/objects/{id}/photos/{photo}/confirm` | The carer accepts the crop |
| `POST /api/objects/{id}/photos/{photo}/retry` | Try a `failed` photo again |
| `DELETE /api/objects/{id}/photos/{photo}` | Remove a photo (204) |

| Code | Meaning |
|------|---------|
| 400 | The body or a value is not valid |
| 401 | The token is missing or wrong |
| 404 | The object, the photo or its file does not exist, or the crop is not ready |
| 409 | The name or the photo exists, the state is wrong, or a limit is reached |
| 413, 415, 507 | The photo is too large, is not an image, or the disk is full |
| 500 | Another error |

The crop answer has `Cache-Control: no-store`, because the crop is a private image.

```
curl -X POST -H "Content-Type: application/json" \
  -d '{"name": "Blue bottle", "category": "a bottle"}' http://<server>:<port>/api/objects
curl -X POST --data-binary @photo.jpg http://<server>:<port>/api/objects/1/photos
curl -X POST http://<server>:<port>/api/objects/1/photos/1/confirm
```

## Errors

| Error | Meaning |
|-------|---------|
| `InvalidInputError` | A name or a phrase is not valid |
| `LimitReachedError` | The gallery or the object has the largest number of objects or photos |
| `NotFoundError` | The object or the photo does not exist |
| `db.DuplicateObjectError`, `db.DuplicateImageError` | The name exists, or the object has this picture already |
| `db.InvalidStateError` | The action does not fit the state, for example to remove a photo that the worker holds |

The route and the UI turn each error into a fixed message without internal details.

## Security and privacy

- A reference photo is a private image. It has no metadata, a random name and a safe path, like a frame.
- A delete removes the rows first and the files last. If the application stops in between, only orphan files remain.
- The orphan job knows the reference photos (`db.list_all_file_paths()`). It does not delete them. A test checks this.
- Retention deletes frames only. It never deletes a reference photo.
- The names and the phrases are not in the log.

## First measurements

The test used the real application and the real models on the development machine, through the HTTP routes.

- Four variants of the test photo (same size, different brightness, one cropped) were added to one object.
  The worker found the phone in all four photos in 33 seconds (about 8 seconds for each photo).
  The crop of the first photo showed the phone.
- The same photo sent again gave 409.
- After the carer accepted the four photos, the object was `ready`. `tau` was 0.98 and `tau_strong` was 1.0 (the values of that time, before `tau_min`).
  The value is high because the four variants are almost the same picture. Real photos from different places give lower values.
- A frame (the same test photo) used the gallery phrases: the detector made one pass. The default list needs more passes.
  But the frame had 0 candidates (see "Known limits").

## Known limits

- **The depiction rule marks the object in the frame.** In the test, the phone lay on a book cover.
  The detector found the phone with the score 0.57 when the prompt had no extra phrases.
  With the screen, frame and mirror phrases in the prompt, the phone box got the label "a phone a smartphone a mirror" (0.39),
  and a "television screen" box covered it. The old rule dropped such a box, so the frame had no regions.
  The owner decided to keep the box and set `quality_ok = False`. The region stays in the archive with its vector.
  The matching step must not use it for the last-seen record. Enrollment uses `context_phrases=False` and is not affected.
- The detector is not perfect. The carer must check each crop. The system has no automatic check.
- Boxes only. There are no masks, so the views contain some background. Mask pooling and context negatives are not built.
- No re-embedding. If the embedding model changes, the carer must add the photos again. The photos are kept for this.
- An object cannot be changed after it is created. The carer deletes it and adds it again.
- The carer cannot draw the box by hand.
- `small_phrases` is a list in the config. There is no flag for a small object in the gallery.
- A photo of the same picture can belong to two objects, but only one time to the same object.

## Tests

| File | What it checks |
|------|----------------|
| `tests/test_db_gallery.py` | The tables and the helpers of the gallery |
| `tests/test_ai_calibration.py`, `tests/test_ai_enrollment.py` | The calibration, the views and the box scaling |
| `tests/test_ai_models.py` (slow) | `locate_object` and `embed_views` with the real models |
| `tests/test_backend_enrollment.py` | The service: words, limits, photos, delete, calibration, rollback |
| `tests/test_backend_worker.py` | The photo step, the order, the gallery phrases for a frame |
| `tests/test_backend_http_objects.py` | Every route, the token, bad ids and bodies, the fixed messages |
| `tests/test_backend_ui_enrollment.py` | The adapter for the UI |

The tests use the actual database and storage. A fake engine replaces the models. Each test removes its own data.
