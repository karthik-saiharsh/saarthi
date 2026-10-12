# Memory (Matching and Last Seen)

## Purpose

The memory answers one question: where did the system last see each object of the owner?
It does the stages I4 (match) and I6 (update memory) of `docs/guide/DINO-Based Object Recall Pipeline.md`.
The search (stage Q1) reads the memory. See `docs/backend/Search.md`.

## Where it fits

```
Worker: frame ──▶ regions and vectors stored ──▶ MemoryService.remember ──▶ db: sighting log, last_seen
                                                     │
                                                     └─▶ AiEngine.match_regions (see docs/ai/AI Ingestion Module.md)
```

- `backend/memory.py` has `MemoryService`. The worker owns one.
- The worker calls it after it stores the regions and the vectors, and before it marks the image `done`.
- A frame with no region, or a gallery with no `ready` object, does nothing.

## The steps for one frame

1. Load the `ready` objects of the gallery with their views. An object with no view of the current model is left out.
2. Match the regions with the objects (`AiEngine.match_regions`). A region with `quality_ok = False` is never matched.
3. Check each weak match against the reference photos of the object (stage I5, see `docs/backend/Verification.md`).
   A match that passes the check counts as strong.
4. Write each accepted match to the table `sighting`. The time is the time at which the server received the frame (`Image.captured_at`).
5. Apply the rule below. It decides if the match changes the record `last_seen` of the object.

## The rule (hysteresis)

| Sighting | What happens |
|----------|--------------|
| Strong (`similarity >= tau_strong` and `margin >= margin_strong`), or weak and verified | The record changes at once |
| Weak (accepted, not strong, and not verified) | The record changes only if the object has enough **weak** sightings in different frames inside the time window. The new sighting counts. A strong sighting does not count, so a weak false match after a true strong sighting cannot replace the record alone |

The settings are in `[memory]` of `backend/config.toml`:

| Key | Default | Meaning |
|-----|---------|---------|
| `hysteresis_window_seconds` | 60 | The window ends at the time of the new sighting |
| `hysteresis_hits` | 2 | The number of frames with a weak sighting that a weak sighting needs. 1 means that every accepted sighting changes the record |
| `verify_weak` | true | Check a weak sighting against the reference photos (stage I5) |
| `verify_max_references` | 3 | The check compares with at most this number of reference photos |

- The count reads the table `sighting`. A restart loses nothing.
- A frame counts one time, also after a repair that processes it again.
- An older sighting never replaces a newer record. This protects the record if the frames are not in order.
- The pipeline document uses "verified" for a sighting that needs no second hit. Here, a strong sighting and a verified sighting
  are the same in the data: `strong = true`. The check can only promote a weak sighting. It never rejects one.
  A sighting that fails the check stays weak and follows the rule of the hysteresis. An error of the check changes nothing.
- A sighting with `in_hand` is stored with the flag. It changes the record by the same rule.

## The tables

Both tables are in `db/schema.py`. The helpers are in `db/sightings.py` (`db.sightings`).

| Table | One row for | Content |
|-------|-------------|---------|
| `sighting` | Each accepted match | object, image, region, `seen_at`, `similarity`, `margin`, `strong`, `in_hand` |
| `last_seen` | Each object | object (the key), image, the box (a copy), `seen_at`, `similarity`, `margin`, `strong`, `in_hand`, `updated_at` |

- The rows of `sighting` go with the object, the image and the region (cascade).
- The row of `last_seen` goes with the object.
- `last_seen.image` is `RESTRICT`. The answer shows a crop of this frame.
  `db.list_images_older_than` does not list such an image, so retention keeps it.
  `db.delete_images` refuses it with `InvalidStateError`.
  When a newer sighting replaces the record, the old frame can expire again.
- The box is a copy. The region row can go away (for example in a repair) and the record stays correct.

## Failures

- An error in the matching or in the tables marks the image `failed`, like any other error of the worker.
  The regions and the vectors of the image stay. Searches can use them.
- If a carer deletes an object while the worker handles a frame, the sighting of this object is dropped. The image is not marked `failed`.
- A crash between the sighting and the status `done` leaves the image `processing`. The start-up recovery deletes its regions, and the sightings go by cascade.
  The worker processes the image again. The record `last_seen` can stay, and the new sighting replaces it with the same values.

## Security and privacy

- The log shows ids, counts and the words strong or weak. It never shows a name or a phrase.
- Retention keeps one full frame for each object that is in the memory. This is a decision of the owner. See "Known limits".

## Known limits

- The retention keeps a full frame of the home for each object, as long as the frame is the last sighting.
  A separate crop file would hold less (see the pipeline document), but needs a new file kind in the storage module.
- The time is the time of the server, not of the wearable. Frames that the wearable keeps and sends later get a wrong time.
- The rule has no check of the place. The data has no room. The pipeline document asks for it (technique 12).
- The sighting log is not pruned by age. A frame that expires takes its rows with it.
  With `retention.days = 0` the log grows without a limit. The rows are small.
- The rule counts the weak sightings of the same object only. A weak sighting does not show if a look-alike was nearby.
- The views come from one embedding model. If the model changes, an object is ignored until the carer adds its photos again.
- The thresholds are not tested with real photos. The tests use fake models with identical vectors.

## Tests

| File | What it checks |
|------|----------------|
| `tests/test_db_sightings.py` | The log, the count in a window, the record, the older sighting, the frame that stays, the cascades, bad values |
| `tests/test_backend_memory.py` | The rule: strong, weak, the window, other objects, one frame twice, the limit 1, a deleted object, `load_objects` |
| `tests/test_backend_worker.py` | The whole step with a fake detector: a record is set, a look-alike blocks, a look-alike alone is not reported, a region that is not good, a matching error, retention keeps the frame |
| `tests/test_backend_settings.py` | The keys of `[memory]` |
