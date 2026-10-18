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
2. Match the regions with the objects. A region with `quality_ok = False` is never matched.
   The mode is `[matching].mode` in `ai/config.toml` (see "The matching modes"). The default is `keys` (owner decision 2026-11-12).
   The mode `keys` matches before the regions are stored (see below).
3. Check each weak match against the reference photos of the object (stage I5, see `docs/backend/Verification.md`).
   A match that passes the check counts as strong.
4. Write each accepted match to the table `sighting`. The time is the time at which the server received the frame (`Image.captured_at`).
5. Apply the rule below. It decides if the match changes the record `last_seen` of the object.

## The matching modes

| Mode | How a region becomes a sighting |
|------|---------------------------------|
| `tau` | One test: the DINOv2 similarity must reach the `tau` of the object, with the margin `margin_min`. The detector phrase of the region must fit the object (`AiEngine.match_regions`) |
| `evidence` | Several independent signals give one score (`AiEngine.score_evidence`, `ai/evidence.py`). The detector phrase is not a gate |
| `keys` (default) | Two picture keys of the crop (SigLIP and the DINOv2 CLS token) with fixed bars, and a word vote (`AiEngine.match_keys`, `ai/keys.py`). It also scores the boxes of the broad detector |

**The steps of the mode `keys`** (`Worker._match`, `MemoryService.match_keys`, `MemoryService.record`):

1. **The boxes.** The worker stores all boxes of the frame in the table `region`: the candidates of the detector and the boxes of
   the broad detector (YOLOE, `source = broad`). All boxes are kept: a YOLOE box that overlaps a candidate is not dropped, because
   it can be sharp where the candidate is blurred (`benchmarks/recall/README.md`). The crop gate applies to all of them.
2. **The keys.** The worker makes the SigLIP crop vector and the DINOv2 key of every box, one time, and stores them
   (`db.crop_vectors`). The reference photos get the same two vectors (`ReferenceVectors`).
3. **The decision** (`docs/ai/AI Ingestion Module.md`, "Mode `keys`"). Strong: SigLIP 0.77 and DINOv2 0.42. Weak: SigLIP 0.72 and
   one more vote (DINOv2 0.40, or a detector score of 0.30 for a word of the object). A box with a bad crop (small, blurred, cut by the edge)
   is weak at most. A box inside a screen, frame or mirror is never a sighting. One box is one object, and a look-alike can take a box.
4. **The record.** Stages I5 and I6 as in the other modes (`MemoryService.record`).

The modes `tau` and `evidence` match the candidates of the detector only, as before. The mode `evidence` takes the stored SigLIP
vectors of its chosen regions, so no box gets a second SigLIP run.

`Sighting.similarity` is the SigLIP key and `Sighting.margin` the SigLIP key minus the best SigLIP key of another object.
The bars are the same for all objects. They come from the replay of `benchmarks/recall` (phase R1a). They are start values.
Without the scene model the mode is `tau`.

**Why the mode `evidence` exists.** One threshold is a single point of failure. In a real test with a watch, a headset and a book,
all reference photos were taken on the same table. The crops of the watch and the headset were mostly table, so DINOv2 found them similar (0.64).
The calibration then raised `tau` of both objects to 0.674. No frame reached it, and the search never found them.
The detector also called the watch on a wrist "a headset", so the phrase rule blocked the true box.

**The steps of the mode `evidence`** (`MemoryService.remember`):

1. **Candidates.** A region is a candidate if its DINOv2 similarity to a tracked object is at least `matching.candidate_min` (0.20).
   This bar is low on purpose: the watch on a wrist scored 0.26 and was correct. At most `matching.max_regions` (8) regions are scored.
2. **SigLIP crops.** The worker cuts each candidate from the decoded frame (`matching.crop_padding`) and makes a SigLIP vector.
3. **The score.** For each candidate and each tracked object (see `docs/ai/AI Ingestion Module.md`, "Stage I4"):

   `score = bias + w_meaning * min(logit(meaning), meaning_cap) + w_visual * (visual - visual_centre) / visual_scale + w_dino * (dino_agree - dino_centre) / dino_scale + w_margin * min(margin, margin_cap) / margin_scale`

   | Signal | Meaning |
   |--------|---------|
   | `dino_agree` | The mean of the best DINOv2 cosine of the 2 best reference photos. One lucky view is not enough |
   | `margin` | The best DINOv2 cosine to this object minus the best to any other enrolled object |
   | `visual` | The same agreement with SigLIP vectors of the reference crops. SigLIP is a second model with other errors |
   | `meaning` | The SigLIP share of the words of the object against the words of the other objects and of common things (`matching.contrast_phrases`) |

4. **The decision.** `score >= score_sure` (6.0): strong. `score >= score_maybe` (3.0): weak. Else: no sighting.
5. **Two rules that only refuse.** A region with `quality_ok = False` is not scored. A region with `margin < margin_min` of the object is refused,
   so a look-alike that is as close as the object still blocks the match.

The positive part of `meaning` is capped (`meaning_cap`). The meaning tells the kind of thing, not which one.
In the test, another book scored 0.98 for "book". It must not make a match, and the cap keeps it below `score_maybe`.

**The reference vectors.** `backend/references.py` (`ReferenceVectors`) makes the SigLIP vectors of the confirmed reference crops from the stored photos.
They stay in memory, and they are made again only when the set of confirmed photos of an object changes. The worker and the search share one instance.
A photo that cannot be read is left out. A tracked object without a readable photo is not matched in this mode. A look-alike needs no reference vectors:
it takes part only in the margin.

**Fallback.** Without the decoded frame, or with the scene model off (`scene.enabled = false`), the memory uses the mode `tau`.
The keypoint check (stage I5) and the rule below do not change with the mode.

**Measured** (`benchmarks/matching/README.md`): on 15 real frames, the mode `tau` found 0 of 4 frames with the watch and 1 of 2 with the book.
The mode `evidence` found 4 of 4 and 2 of 2, with no false sighting. On the 14 photos of the speckled bottle (not used to set the values), both modes found the 2 good frames.
The headset was not found in either mode: its reference photos do not look like the frames (see "Known limits").
The step costs about 1.8 s for a frame with 7 candidates on the development CPU.

## The rule (hysteresis)

| Sighting | What happens |
|----------|--------------|
| Strong (mode `tau`: `similarity >= tau_strong` and `margin >= margin_strong`; mode `evidence`: `score >= score_sure`), or weak and verified | The record changes at once |
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
- The values of `[matching]` come from 3 objects in 15 frames and one hold-out object. They are start values. Measure again with more objects.
- The score is not stored. The `similarity` of a sighting is the DINOv2 similarity in both modes. The log line of a sighting shows the score.
- Reference photos that all show the same table make two small objects look alike to DINOv2. Photos of each object in the places where it is used help most.
  In the test, the headset was never found: all its photos were close-ups on one table, and the frames showed it from the side or in a hand.
- A single weak sighting does not change the record. In the test, the book in frame 10 was weak and alone, so the record stayed on the older strong sighting.

## Tests

| File | What it checks |
|------|----------------|
| `tests/test_db_sightings.py` | The log, the count in a window, the record, the older sighting, the frame that stays, the cascades, bad values |
| `tests/test_backend_memory.py` | The rule: strong, weak, the window, other objects, one frame twice, the limit 1, a deleted object, `load_objects` |
| `tests/test_backend_worker.py` | The whole step with a fake detector: a record is set, a look-alike blocks, a look-alike alone is not reported, a region that is not good, a matching error, retention keeps the frame |
| `tests/test_backend_settings.py` | The keys of `[memory]` |
| `tests/test_backend_keys.py` | The mode `keys`: the DINOv2 keys of the references, all boxes matched, only a broad box with a sighting becomes a region, the shared SigLIP vectors of the open search, the hysteresis, an error of the broad detector, the archive search |
| `tests/test_backend_evidence.py` | The mode `evidence`: the reference vectors (only confirmed photos, kept, made again after a change, a missing file), the memory (strong, weak and the window, no picture, no readable reference), the archive search |
