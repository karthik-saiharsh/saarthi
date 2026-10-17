# Search (Q1)

## Purpose

The search answers one question: where did the system last see my object?
It is stage Q1 of `docs/guide/DINO-Based Object Recall Pipeline.md`.

Speech is part of this flow: a recognizer before the resolver, and a description and a voice after the answer (see `docs/backend/Voice.md`).
When no enrolled object fits the words, the open search looks for objects that nobody registered (see `docs/backend/Open Search.md`).

The user can say which object in two ways. The user can choose it from a list.
The user can also write it in any words, for example "the blue thing I use to drink stuff".
The search has two parts:

1. The **resolver** turns the words of the user into one object (`backend/query.py`).
2. The **search service** finds where and when the object was seen last (`backend/search.py`).

## Where it fits

```
words ──▶ QueryResolver ──▶ object ──▶ SearchService.find ──▶ Answer
                │                            │
         ai: query encoder            db: last_seen, then the archive
                                      ai: the matching of [matching].mode
```

- The HTTP routes (`backend/http_search.py`) and the UI adapter (`UiFindService`) call both parts.
- The search reads data only. It never writes to the database, the index or the storage.
- The search uses the same matching rule as the ingestion: the mode `evidence` (`AiEngine.score_evidence`), `keys` (`AiEngine.match_keys`) or `tau` (`AiEngine.match_regions`).
  A look-alike, a margin or a phrase works in the same way in both places.

## The resolver

`QueryResolver.resolve(text)` finds the object that the words mean.

1. It cleans the words. It removes extra spaces. It refuses words that are empty, longer than `query.max_chars`, or not printable.
   The error is `InvalidQueryError`.
2. It takes the objects that a search can find: `ready` objects that are not look-alikes.
3. It writes a description of each object: `name: category phrase, synonyms`.
   Example: `Blue bottle: a bottle, a water bottle`.
4. The query encoder turns the words and each description into a vector (see `docs/ai/AI Ingestion Module.md`).
   The resolver encodes a description one time and keeps the vector.
5. The score of an object is the cosine between the vector of the words and the vector of its description.

| Result | When | What the caller does |
|--------|------|----------------------|
| `matched` | The best score is at least `matched_min_score`, and at least `matched_min_gap` above the second best | Use this object |
| `ambiguous` | No object is sure, but the best score is at least `suggest_min_score` | Show up to `max_candidates` objects. The user chooses |
| `none` | The best score is below `suggest_min_score`, or no object can be found | Run the open search. If it finds nothing, tell the user that no object fits |

The user always has the list of objects. A wrong guess costs one tap.
The resolver never returns an object that is not in the gallery.

### Why a sentence encoder and not a language model

The owner asked for a small model that understands free text. The task is a choice between at most 50 known objects.
A sentence encoder (`sentence-transformers/all-MiniLM-L6-v2`, about 22 million parameters) is a good fit:

- It gives a score for each object. It cannot invent an object.
- It is small (about 91 MB) and fast. It runs on the device.
- The gap between the best and the second best score tells how sure it is.

A real test with the real model and 10 example objects gave the right object first for these words:

| Words | Best object | Score of the best | Score of the second |
|-------|-------------|-------------------|---------------------|
| "I can't find this blue thing I use to drink stuff" | Blue bottle | 0.54 | 0.25 |
| "where are my glasses" | Reading glasses | 0.62 | 0.24 |
| "I need to unlock the front door" | House keys | 0.30 | 0.12 |
| "something I drink tea from" | Tea cup | 0.62 | 0.37 |

Words that fit nothing also get a score. "hello how are you" gave 0.24 for the nearest object.
So the resolver uses the gap, and it asks the user when it is not sure.

## The search service

`SearchService.find(object_id)` returns an `Answer`. The steps are:

1. The object must exist (`NotFoundError`). It must be `ready` and not a look-alike (`ObjectNotReadyError`).
2. **A fresh record.** If the `last_seen` record is younger than `search.stale_after_hours`, it is the answer. The archive is not searched.
3. **The archive.** Otherwise the search goes back in time, one window of `search.window_minutes` at a time, newest window first.
   - A window holds the good regions of `done` frames that have a vector of the present model.
   - Each frame is matched with the same mode as the memory (`[matching].mode`, see `docs/backend/Memory.md`). The first frame with an accepted match for the object is the answer.
   - In the mode `evidence`, a frame is read from the storage and scored only if one of its regions is a candidate for the object
     (DINOv2 similarity of at least `matching.candidate_min`). At most `search.evidence_max_frames` frames are scored.
     A frame that cannot be read is skipped.
   - The mode `keys` reads no frame. It scores the stored crop vectors of all boxes of each frame (`db.crop_vectors.list_keyed_regions`),
     the boxes of the broad detector too. A region with `quality_ok = False` (a bad crop, or inside a depiction: the db does not keep the reason) is weak at most.
     The modes `tau` and `evidence` compare the candidates of the detector only.
   - A weak match is checked against the reference photos of the object (stage I5, see `docs/backend/Verification.md`).
     If it passes, it counts as strong, and the answer has no "maybe". If it fails, the answer stays a "maybe".
   - With an old record, the search looks only for a frame that is newer than the record.
   - Without a record, the search goes back up to `search.max_scan_hours`.
4. **An old record.** If the archive has nothing newer, the old record is the answer. It is marked `stale`.
5. **Not seen.** Without a record and without a match, the answer has `found = False`.

The frame limit protects a slow device. A search compares at most `search.max_frames` frames, and scores at most `search.evidence_max_frames` frames in the mode `evidence`.
If it stops at a limit, the answer has `complete = False`. "Not seen" may then be wrong, and the page says so.

### The answer

| Field | Meaning |
|-------|---------|
| `found` | The system saw the object |
| `source` | `last_seen` (the record) or `archive` (a search in the frames) |
| `seen_at` | The time at which the server received the frame (UTC) |
| `age_seconds` | The time between the frame and the search |
| `image_id`, `box` | The frame and the place of the object in it |
| `similarity` | The cosine to the best view of the object |
| `strong` | False means that the match is weak. The caller says "maybe" |
| `in_hand` | The person held the object. The place is then the person |
| `stale` | An old record, and nothing newer was found |
| `complete` | False: the search stopped at `search.max_frames` |
| `newer_maybe` | A second place, or none: the newest weak sighting of the log that is newer than the answer, in another frame, and younger than `search.stale_after_hours`. It has `seen_at`, `age_seconds`, `image_id`, `box` and `in_hand` |

**The second place (`newer_maybe`, owner decision 2026-11-12).** A weak sighting changes the last-seen record only after a second weak
sighting within 60 s (see `docs/backend/Memory.md`). So the newest place of an object can be in the sighting log only. Example from the
stored data: the record showed the watch on the bed (frame 4), and the log had a newer, weak and correct sighting on the desk (frame 7).
The answer stays the same, and it also names the newer place as a "maybe". The voice says it with "I am not sure", and the page shows a
second picture. `SearchService.maybe_picture` cuts that picture. A false weak sighting can become a "maybe" place: the text always says so.

### The picture

`SearchService.get_picture(object_id, view, max_side)` makes a JPEG of the answer from the stored frame.

- `crop` (the default) shows the object with `search.crop_padding` of its size on each side. The user sees the place around it.
- `frame` shows the whole frame.
- The longest side is at most `max_side`.
- A picture is always made from the answer of the object. No function gives a frame by its number.

`find_with_picture` makes the answer and the `crop` picture with one search. The UI uses it.
If the file of the frame is missing, the answer stays and the picture is None.

## HTTP routes

`backend/http_search.py`. The routes use the same optional token and the same fixed messages as the other routes.
The token is checked first. A wrong id or a wrong option does not skip the check.

| Route | Action |
|-------|--------|
| `POST /api/search` | The body is JSON: `{"query": "..."}`. The body has at most 16 KiB. The answer has `status` (`matched`, `ambiguous`, `open` or `none`), `candidates` and, for `matched` and `open`, the `answer`. `open` is a weak answer of the open search, with `label` and `via` |
| `GET /api/objects/{id}/last-seen` | The answer for one object |
| `GET /api/objects/{id}/last-seen/picture` | A JPEG. `?view=crop` or `frame`. `?max_side=` from 32 to 1024 |

| Code | Meaning |
|------|---------|
| 400 | The body, the words or an option is not valid |
| 401 | The token is missing or wrong |
| 404 | The object does not exist, or the object was not seen (picture), or the file of the frame is missing |
| 409 | The object is a draft or a look-alike |
| 500 | Another error. The message shows no internal detail |

The answers and the pictures have `Cache-Control: no-store`. A picture also has `X-Content-Type-Options: nosniff`.

```
curl -X POST -d '{"query": "where is my blue bottle"}' http://<server>:8080/api/search
curl http://<server>:8080/api/objects/1/last-seen
curl -o answer.jpg "http://<server>:8080/api/objects/1/last-seen/picture?view=frame"
```

## Configuration

`backend/config.toml`. All values are start values. They are not tuned.

| Key (table `[search]`) | Default | Meaning |
|------------------------|---------|---------|
| `stale_after_hours` | 24 | A record is fresh for this time |
| `window_minutes` | 60 | The size of one window of the archive search |
| `max_scan_hours` | 72 | The search does not look further back than this |
| `max_regions_per_window` | 5000 | A window holds at most this number of regions. The oldest regions are left out |
| `max_frames` | 3000 | The search compares at most this number of frames |
| `crop_padding` | 0.25 | Space around the object in the picture, as a part of the size of its box |
| `verify_weak` | true | Check a weak hit of the archive against the reference photos (stage I5) |
| `verify_max_references` | 3 | The check compares with at most this number of reference photos |
| `evidence_max_frames` | 30 | The mode `evidence` scores at most this number of frames in one search. Each costs a read of the frame and a SigLIP run (about 0.5 s on the development CPU) |

| Key (table `[query]`) | Default | Meaning |
|-----------------------|---------|---------|
| `matched_min_score` | 0.35 | The best object is sure above this score |
| `matched_min_gap` | 0.10 | ...and this far above the second best |
| `suggest_min_score` | 0.20 | Below `matched_min_score`, the user chooses between the objects above this score |
| `max_candidates` | 3 | The most objects that the user chooses from |
| `max_chars` | 300 | A longer question is refused |

`query.suggest_min_score` must not be above `query.matched_min_score`.
The model and its revision are in `[models]` of `ai/config.toml`.

## Errors

| Error | Meaning |
|-------|---------|
| `InvalidQueryError` | The words are empty, too long or not printable |
| `NotFoundError` | The object does not exist |
| `ObjectNotReadyError` | The object is a draft, or a look-alike that the system never reports |

The route and the UI turn each error into a fixed message.

## Security and privacy

- The words of the user are private. The log never shows them. Tests check this.
- The log shows ids, counts, and the words `matched`, `ambiguous`, `none`, `found` and `stale`.
- The query encoder reads text only, on the device. Nothing goes to another service. The only network use is the first download of the model.
- The model is loaded from a fixed revision, and only from safetensors files. No code of the model repository runs.
- The search cannot show an arbitrary frame. Each picture belongs to the answer for an object.
- The names of objects are in the descriptions that the encoder reads. They stay in memory.

## First run with the real models

The test used the real application and the real models, through the HTTP routes. The machine had a CPU and no GPU.
The run is from before the calibration. The object had `tau` 0.418. The calibrated values are in `benchmarks/calibration/README.md`.
The photos were 10 first-person photos of a corridor and a balcony. A white speckled bottle is in 5 of them.

- **Enrollment.** The object "Speckled bottle" got 3 photos. The detector found the bottle in all 3 (about 8 seconds for each photo).
  The three crops were correct. The object was `ready` with `tau` 0.418 and `tau_strong` 0.468.
- **Frames.** Seven other photos went through `POST /api/images` and took 55 seconds in total.

| Frame | Content | Result |
|-------|---------|--------|
| 10 | The bottle in a hand. Not used for enrollment | A strong sighting, similarity 0.634, `in_hand` |
| 8 | The bottle, partly out of the picture and blurred by motion | Not matched. The regions have `quality_ok = False`. A later check found the reason: the crop blur check (the crops score 8 to 11, the minimum is 20). The best similarity was 0.511, above `tau` |
| 1, 4, 5 | A notebook, a balcony, a backpack and chairs | The detector made boxes with the label "a bottle". Their best similarity was 0.14 to 0.32. No sighting |
| 6, 7 | An empty corridor | No region |

- **Questions.** "where is my speckled bottle" scored 0.77 (`matched`). "where did I leave my water flask" scored 0.39 (`matched`).
  "I can't find the white thing with black dots that I drink water from" scored 0.32. This is `ambiguous`: the user confirms.
  "where are my keys", "where is my phone" and "hello how are you" gave `none`.
- **Answer.** The answer came from the `last_seen` record. The crop showed the bottle with its surroundings, and the frame view showed the scene.
- **The page** `/find` showed the same result with the real picture.
- **Restart.** A stop with Ctrl+C saved 12 vectors. The next start loaded the 12 vectors, and the recovery changed nothing.

This is a very small test. It shows that the chain works. It does not show how good the thresholds are.
The test found one picture problem, which is fixed (see `docs/ui/Find Page.md`).

## Known limits

- **The cutoffs are not tuned.** The values of `[query]` come from a test with 10 objects. A developer wrote their descriptions.
  Calibrate them with the real objects of a home.
- **Names and phrases matter.** A good name and good synonyms give better results. The carer writes them at enrollment.
- **Weak matches.** The archive search can return a match that is not strong. The answer marks it, and the page says "maybe".
  The check (I5) turns some of them into firm answers. A plain object, for example keys, seldom passes it.
  The search checks only the newest weak hit. If the check fails, the search does not look for an older strong hit.
- **The clock.** The time is the time of the server. A frame that the wearable sends late gets a wrong time.
  The page shows an age ("12 minutes ago"), not a time of day.
- **The blur check of the crop.** A region is not matched if its crop is blurred (`quality.crop_blur_min`). A wearable moves, so held objects are often blurred.
  In the first run, this check hid a correct sighting (photo 8). The first report of this limit named the edge of the picture. That was wrong: the edge was not the reason.
- **A held object.** If the person held the object, the place is the person. The answer says so with `in_hand`.
- **Cost.** The search matches each frame alone, because the newest match must win. It is exact and not fast.
  The mode `evidence` also reads each frame with a candidate and runs SigLIP on it. `search.evidence_max_frames` bounds this cost.
  The frame limit and the time limit bound it. The speed on the Raspberry Pi is not measured.
- **A long window.** If a window has more than `max_regions_per_window` regions, the oldest regions of it are left out.
- **One model.** The search compares only the vectors of the present embedding model. After a model change, an object needs new photos.
- **No rate limit** on the search route. The size limit and the local network protect it.

## Tests

| File | What it tests |
|------|---------------|
| `tests/test_db_search.py` | The window listing and the vector reading of the db |
| `tests/test_backend_search.py` | The record, the stale record, the archive, the windows, weak matches, the limits, look-alikes, draft objects, one search for the picture |
| `tests/test_backend_evidence.py` | The archive search in the mode `evidence`: a hit, only frames with a candidate are read, the limit `evidence_max_frames` |
| `tests/test_backend_search_verification.py` | The check of weak hits: a firm answer after a pass, a "maybe" after a fail, the cases without a check |
| `tests/test_backend_query.py` | The three results, the choice, the cache, bad words, no logging of the words |
| `tests/test_backend_http_search.py` | Every route, the picture sizes, bad options, the token, fixed messages |
| `tests/test_backend_ui_find.py` | The UI adapter with the real database |
| `tests/test_ai_query_model.py` (slow) | The real encoder: the vectors, the padding, four vague questions |
| `tests/test_backend_settings.py` | The keys of `[search]` and `[query]` |

The tests use fake models, except the slow file. The slow file needs the model download.
