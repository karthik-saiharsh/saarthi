# AI Ingestion Module

## Purpose

The ai module turns one stored image into data for the database.
It does these things:

1. It checks the quality of the image (stage I1).
2. It finds boxes that can show an object that the user may lose (stage I2).
3. It makes one vector for each box (stage I3).
4. It matches the boxes with the enrolled objects (stage I4).
5. It checks a borderline sighting against the reference pictures of an object (stage I5).
6. It turns the words of a user into a vector, to find the object that the user means (stage Q1).
7. It scores the room and the surface of a place from fixed word lists (stage Q2).
8. It turns speech into text (stage Q0).
9. It finds and names objects that nobody registered (the broad detector), and makes SigLIP vectors for them.
10. It rewrites a description into object names, and writes a sentence about a place (the language model).

The design follows `docs/guide/DINO-Based Object Recall Pipeline.md`.
Items 8 to 10 are described in `docs/backend/Voice.md` and `docs/backend/Open Search.md`.
The enrollment stages E2 to E4 are in the section "Enrollment stages E2 to E4".
The stages that use the results are in `docs/backend/Memory.md` and `docs/backend/Search.md`.

## Where it fits

The `backend` module controls the flow. The ai module gives one function for each stage.

```
worker (backend):
  image_row = db.claim_next_pending()
  image     = engine.open_image(storage.read_image(image_row.file_path))
  report    = engine.assess_quality(image)
  db.set_image_analysis(id, report.phash, report.blur_score)
  if engine.is_blurry(report):                                   db.mark_image_rejected(...)
  if engine.is_near_duplicate(report.phash, db.list_recent_phashes()): db.mark_image_rejected(...)
  candidates = engine.detect_candidates(image, include_generic=every_nth_frame)
  vectors    = engine.embed_regions(image, [c.box for c in candidates])   # before the rows: an error leaves nothing
  regions    = db.add_regions(id, [to_new_region(c) for c in candidates])
  db.store_region_vectors([r.id for r in regions], vectors, engine.embedding_model)
  db.mark_image_done(id)
```

The worker in `backend/worker.py` does these steps (see `docs/backend/Worker.md`).

Every candidate that the backend stores must also get a vector.
The start-up recovery of the db module sets an image back to `pending` if one of its regions has no vector.

`Candidate` has the same fields as `db.api.NewRegion`. The backend converts one to the other.
The ai module does not import the db module.

## Interface

| Helper | Task | Result |
|--------|------|--------|
| `AiEngine(config=None)` | Load up to eight models. The detector, the embedder and the query encoder always load. The verifier, the scene model, the speech model, the broad detector and the language model load only if their `enabled` key is true | The engine. Call it one time at start-up |
| `open_image(data, max_pixels)` | Decode image bytes. Apply the EXIF orientation. Convert to RGB | A PIL image. Raises `ImageError` |
| `engine.open_image(data)` | The same, with `image.max_pixels` of the config | A PIL image |
| `engine.assess_quality(image)` | Blur score and perceptual hash | `QualityReport` |
| `engine.is_blurry(report)` | The blur score is below `quality.blur_min` | `bool` |
| `engine.is_near_duplicate(phash, recent_phashes)` | A recent hash is `quality.phash_max_distance` bits away or closer | `bool` |
| `engine.detect_candidates(image, phrases=None, small_phrases=None, include_generic=False, tiling=None)` | Find candidate boxes | List of `Candidate`, best score first |
| `engine.embed_regions(image, boxes)` | One vector for each box | `float32` array, shape `(n, 384)`, unit length |
| `engine.match_regions(vectors, regions, objects)` | Match the regions of a frame with the `ready` objects (stage I4) | List of `Sighting` |
| `engine.verify_match(references, candidate)` | Compare a candidate picture with reference pictures (stage I5) | `VerifyResult` |
| `engine.verification_crop(image, box)` | Cut the picture that the verifier compares: the box with `verification.crop_padding` around it | A PIL image |
| `engine.verification_enabled` | The verifier is loaded | `bool` |
| `engine.score_rooms(image)` | The probability of each room of `scene.rooms` for a whole picture (stage Q2) | `dict`, the values add up to 1 |
| `engine.score_surfaces(image, box)` | The surfaces of `scene.surfaces` that fit the place of the object, the best first | List of (word, probability) |
| `engine.scene_enabled`, `engine.scene_model` | The scene model is loaded. Its name, for the cache | `bool`, `str` |
| `engine.encode_texts(texts)` | One vector for each text (stage Q1) | `float32` array, shape `(n, 384)`, unit length |
| `engine.query_model` | Name of the query encoder, for example `all-MiniLM-L6-v2@1110a243` | `str` |
| `engine.embedding_model` | Name for the database, for example `dinov2-small@ed25f3a3` | `str` |
| `engine.embedding_dimension` | Length of one vector | `int` |
| `engine.detect_phrases(image, phrases)` | One detector pass with the words of a user (the open search, mode `verified`) | List of `RawDetection` |
| `engine.transcribe(samples)` | Speech to text with Whisper base (stage Q0). `speech_enabled`, `speech_max_seconds`, `speech_sample_rate` describe the input | `str` |
| `engine.detect_open(image)` | The broad detector (YOLOE, prompt-free) | List of `OpenDetection`, the most confident first |
| `engine.embed_open_crops(image, boxes)`, `engine.embed_open_phrases(phrases)` | SigLIP vectors of crops and of phrases, for the open search | `float32` array, 768 numbers each, unit length |
| `engine.open_pair_scores(crops, phrases)` | The SigLIP probability of each pair | Array of shape (crops, phrases) |
| `engine.open_detection_enabled`, `open_vector_dimension`, `open_vector_model` | The broad detector is loaded. The size and the name of the open vectors | `bool`, `int`, `str` |
| `engine.rewrite_query(words)` | The language model turns a description into up to 3 object names | List of `str` |
| `engine.describe_scene(picture, facts)` | The language model writes one sentence about the place. The backend checks it | `str` |
| `engine.language_enabled`, `description_model` | The language model is loaded. The name for the stored sentences | `bool`, `str` |

`detect_candidates` uses `detection.default_phrases` when you give no phrases.
`include_generic` adds the generic pass. `tiling` switches the tiled pass for this call. The value `None` uses the config.
The backend chooses both switches for each frame. The ai module does not count frames.

### Data objects

| Object | Fields |
|--------|--------|
| `Box` | `x0`, `y0`, `x1`, `y1` in pixels. `x1` and `y1` are exclusive |
| `QualityReport` | `blur_score`, `phash` (16 hex characters) |
| `MatchRegion`, `MatchObject`, `Sighting` | The input and the result of the matching (see "Stage I4") |
| `Candidate` | `box`, `det_score`, `source` (`full`, `tile` or `generic`), `phrases`, `in_hand`, `quality_ok` |

`det_score` is for analysis only. Do not use it to decide the identity of an object.

## Stage I1: quality

- **Blur score.** The variance of a 3 x 3 Laplacian on a grey copy of the image.
  The copy is scaled so that its long side is `quality.analysis_max_side`. The score then does not depend on the camera resolution.
  A low score means a blurry image.
- **Perceptual hash.** A 64-bit difference hash, written as 16 hex characters.
  It fits the column `Image.phash`.
- **Near duplicate.** The module compares the hash with the recent hashes that the backend gives.
  SQLite cannot compare hashes, so the backend reads them with `db.list_recent_phashes()`.

## Stage I2: detect candidates

The detector is Grounding DINO. It finds boxes for text phrases.
The module runs up to three kinds of pass.

| Pass | Switch | Phrases | Source value |
|------|--------|---------|--------------|
| Full frame | Always | The vocabulary, a hand, and the depiction phrases | `full` |
| Tiled | `detection.tiling_enabled`, or `tiling=True` | The small phrases only. The image is split in 2 x 2 tiles with an overlap | `tile` |
| Generic | `include_generic=True` | "an object" | `generic` |

Each pass is a full run of the model. The tiled pass costs four runs for each prompt.

The prompts are split by `detection.chunk_size` and by the limit of 256 text tokens.
Every prompt also holds the hand phrase (and the depiction phrases in the full-frame pass).

### Post-processing

`select_candidates` does these steps in this order:

1. **Map labels to phrases.** The detector joins all phrases that it links to a box in one label,
   for example `a mobile phone a mirror`. The module splits the label at the articles and matches each part.
   The match uses the same words, then the same head noun, then words that are inside a phrase.
2. **Keep the best boxes.** The module keeps `detection.max_per_phrase` boxes for each main phrase.
3. **Clip and filter.** It clips each box to the image. It drops a box with a side below `detection.min_box_side`.
   It drops a box that covers more than `detection.max_frame_fraction` of the frame.
   A box of this size is a detector error. It could hide or flag every object in the frame.
4. **Sort the boxes.** A box that names an object phrase is an object, even if its label also names a hand or a screen.
   A box without an object phrase is a hand or a depiction (a screen, a frame, a mirror).
5. **Merge duplicates.** Class-agnostic NMS at `detection.nms_iou`. The merged box keeps the union of the labels.
6. **Mark depictions.** An object with at least `detection.depiction_inside` of its area inside a depiction box gets `quality_ok = False`.
   The module does not drop it, because the detector can box a book cover as a screen (see "Known limits and open items").
7. **Flag hands.** An object gets `in_hand` if a hand box covers more than `detection.hand_overlap` of its area.

Hand boxes and depiction boxes are never returned as candidates. An object inside a depiction box is returned with `quality_ok = False`.

### Crop quality

`apply_crop_quality` sets `quality_ok`. A candidate is good if all three are true:

- It is not inside a depiction box (step 6 of the post-processing). `apply_crop_quality` never sets this flag back to True.
- Its short side is at least `quality.min_crop_side` pixels.
- It does not touch the edge of the image.
- The blur score of its crop is at least `quality.crop_blur_min`.

A candidate with `quality_ok = False` is still returned. The archive keeps it.
It must still get a vector (see "Where it fits").

## Stage I3: embed

The embedder is DINOv2 small. The vector has 384 numbers.

- **Large boxes.** One model run on the whole frame gives a grid of patch vectors (one for each 14 x 14 pixels).
  The frame is scaled to `embedding.frame_short_edge`. The long side is cut to `embedding.frame_long_edge_max`.
  The vector of a box is the mean of the patches whose centres are in the box.
  If no centre is in the box, the module uses the patch under the centre of the box.
- **Small boxes.** A box with a side below `embedding.small_box_min_side` is embedded again from a padded square crop.
  The side is measured in model pixels: the box is scaled with the frame first (`is_small_box`).
  So the cutoff means the same number of patches for every camera resolution.
  The crop has `embedding.crop_padding` of padding on each side. It is scaled to `embedding.crop_size`.
  The module uses the same mean-of-patches pooling, so all vectors stay in one space.
  Crops go through the model in batches of 16.
- Each vector is scaled to unit length. The inner product of two vectors is then their cosine similarity.

The vectors are not compared with a model change. A threshold is not portable between two models.
The name in `Region.embedding_model` tells which model made a vector.

## Stage I4: match

`engine.match_regions(vectors, regions, objects)` compares the region vectors of one frame with the enrolled objects.
It is pure numpy (`ai/matching.py`). It does not read the database. The backend gives it the vectors, a `MatchRegion` for each region
(`phrases`, `quality_ok`) and a `MatchObject` for each `ready` object (the views, `tau`, `tau_strong`, `margin_min`, the phrases, `tracked`).

1. **Skip.** A region with `quality_ok = False` is not matched. This includes an object inside a screen, a frame or a mirror.
2. **Candidate objects.** The phrases of the region choose the objects: the region and the object need one phrase in common.
   A region without a phrase, and a region from the generic pass ("an object"), are compared with all objects.
3. **Similarity.** The highest cosine between the region and one view of the object (the owner chose the maximum, not a mean of the best views).
4. **Margin.** The similarity minus the highest similarity to any other object. The other objects are the negatives of the calibration:
   all other `ready` objects, also the look-alikes, whatever their phrases are. An object with no other object has the margin `similarity - 0`.
5. **Accept.** `similarity >= tau` and `margin >= margin_min` of the object. The sighting is **strong** if also
   `similarity >= tau_strong` and `margin >= enrollment.margin_strong`.
6. **One sighting for each object.** The region with the highest similarity wins.

An object with `tracked = False` (a look-alike) is a negative only. It never gets a sighting.
A look-alike that is as close as the owner's object makes the margin too small, so no sighting is made. This is on purpose.

The result is a list of `Sighting` (`object_id`, `region_index`, `similarity`, `margin`, `strong`), sorted by object id.
A bad vector or view (not finite, not unit length, another size, no views) raises `MatchingError`.

The worker calls it through `MemoryService` after it stores the regions (see `docs/backend/Memory.md`). The rule that updates the last-seen record is there.
The ratio test and the one-to-one assignment of the pipeline document are not built.
The backend checks a weak sighting with stage I5 (see `docs/backend/Verification.md`).

## Stage I5: verify

`engine.verify_match(references, candidate)` checks a borderline sighting. It counts the keypoints that match between the candidate and each reference picture.
The backend gives the pictures. It cuts all of them with `engine.verification_crop`, so the cuts are alike. The full design is in `docs/backend/Verification.md`.

- The model is LightGlue with SuperPoint (`ETH-CVG/lightglue_superpoint`, about 55 MB). It finds keypoints in both pictures and matches them.
- The module compares the references in the given order. It stops at the first reference with at least `verification.min_matches` matches.
- The result is `VerifyResult`: `verified`, `matches` (the highest count) and `compared`. Without references, `verified` is false.
- The processor resizes each picture to 640 x 480 pixels. One comparison takes about 1.7 seconds on the development machine.
- **Licence.** The weights are for academic or non-profit non-commercial research use only. The owner confirmed that this project is for academic research.
  Change the verifier if the use changes.
- With `verification.enabled = false`, the module does not load the model. `verify_match` must not be called then.
- An error of the model raises `InferenceError`. A model that cannot be loaded raises `ModelLoadError`.

## Stage Q2: the place of an object

`engine.score_rooms(image)` and `engine.score_surfaces(image, box)` describe the place of an object with words from fixed lists.
The planned voice uses them (see `docs/backend/Voice.md`). The backend builds the sentence and chooses the room from neighbouring frames.

- The model is SigLIP (`google/siglip-base-patch16-224`, Apache-2.0, about 813 MB). It scores a picture against one sentence for each word.
  The sentence is `scene.prompt` with the word. The scores become probabilities that add up to 1.
- The lists are `scene.rooms` and `scene.surfaces` in `config.toml`. The description can only use these words.
  A wrong word from a closed list is a safer error than a text that a model invented (see `benchmarks/scene/README.md`).
- The model encodes the sentences of the lists one time, when it loads. Then a picture costs one run of the image side of the model.
  The result equals a full run of the model. A test checks this. One picture takes about 0.3 seconds on the development machine.
- **The room** comes from the whole picture. **The surface** comes from the box of the object with space around it:
  `scene.surface_context` times the size of the box on each side and below it, and half of this above it.
- The scores are not a safe sign that a word is right. In the test, a floor was called "a bathroom" with a probability of 0.93.
  The backend must take the room from neighbouring frames, and must say nothing when they disagree.
- With `scene.enabled = false`, the module does not load the model. The two methods must not be called then.
- The model reads pictures and fixed texts only. It uses the safetensors file and the fast tokenizer file of the repository. It needs no new library.
- An error of the model raises `InferenceError`. A model that cannot be loaded raises `ModelLoadError`.

## Query encoder (stage Q1)

`engine.encode_texts(texts)` turns texts into unit-length vectors. The backend uses it for two kinds of text:
the words of a user, and the description of each enrolled object (`docs/backend/Search.md`).
The cosine between two vectors is the score of a pair.

- The model is `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, about 22 million parameters, about 91 MB). The vector has 384 numbers.
- The module cuts a text after `query.max_tokens` tokens (128, the length that the model learned with).
- The vector is the mean of the token vectors. The padding does not count. The module scales it to unit length.
- The module loads the model one time, with the other models. One lock protects it.
- These vectors are not comparable with the vectors of the regions. Compare them with each other only.
- The encoder reads text only. It never sees an image. It runs on the device, so the words stay on the device.

An error raises `InferenceError`. A model that cannot be loaded raises `ModelLoadError`.

## Enrollment stages E2 to E4

The enrollment gives the system the objects that it looks for (see `docs/backend/Enrollment.md`).
The module gives three helpers. The `backend` module calls them.

- **E2 `engine.locate_object(image, category_phrase, synonyms)`.** One detector pass with the phrases of the object.
  The hand and depiction phrases are not in the prompt (`context_phrases=False`), because a depiction box can cover the object.
  The post-processing is the usual one. The result is the best box and its score, or `None`.
  The detector also gives a box for an object that is not in the photo, so the carer must check the crop.
- **`engine.detect_phrases(image, phrases)`.** One detector pass with the words of a user. The open search in the mode `verified` uses it
  (`docs/backend/Open Search.md`). Each phrase is made lowercase, loses its periods and is cut to `[detection].phrase_max_chars` (60).
  A duplicate phrase is dropped. The prompt has no hand phrase, no depiction phrase and no tiles (as in `locate_object`).
  The result is the list of raw `RawDetection` objects (box, score, label). There is no merge and no filter, so the caller sees every box
  that is at or above `box_threshold`. If no phrase is left, the result is empty and the detector does not run.
  The detector raises `InferenceError` when a pass fails.
- **E3 `engine.embed_views(image, box)`.** One vector for each view of `[enrollment].augmentations`:
  `original`, `box_smaller` and `box_larger` (the box scaled by `box_scale`), and `brighter` (`brightness`).
  The three box views use one model run.
- **E4 `engine.calibrate_object(views, photo_ids, others)`.** Pure numpy functions in `calibration.py`.
  `own` is the list of similarities between views of different photos. `tau = max(mean - tau_std_factor * std, tau_min)`.
  With negatives, `tau = max(tau, neg_max + margin_min)`. `tau_strong = tau + tau_strong_offset`. No value is above 1.0.
  The result has a note when another object is too similar.
  It has another note when the photos of the object look very different from each other.

The table `[enrollment]` in `config.toml` has `augmentations`, `box_scale`, `brightness`, `tau_std_factor`, `tau_strong_offset`, `tau_min`,
`margin_min`, `margin_strong` and `min_photos`.

## Configuration

All settings are in `ai/config.toml`. The start values come from the pipeline document.

| Table | Keys |
|-------|------|
| `[models]` | `detector_id`, `detector_revision`, `embedder_id`, `embedder_revision`, `query_encoder_id`, `query_encoder_revision`, `verifier_id`, `verifier_revision`, `scene_id`, `scene_revision`, `speech_id`, `speech_revision`, `language_id`, `language_revision`, `open_detector_file`, `open_detector_url`, `open_detector_sha256`, `device` (`auto`, `cpu` or `cuda`), `cpu_threads`, `local_files_only` |
| `[image]` | `max_pixels` |
| `[quality]` | `blur_min`, `analysis_max_side`, `phash_max_distance`, `min_crop_side`, `crop_blur_min` |
| `[detection]` | `box_threshold`, `text_threshold`, `input_short_edge`, `max_per_phrase`, `chunk_size`, `phrase_max_chars`, `nms_iou`, `max_frame_fraction`, `min_box_side`, `hand_overlap`, `depiction_inside`, `default_phrases`, `small_phrases`, `hand_phrase`, `depiction_phrases`, `generic_phrase`, `tiling_enabled`, `tile_overlap` |
| `[embedding]` | `frame_short_edge`, `frame_long_edge_max`, `crop_size`, `small_box_min_side`, `crop_padding` |
| `[query]` | `max_tokens` |
| `[verification]` | `enabled`, `min_matches`, `match_threshold`, `crop_padding` |
| `[scene]` | `enabled`, `prompt`, `rooms`, `surfaces`, `surface_context` |
| `[speech]` | `enabled`, `max_seconds`, `max_new_tokens` |
| `[open_detection]` | `enabled`, `confidence`, `max_boxes`, `input_size`, `min_box_side`, `crop_padding` |
| `[language]` | `enabled`, `max_input_chars`, `max_new_tokens`, `max_names`, `rewrite_system`, `rewrite_examples`, `describe_system`, `describe_max_new_tokens`, `describe_image_side` |

- `embedding.frame_short_edge`, `embedding.frame_long_edge_max` and `embedding.crop_size` must be multiples of 14.
- Each phrase in `detection.small_phrases` must also be in `detection.default_phrases`.
- `image.max_pixels` must stay equal to `limits.max_pixels` in `storage/config.toml`.
- The dimension of the vectors (384) must equal `vectors.dimension` in `db/config.toml`.
  A test compares the two values.
- `detection.default_phrases` is a placeholder list. The product owner sets the real list.

## Models and downloads

| Task | Model | Size |
|------|-------|------|
| Detect | `IDEA-Research/grounding-dino-tiny` | About 690 MB |
| Embed | `facebook/dinov2-small` | About 90 MB |
| Understand the words of a user | `sentence-transformers/all-MiniLM-L6-v2` | About 91 MB |
| Check a borderline sighting | `ETH-CVG/lightglue_superpoint` (non-commercial research use only) | About 55 MB |
| Describe the place, and the vectors of the open search | `google/siglip-base-patch16-224` | About 813 MB |
| Turn speech into text | `openai/whisper-base` (Apache-2.0) | About 290 MB |
| Find objects that nobody registered | YOLOE `yoloe-11s-seg-pf.pt` (AGPL-3.0). Downloaded into `ai/models/`, SHA-256 checked | About 28 MB |
| Rewrite a description, write a sentence about a place | `Qwen/Qwen3.5-0.8B` (Apache-2.0) | About 1.75 GB |

All eight models together are about 3.8 GB.

- The first start downloads the models to the Hugging Face cache, `~/.cache/huggingface/hub/`.
  Set the variable `HF_HOME` (or `HF_HUB_CACHE`) to use another folder.
- Each model has a fixed revision in `config.toml`. Change it only after a test run.
- The module loads only safetensors files. It never runs code from the model repository.
- After the first download, set `models.local_files_only = true`. The module then never uses the internet.
- The cache is per user and per machine. For the Raspberry Pi, run the application one time with internet, or copy the `hub/` folder.
- The module loads the models one time. Loading the detector and the embedder takes about 6 seconds on the development machine.
  The query encoder takes about 2 seconds more (measured once, with the files in the cache). The verifier was not timed.
- One lock protects each model, so two threads cannot use the same model at the same time.

## Errors

| Error | Meaning |
|-------|---------|
| `AiError` | Base class of all errors below |
| `AiConfigError` | `config.toml` is missing or has a wrong value. The message names the key |
| `ModelLoadError` | A model cannot be loaded |
| `ImageError` | The file is not an image, is broken or truncated, or has too many pixels |
| `InferenceError` | A model failed while it worked on an image |
| `MatchingError` | A vector or a view of the matching is not usable |

## Security and privacy

- Images and vectors stay on the server. The module sends nothing to another service.
  The only network use is the first download of the models.
- The module refuses an image that has more than `image.max_pixels` pixels before it decodes the image.
- The logs show counts, sizes and times only. They do not show image content or file names.
- The module does not run code from a model repository.

## First measurements

The development machine has an Intel Core Ultra 9 CPU and no GPU.
The test photo is 968 x 728 pixels. Each time is the median of one run after a warm-up run.

| Stage | Time |
|-------|------|
| Open the image, quality check | 0.02 s |
| Detect: full frame (10 phrases) | 9.5 s |
| Detect: full frame and tiled pass | 49 s |
| Detect: full frame and generic pass | 19.7 s |
| Detect: all three passes | 59 s |
| Embed 13 boxes | 1.0 s |

- An image takes about 10 s with the default passes. The detector is the slow part.
  A 1920 x 1080 frame also took about 10 s, because the detector scales its input.
- The tiled pass costs about five times the full-frame pass. It is too slow to run for every frame on this machine.
  The Raspberry Pi will be slower. Measure it there before you choose the frame rate.
- Run `benchmarks/ingestion/benchmark.py PHOTO` to measure your own hardware.

### What the module finds in the test photo

The photo shows a phone in a pink case on a book, a charger with a cable, and a green pen.

- The full-frame pass finds 13 candidates. The phone is one of them, with the correct box.
- The generic pass finds the pen. No phrase in the vocabulary names a pen.
- The charger gets the labels "a set of keys" and "a remote control". The cable gets "a pair of glasses".
  The boxes are correct. The vocabulary has no phrase for a charger or a cable.
- The module gives about 13 overlapping boxes for 5 objects.
  The NMS value of 0.7 only merges boxes that are almost the same.

## Known limits and open items

- **Blur threshold.** The owner set `blur_min = 0`, so the gate is off until real frames exist.
  The pipeline start value of 100 calls the test photo blurry. The photo scores 94.
  The score also depends on the JPEG compression: the same photo saved again at another quality scored 117.1.
  Calibrate the value with 20 or more real frames from the wearable, both good and bad.
- **Depiction rule.** The owner chose to mark, not drop. In a real test, a phone on a book cover got the label
  "a phone a smartphone a mirror" and a "television screen" box covered it. The phone now stays in the archive with `quality_ok = False`.
  The later last-seen step must not use a candidate with `quality_ok = False`.
  The cost: a real object on a book cover also cannot update the last-seen record. A better test for a screen is an open item.
- **Vocabulary.** The phrases are a placeholder. The labels are only as good as the phrases.
- **Duplicate boxes.** The value `nms_iou = 0.5` came from the calibration (`benchmarks/calibration/README.md`). One box for each object needs a decision after the vocabulary is real.
- **Calibration.** The values were tuned on 14 photos with one object. The README of `benchmarks/calibration` lists what was set, what was kept, and what the data could not decide.
  `text_threshold` was not tuned.
- **Speed on the Raspberry Pi.** It is not measured. The installation of the aarch64 wheels of torch is not tested.
- **Not tested.** `device = "cuda"`, `local_files_only = true`, `cpu_threads`, and two threads that call a model at the same time.
- **Stored image.** The storage module saves the image again as a JPEG. On the test photo, the number of candidates changed from 13 to 14.
  12 of the 13 boxes stayed the same (IoU of 0.8 or more). The blur score did not change.
- **Orientation.** `open_image` applies the EXIF orientation. The storage module applies it before it removes the metadata,
  so the stored file has the same pixel grid that the boxes in the database use.

## Not in this module

These parts are not built yet:

- A check of a sighting with a vision-language model, for plain objects. Stage I5 uses keypoints, so it fits textured objects.
- Techniques from the pipeline document: pooling over a mask (SAM), background-free enrollment, context negatives, and the calibrated accept and reject model.
  The owner decided that enrollment uses boxes only.

## Tests

| File | What it tests |
|------|---------------|
| `tests/test_ai_quality.py` | Blur score and perceptual hash |
| `tests/test_ai_geometry.py` | IoU, overlap, clip, NMS, tile grid |
| `tests/test_ai_postprocess.py` | Label matching, every post-processing rule, crop quality, prompt planning |
| `tests/test_ai_settings.py` | The config file |
| `tests/test_ai_embedder_math.py` | Frame size, patch pooling, padded crops |
| `tests/test_ai_api.py` | `AiEngine` with fake models, and `open_image` |
| `tests/test_ai_calibration.py`, `tests/test_ai_enrollment.py` | The calibration, the box scaling and the brightness |
| `tests/test_ai_matching.py` | The matching: threshold, margin, look-alikes, phrases, skipped regions, bad data |
| `tests/test_ai_scene.py` | The context box, the rooms and the surfaces in the engine (a fake model), errors, the switch-off |
| `tests/test_ai_scene_model.py` (slow) | The real scene model on real photos: probabilities, equality with the full model, rooms and surfaces |
| `tests/test_ai_verification.py` | The check in the engine: the first pass stops it, the best count, the limit, the crop at the edge, the switch-off |
| `tests/test_ai_verifier_model.py` (slow) | The real verifier on real photos: the bottle passes, other parts of the photos do not |
| `tests/test_ai_query_model.py` (slow) | The real query encoder: the vectors, the padding, four vague questions |
| `tests/test_ai_models.py` (slow) | The real models: vector shape and length, the three passes, the engine |
| `tests/test_ai_speech_model.py` (slow) | The real speech model |
| `tests/test_ai_open_detector.py`, `tests/test_ai_open_detector_model.py` (slow) | The broad detector: the settings, the weights check, the offline switches, the real model |
| `tests/test_ai_open_vectors.py`, `tests/test_ai_open_vectors_model.py` (slow) | The SigLIP vectors of crops and phrases |
| `tests/test_ai_language.py`, `tests/test_ai_language_model.py` (slow) | The query rewriter and the description, with a fake and with the real model |
| `tests/test_ingestion_chain.py` (slow) | One photo from the upload to the retention, with all modules |

Run the slow tests with `uv run pytest -m slow`.
They need a photo. Set `SAARTHI_TEST_IMAGE` or put `test_img.jpg` in the project root.
