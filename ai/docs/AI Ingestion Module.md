# AI Ingestion Module

## Purpose

The ai module turns one stored image into data for the database.
It does three things:

1. It checks the quality of the image (stage I1).
2. It finds boxes that can show an object that the user may lose (stage I2).
3. It makes one vector for each box (stage I3).

The design follows `docs/ai/DINO-Based Object Recall Pipeline.md`.
The ingestion pipeline covers the stages I1, I2 and I3.
The stages that need a gallery of enrolled objects are not in this module yet (see "Not in this module").

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
| `AiEngine(config=None)` | Load both models | The engine. Call it one time at start-up |
| `open_image(data, max_pixels)` | Decode image bytes. Apply the EXIF orientation. Convert to RGB | A PIL image. Raises `ImageError` |
| `engine.open_image(data)` | The same, with `image.max_pixels` of the config | A PIL image |
| `engine.assess_quality(image)` | Blur score and perceptual hash | `QualityReport` |
| `engine.is_blurry(report)` | The blur score is below `quality.blur_min` | `bool` |
| `engine.is_near_duplicate(phash, recent_phashes)` | A recent hash is `quality.phash_max_distance` bits away or closer | `bool` |
| `engine.detect_candidates(image, phrases=None, small_phrases=None, include_generic=False, tiling=None)` | Find candidate boxes | List of `Candidate`, best score first |
| `engine.embed_regions(image, boxes)` | One vector for each box | `float32` array, shape `(n, 384)`, unit length |
| `engine.embedding_model` | Name for the database, for example `dinov2-small@ed25f3a3` | `str` |
| `engine.embedding_dimension` | Length of one vector | `int` |

`detect_candidates` uses `detection.default_phrases` when you give no phrases.
`include_generic` adds the generic pass. `tiling` switches the tiled pass for this call. The value `None` uses the config.
The backend chooses both switches for each frame. The ai module does not count frames.

### Data objects

| Object | Fields |
|--------|--------|
| `Box` | `x0`, `y0`, `x1`, `y1` in pixels. `x1` and `y1` are exclusive |
| `QualityReport` | `blur_score`, `phash` (16 hex characters) |
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
6. **Drop depictions.** An object with at least `detection.depiction_inside` of its area inside a depiction box is dropped.
7. **Flag hands.** An object gets `in_hand` if a hand box covers more than `detection.hand_overlap` of its area.

Hand boxes and depiction boxes are never returned as candidates.

### Crop quality

`apply_crop_quality` sets `quality_ok`. A candidate is good if all three are true:

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
  The crop has `embedding.crop_padding` of padding on each side. It is scaled to `embedding.crop_size`.
  The module uses the same mean-of-patches pooling, so all vectors stay in one space.
  Crops go through the model in batches of 16.
- Each vector is scaled to unit length. The inner product of two vectors is then their cosine similarity.

The vectors are not compared with a model change. A threshold is not portable between two models.
The name in `Region.embedding_model` tells which model made a vector.

## Configuration

All settings are in `ai/config.toml`. The start values come from the pipeline document.

| Table | Keys |
|-------|------|
| `[models]` | `detector_id`, `detector_revision`, `embedder_id`, `embedder_revision`, `device` (`auto`, `cpu` or `cuda`), `cpu_threads`, `local_files_only` |
| `[image]` | `max_pixels` |
| `[quality]` | `blur_min`, `analysis_max_side`, `phash_max_distance`, `min_crop_side`, `crop_blur_min` |
| `[detection]` | `box_threshold`, `text_threshold`, `input_short_edge`, `max_per_phrase`, `chunk_size`, `nms_iou`, `max_frame_fraction`, `min_box_side`, `hand_overlap`, `depiction_inside`, `default_phrases`, `small_phrases`, `hand_phrase`, `depiction_phrases`, `generic_phrase`, `tiling_enabled`, `tile_overlap` |
| `[embedding]` | `frame_short_edge`, `frame_long_edge_max`, `crop_size`, `small_box_min_side`, `crop_padding` |

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

- The first start downloads the models to the Hugging Face cache, `~/.cache/huggingface/hub/`.
  Set the variable `HF_HOME` (or `HF_HUB_CACHE`) to use another folder.
- Each model has a fixed revision in `config.toml`. Change it only after a test run.
- The module loads only safetensors files. It never runs code from the model repository.
- After the first download, set `models.local_files_only = true`. The module then never uses the internet.
- The cache is per user and per machine. For the Raspberry Pi, run the application one time with internet, or copy the `hub/` folder.
- The module loads the models one time. Loading both takes about 6 seconds on the development machine.
- One lock protects each model, so two threads cannot use the same model at the same time.

## Errors

| Error | Meaning |
|-------|---------|
| `AiError` | Base class of all errors below |
| `AiConfigError` | `config.toml` is missing or has a wrong value. The message names the key |
| `ModelLoadError` | A model cannot be loaded |
| `ImageError` | The file is not an image, is broken or truncated, or has too many pixels |
| `InferenceError` | A model failed while it worked on an image |

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

- **Blur threshold.** The start value `blur_min = 100` calls the test photo blurry. The photo scores 94.
  The score also depends on the JPEG compression: the same photo saved again at another quality scored 117.1.
  Calibrate the value with 20 or more real frames from the wearable, both good and bad.
  The value `blur_min = 0` turns the gate off.
- **Vocabulary.** The phrases are a placeholder. The labels are only as good as the phrases.
- **Duplicate boxes.** A lower `nms_iou`, or one box for each object, needs a decision after the vocabulary is real.
- **Other start values.** `crop_blur_min`, `box_threshold` and `text_threshold` are values from the pipeline document. They are not tuned.
- **Speed on the Raspberry Pi.** It is not measured. The installation of the aarch64 wheels of torch is not tested.
- **Not tested.** `device = "cuda"`, `local_files_only = true`, `cpu_threads`, and two threads that call a model at the same time.
- **Stored image.** The storage module saves the image again as a JPEG. On the test photo, the number of candidates changed from 13 to 14.
  12 of the 13 boxes stayed the same (IoU of 0.8 or more). The blur score did not change.
- **Orientation.** `open_image` applies the EXIF orientation. The storage module applies it before it removes the metadata,
  so the stored file has the same pixel grid that the boxes in the database use.

## Not in this module

These parts need a gallery of enrolled objects in the database. They are not built yet:

- Enrollment of objects (stages E1 to E4).
- Matching of a box with an enrolled object (I4), verification (I5) and the last-seen record (I6).
- The search for an object and the answer to the user.
- Techniques from the pipeline document: pooling over a mask, background-free enrollment, and the calibrated accept and reject model.

## Tests

| File | What it tests |
|------|---------------|
| `tests/test_ai_quality.py` | Blur score and perceptual hash |
| `tests/test_ai_geometry.py` | IoU, overlap, clip, NMS, tile grid |
| `tests/test_ai_postprocess.py` | Label matching, every post-processing rule, crop quality, prompt planning |
| `tests/test_ai_settings.py` | The config file |
| `tests/test_ai_embedder_math.py` | Frame size, patch pooling, padded crops |
| `tests/test_ai_api.py` | `AiEngine` with fake models, and `open_image` |
| `tests/test_ai_models.py` (slow) | The real models: vector shape and length, the three passes, the engine |
| `tests/test_ingestion_chain.py` (slow) | One photo from the upload to the retention, with all modules |

Run the slow tests with `uv run pytest -m slow`.
They need a photo. Set `SAARTHI_TEST_IMAGE` or put `test_img.jpg` in the project root.
