# AI

This is the ai module.
It looks at one image and gives the data that the database needs:
a quality check, the boxes of candidate objects, and one vector for each box.
It also gives the stages of enrollment: find one object in a photo, make its views, and calibrate its thresholds.

For the full design, see `ai/docs/AI Ingestion Module.md`.
The source of the design is `docs/ai/DINO-Based Object Recall Pipeline.md`.

## Files

| File | Purpose |
|------|---------|
| `api.py` | The public interface: `AiEngine` and `open_image` |
| `quality.py` | Blur score, perceptual hash, Hamming distance. No model |
| `geometry.py` | Box functions: IoU, overlap, clip, NMS, tile grid. No model |
| `postprocess.py` | Turns detector output into candidates. No model |
| `detector.py` | `CandidateDetector`: Grounding DINO with three kinds of pass |
| `embedder.py` | `RegionEmbedder`: DINOv2 small, one vector for each box |
| `calibration.py` | The thresholds of an object from its views (stage E4). No model |
| `enrollment.py` | Box scaling and brightness for the views of a photo. No model |
| `runtime.py` | Safe model loading, device choice, model names |
| `settings.py` | Reads `config.toml` |
| `types.py` | `Box`, `QualityReport`, `Candidate`, `RawDetection` |
| `errors.py` | The error classes |
| `config.toml` | Every setting: models, thresholds, phrases, switches |
| `docs/` | The design documents of this module |

## Rules

- This module does not control the flow of the application. The `backend` module does that.
- This module gives stage functions. The `backend` module calls them in the order that it chooses.
- The `backend` module uses `ai/api.py` only.
- This module does not read the database and does not read or write image files. It never imports `db`, `storage`, `backend` or `ui`.
- Load the models one time when the application starts. Never load a model for each request.

## Stages

| Stage | Helper | Result |
|-------|--------|--------|
| Decode | `open_image(data, max_pixels)` or `engine.open_image(data)` | An RGB image. The EXIF orientation is applied |
| I1 Quality | `engine.assess_quality(image)` | `QualityReport` (blur score and hash) |
| I1 Gate | `engine.is_blurry(report)`, `engine.is_near_duplicate(hash, recent_hashes)` | True or False |
| I2 Detect | `engine.detect_candidates(image, ...)` | A list of `Candidate`. The best score comes first |
| I3 Embed | `engine.embed_regions(image, boxes)` | A `float32` array of shape `(n, 384)`, each row of unit length |
| E2 Locate | `engine.locate_object(image, category_phrase, synonyms)` | A `LocatedObject` (the best box and its score), or `None` |
| E3 Views | `engine.embed_views(image, box)` | A list of (view name, vector). The views are in `[enrollment]` of `config.toml` |
| E4 Calibrate | `engine.calibrate_object(views, photo_ids, others)` | A `Calibration` with `tau`, `tau_strong` and a note. `engine.min_photos` is the number of photos that an object needs |

`engine.embedding_model` is the name that goes into `Region.embedding_model`.
`engine.embedding_dimension` must equal `vectors.dimension` in `db/config.toml`.

## Usage

```python
from ai.api import AiEngine

engine = AiEngine()  # Loads both models. Do this one time.
image = engine.open_image(data)
report = engine.assess_quality(image)
candidates = engine.detect_candidates(image)
vectors = engine.embed_regions(image, [c.box for c in candidates])
```

## Configuration

Edit `config.toml`. The file explains each key.
`load_config` checks every value and names the key of a wrong value. It raises `AiConfigError`.

## Models

| Task | Model | Size |
|------|-------|------|
| Detect candidates | `IDEA-Research/grounding-dino-tiny` | About 690 MB |
| Embed regions | `facebook/dinov2-small` | About 90 MB |

- Each model is fixed to one revision in `config.toml`. Only safetensors files are loaded.
  No code from the model repository is run.
- The first start downloads the models to the Hugging Face cache, `~/.cache/huggingface/hub/`.
  Set the variable `HF_HOME` to use another folder.
- After the first download, set `models.local_files_only = true` for a start without internet.

## Errors

| Error | Meaning |
|-------|---------|
| `AiError` | Base class of all errors below |
| `AiConfigError` | `config.toml` is missing or has a wrong value |
| `ModelLoadError` | A model cannot be loaded |
| `ImageError` | The file is not an image, is broken, or has too many pixels |
| `InferenceError` | A model failed while it worked on an image |
