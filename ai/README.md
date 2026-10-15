# AI

This is the ai module.
It looks at one image and gives the data that the database needs:
a quality check, the boxes of candidate objects, and one vector for each box.
It also gives the stages of enrollment: find one object in a photo, make its views, and calibrate its thresholds.
It matches the boxes of a frame with the enrolled objects.
It turns the words of a user into a vector, to find the object that the user means.

It also scores the room and the surface of a place, turns speech into text, finds objects that nobody registered, and runs a small language model.

For the full design, see `docs/ai/AI Ingestion Module.md`.
The source of the design is `docs/guide/DINO-Based Object Recall Pipeline.md`.

## Files

| File | Purpose |
|------|---------|
| `api.py` | The public interface: `AiEngine` and `open_image` |
| `quality.py` | Blur score, perceptual hash, Hamming distance. No model |
| `geometry.py` | Box functions: IoU, overlap, clip, NMS, tile grid. No model |
| `postprocess.py` | Turns detector output into candidates. No model |
| `detector.py` | `CandidateDetector`: Grounding DINO with three kinds of pass |
| `embedder.py` | `RegionEmbedder`: DINOv2 small, one vector for each box |
| `scene.py` | `SceneClassifier`: SigLIP. It chooses the best room and the best surface from fixed lists of words (stage Q2). It also makes the crop vectors and the phrase vectors of the open search |
| `language.py` | `LanguageModel`: Qwen3.5-0.8B. It rewrites a description of an object into common object names for the open search, and it writes one sentence about the place of an object from a picture and checked facts |
| `open_detector.py` | `OpenDetector`: YOLOE without a list of classes. It finds and names objects that nobody registered (broad detection) |
| `speech.py` | `SpeechRecognizer`: Whisper base. It turns speech into English text (stage Q0). The audio stays in memory |
| `verifier.py` | `KeypointVerifier`: LightGlue with SuperPoint. It counts the keypoints that match between two pictures (stage I5) |
| `query.py` | `QueryEncoder`: a small sentence encoder. It turns the words of a user into vectors, to find the object that the user means |
| `matching.py` | Match region vectors with the enrolled objects (stage I4, mode `tau`: one threshold). No model |
| `keys.py` | Match boxes with the enrolled objects (stage I4, mode `keys`): a SigLIP key and a DINOv2 key (the CLS token of the crop) with fixed bars, and a word vote. No model |
| `evidence.py` | Match regions with the enrolled objects (stage I4, mode `evidence`): DINOv2 agreement, margin, SigLIP agreement with the reference crops, and the SigLIP meaning give one score. No model |
| `calibration.py` | The thresholds of an object from its views (stage E4). No model |
| `enrollment.py` | Box scaling and brightness for the views of a photo. No model |
| `runtime.py` | Safe model loading, device choice, model names |
| `settings.py` | Reads `config.toml` |
| `types.py` | `Box`, `QualityReport`, `Candidate`, `RawDetection` |
| `errors.py` | The error classes |
| `config.toml` | Every setting: models, thresholds, phrases, switches |

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
| Q Detect phrases | `engine.detect_phrases(image, phrases)` | A list of `RawDetection` (box, score, label), with no merge and no filter. The open search in the mode `verified` uses it |
| E3 Views | `engine.embed_views(image, box)` | A list of (view name, vector). The views are in `[enrollment]` of `config.toml` |
| I4 Match, mode `keys` | `engine.broad_candidates(image, detections)`, `engine.embed_evidence_crops(image, boxes)`, `engine.embed_key_crops(image, boxes)`, `engine.match_keys(...)` | A list of `Sighting`. Blurred crops are weak at most |
| I4 Match, mode `evidence` | `engine.evidence_regions(...)`, `engine.embed_evidence_crops(image, boxes)`, `engine.score_evidence(...)` | A list of `Sighting` with a `score`. `engine.matching_mode` says which mode is on (`[matching]` of `config.toml`) |
| E4 Calibrate | `engine.calibrate_object(views, photo_ids, others)` | A `Calibration` with `tau`, `tau_strong` and a note. `engine.min_photos` is the number of photos that an object needs |

`engine.embedding_model` is the name that goes into `Region.embedding_model`.
`engine.embedding_dimension` must equal `vectors.dimension` in `db/config.toml`.

## Usage

```python
from ai.api import AiEngine

engine = AiEngine()  # Loads the models (up to eight). Do this one time.
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
| Understand the words of a user | `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) | About 91 MB |
| Check a borderline sighting (I5) | `ETH-CVG/lightglue_superpoint` (**non-commercial research use only**) | About 55 MB |
| Describe the place (Q2), vectors of the open search | `google/siglip-base-patch16-224` (Apache-2.0) | About 813 MB |
| Turn speech into text (Q0) | `openai/whisper-base` (Apache-2.0) | About 290 MB |
| Find objects that nobody registered | YOLOE `yoloe-11s-seg-pf.pt` (**AGPL-3.0**, accepted for this academic project). Downloaded into `ai/models/`, SHA-256 checked | About 28 MB |
| Rewrite a description, describe a place | `Qwen/Qwen3.5-0.8B` (Apache-2.0) | About 1.75 GB |

- The weights of the verifier are for academic or non-profit non-commercial research use only.
  The owner confirmed that this project is for academic research. Change the verifier if the use changes.
  Set `verification.enabled = false` to not load it.
- Each Hugging Face model is fixed to one revision in `config.toml`. Only safetensors files are loaded. The YOLOE file is checked with its SHA-256.
  No code from the model repository is run.
- The first start downloads the models to the Hugging Face cache, `~/.cache/huggingface/hub/`.
  On a slow network the download can take about 15 minutes. If the download does not start (the partial file stays at 0 bytes),
  set `HF_HUB_DISABLE_XET=1` and start again. This helped on the development machine.
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
