# Open Search (broad detection)

## Status

This feature is **partly built**. Built (phase 1): the broad detector in the `ai` component (`ai/open_detector.py`, `AiEngine.detect_open`).
Not built: the storage of the boxes, the SigLIP vectors, the worker step, the search and the sentence.
The owner decided this on 2026-11-10: the main pipeline stays on the registered objects. The broad detection is for objects that nobody registered.

## Purpose

A user can ask for an object that the carer did not register, for example a laptop. The system must give a result.
The result is always weak: the sentence says "I am not sure". The picture stays on the page.

## Rules

- **Registered objects come first.** If the words of the user fit an enrolled object, the search uses the registered path only.
  The open search starts only when no enrolled object fits.
- **No list of classes.** The detector gets no class names. It names each box from its own list of about 4600 names.
- **The names are hints.** A name can be wrong ("passbook" for a notebook). The backend must check a name before the voice says it.
- **Privacy.** Pictures, boxes and words stay on the device. The package `ultralytics` is set to work offline (see below).

## The plan

| Phase | Content | Status |
|-------|---------|--------|
| 1 | The detector in `ai/`, the weights check, the settings, the tests | Built |
| 2 | SigLIP vectors for the crops, and a text vector for a phrase (`ai/`) | Planned |
| 3 | The tables for the boxes and a second vector index (`db/`) | Planned |
| 4 | The worker step: detect, link to registered boxes, embed, store | Planned |
| 5 | The search: the phrase of the user, the match with the stored names, then the SigLIP retrieval | Planned |
| 6 | The sentence and the page | Planned |
| 7 | The calibration on real frames | Planned |

The search of phase 5, in this order:
1. Make a phrase from the words ("where is my laptop" gives "laptop").
2. Compare the phrase with the stored names with the sentence encoder (MiniLM). A score above a cutoff gives a hit.
3. If there is no hit, compare the SigLIP text vector of the phrase with the stored crop vectors.
4. Show the best hit as a weak result.

## The detector (phase 1)

- Model: YOLOE, prompt-free (`yoloe-11s-seg-pf.pt`, 28 MB). The licence is AGPL-3.0 (Ultralytics). The owner accepted it for this academic project.
- `AiEngine.detect_open(image)` gives `OpenDetection` objects: a box, a name and a confidence. The most confident box comes first.
- The settings are in `ai/config.toml`: `[models]` (file, URL, SHA-256) and `[open_detection]` (`confidence`, `max_boxes`, `input_size`, `min_box_side`).
- The weights are downloaded one time into `ai/models/`, and the SHA-256 is checked each time. The folder is not in git.
- The package `ultralytics` is pinned in `pyproject.toml`. Its import is late, and the module sets these before it:
  `YOLO_OFFLINE` (no online check and no download), `YOLO_AUTOINSTALL=0` (no install at run time), and `sync=False` (no usage data).
- The import of `ultralytics` replaces `PIL.Image.open` for the whole process. The upload step needs the original function to refuse
  a file that is not an image. The module puts the original back. A test checks this.

### Test on 18 frames (4 of the owner, 14 of the dataset)

| Result | Value |
|--------|-------|
| Time | 0.25 to 0.57 s a frame on a CPU, median 0.45 s (Grounding DINO: about 10 s) |
| Boxes at confidence 0.10 or more | 9 to 39 a frame, median 15 |
| Laptop | Found in all 3 frames that show one (0.96, 0.97, 0.98) |
| Bottle in a hand | Found in 2 frames (0.82, 0.47 on the whole bottle). The registered path had missed one of them |
| Wrong names | Many: "studio shot", "passbook", "mistletoe", "bedpan". Some have a confidence of 0.5 to 0.7 |
| Misses | A small bottle on a desk (only "bottle cap", 0.26). A blurred bottle |

The names are noisy, so phase 5 needs the check with SigLIP. The input of 640 pixels loses small objects in a large frame.
The start value of `confidence` is 0.25. Phase 7 sets it with the new images of the owner.
