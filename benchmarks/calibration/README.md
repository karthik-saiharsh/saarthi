# Calibration

These scripts measure the start values of the image pipeline on a set of real photos.
They are not part of the runtime. They found the values that `ai/config.toml` has now.

## Data

The set is the folder `Dataset` (14 photos of the target camera). One object is in it: a speckled bottle.
It is in 6 photos. In 8 photos it is not. The photos show a corridor, a balcony, a room with a window, a shelf and a desk.

`labels.json` holds the ground truth. A person looked at each photo with the boxes of the detector drawn on it,
and chose the box that frames the bottle. The labels are not independent of the detector: only boxes that the detector made can be chosen.
A photo with an empty list has no bottle.

## How to run

```
# 1. The detector output with low thresholds (about 1 minute for each photo, run it one time)
uv run python benchmarks/calibration/collect.py DATASET_DIR CACHE_DIR
# 2. The tests. They read the cache. Set PYTHONPATH=.:benchmarks/calibration
uv run python benchmarks/calibration/analyse_detection.py DATASET_DIR CACHE_DIR      # no model, a few seconds
uv run python benchmarks/calibration/analyse_matching.py DATASET_DIR CACHE_DIR       # embedder, about 1 minute
uv run python benchmarks/calibration/analyse_verification.py DATASET_DIR CACHE_DIR   # verifier, about 4 minutes
```

## Method

- **Detector.** The script replays the cached output with other settings. A box is a hit if it has an IoU of 0.5 or more with a labelled box.
  It is a false box if the IoU is below 0.1 with all labelled boxes.
- **Matching (leave-out test).** The script takes each subset of 3 or 4 of the 6 bottle photos. It enrolls the object from them.
  It calculates `tau` with the formula of the application. Then it tests the bottles of the other photos (recall)
  and all false candidates (false accepts). Recall with gates counts a bottle only if its box passes the quality gates.
- **Verification.** For each candidate, the script counts the matching keypoints with the reference photos.
  The references are the other bottle photos, never the photo of the candidate.

## Results and decisions

| Value | Before | Now | Evidence |
|-------|--------|-----|----------|
| `detection.box_threshold` | 0.25 | 0.25 (kept) | A lower value adds false boxes and no hit. A value above 0.33 loses the blurred bottle of photo 8, which only the generic pass finds, at the score 0.33 |
| `detection.tiling_enabled` | false | false (kept) | The tiled pass gave 84 false boxes instead of 35, and no extra hit |
| `detection.nms_iou` | 0.7 | **0.5** | Boxes per photo 3.3 to 2.6, false boxes 35 to 30, all 6 hits stay |
| `quality.blur_min` | 0 | 0 (kept) | The photos with a bottle have the scores 8, 63, 71, 89, 204 and 564. A limit of 100 would reject 4 of 6 |
| `quality.crop_blur_min` | 20 | 20 (kept) | A limit of 5 raises the recall with gates from 0.63 to 0.80. It also raises the false accepts from 2.2 to 3.5 and the false strong matches from 1.1 to 1.9. A blurred crop cannot be verified. A wrong place costs more than a missed blurred frame |
| `enrollment.tau_min` | none | **0.45** | The formula alone gives 0.18 to 0.54, depending on the 3 or 4 photos. A low value let 10.8 of 35 false candidates pass. A floor of 0.45 gives 2.2, with the same recall (0.94). A floor of 0.50 lowers the recall to 0.78 |
| `enrollment.tau_strong_offset` | 0.05 | **0.10** | With the floor, the false strong matches go from 1.1 to 0.0. The true strong matches change from 0.50 to 0.48 |
| `enrollment.tau_std_factor` | 0.5 | 0.5 (kept) | With the floor it matters little |
| `verification.min_matches` | 30 | 30 (kept) | The sharp true bottles have 81 to 264 matches (8 candidates). The false candidates have 1 to 17 (20 candidates). Any limit from 18 to 80 works |

The False strong match is the number that matters most. It changes the last-seen record at once.
With the old values, 7.6 false candidates for each subset reached the strong level. With the new values, none do.

## What the data could not decide

- `margin_min` and `margin_strong`: there is one object, so there is no other object to compare with.
- `generic_pass_every`: it needs a sequence of frames. One finding: a blurred close-up of the bottle is only found by the generic pass.
  The pass runs on every 10th frame, so such a bottle is missed in most frames.
- The hysteresis (`hysteresis_window_seconds`, `hysteresis_hits`) and `stale_after_hours`: they need photos in time order.
- The values of `[query]`: they need several objects and real questions of users.
- `text_threshold` (0.20): the cache was made at 0.15, so the labels of the boxes are those of 0.15.

## Limits

- **A small set.** There are 14 photos, one object and one camera. A set of this size can fit any setting. Read the numbers as a sign, not as a proof.
- **The test counts boxes, not photos.** A photo with many false boxes weighs more.
- **Photo 14 passes by a very small margin.** The small bottle at the back of a desk has the similarity 0.451.
  The threshold of that object is 0.45.
  The verifier then promoted the match (153 matching keypoints). A small bottle is the weakest case.
- **Blurred crops of the bottle** (photos 3 and 8) are lost by the blur gate. This is a decision for precision.
- **A new camera needs a new run.** The thresholds depend on the images and on the embedding model.
