# Recall benchmark

## Purpose

These scripts measure how well the pipeline finds the registered objects. They are not part of the runtime.
They are phase R0 of the plan for the recall of registered objects. They do two things:

1. They replay **today's pipeline** on a set of frames and show the step where each true object is lost.
2. They measure the **signals of the planned matching** (two keys and a word vote) on the same boxes, and replay its rule.

The model work runs one time (`collect.py`). It writes a signal store. The report (`report.py`) reads the store and runs no model.
So a change of a bar or a rule is tested in seconds.

## Data

The scripts need a **set file** (JSON). It names frames of a home, so it is private. Do not put it in the repository.

```json
{
  "objects": {
    "Watch": {
      "phrases": ["wrist watch"],
      "references": [{"image": "storage/uploads/2026/11/10/a.jpg", "box": [10, 20, 200, 220]}]
    }
  },
  "frames": {"F2": "storage/uploads/2026/11/10/b.jpg"},
  "cohort_dir": "/path/to/frames/with/no/registered/object",
  "truth": {
    "Watch": {"true": [["F2", [899, 1497, 1100, 1773]]], "hard": []}
  }
}
```

| Key | Content |
|-----|---------|
| `objects` | Each registered object: the detector phrases (the category and the synonyms), and 2 or more reference photos with the box of the object |
| `frames` | The frames of the home. A key is a short name. A path is absolute or starts at the project folder |
| `cohort_dir` | A folder of frames that show **no** registered object. Each of their boxes is a negative. The report sets bars from them. Optional |
| `truth` | For each object, the boxes of the frames that show it. `true`: a person is sure. `hard`: the identity or the view is not sure. Hard boxes are reported apart and set no bar |

A box in a frame that the truth does not name is a negative. A box that covers a half or more of a truth box is a part of the object. It is not a negative.

## How to run

Run it on a **copy** of the data or read-only. The scripts open the frames for reading only and write only to `OUT_DIR`.

```
# 1. The models, one time (about 20 s for each frame on the development CPU)
uv run python benchmarks/recall/collect.py SET_JSON OUT_DIR
#    A quick check with the first 3 frames
uv run python benchmarks/recall/collect.py SET_JSON OUT_DIR --limit 3
# 2. Optional: the negatives of the reference photos (a few minutes). They go into the same OUT_DIR
uv run python benchmarks/recall/negatives.py SET_JSON OUT_DIR
# 3. The report (a few seconds). The options set the rule of the replay (section 5)
uv run python benchmarks/recall/report.py SET_JSON OUT_DIR
uv run python benchmarks/recall/report.py SET_JSON OUT_DIR --boxes gdino --dino pool_mask --sig-strong 0.78
uv run python benchmarks/recall/report.py SET_JSON OUT_DIR --dino-weak 0.40 --negatives both
```

| Option | Effect |
|--------|--------|
| `--boxes` | `union`: the boxes of both detectors. `gdino`: only the candidates of today |
| `--dino` | The DINOv2 key: `cls` or `pool_mask` |
| `--sig-strong`, `--dino-strong`, `--sig-weak`, `--word-vote` | The bars of the rule |
| `--dino-weak` | One fixed DINOv2 bar for the weak rule, for all objects. Without it, each object takes the 1 % bar of the cohort |
| `--negatives` | `sig`, `dino` or `both`: refuse a box that is closer to a negative of the reference photos than to the object, with this key. Needs `negatives.py` |
| `--neg-margin` | The box must be closer to the object than to the nearest negative by this value |

The report also works on a partial store. A frame that is not in the store shows `notrun`.

## Method

### Today's pipeline (`collect.py`)

- The script enrolls all objects of the set together, with the calls of the worker: `AiEngine.embed_views` for each reference photo,
  then `AiEngine.calibrate_object` against the views of all other objects.
- For each frame it calls `AiEngine.detect_candidates` with the phrases of all objects, `AiEngine.embed_regions`,
  then the mode `evidence` (`evidence_regions`, `embed_evidence_crops`, `score_evidence`) and the mode `tau` (`match_regions`).
- Not replayed: the generic pass (each 10th frame), the keypoint check (stage I5), the memory rule (stage I6) and the search.
  The keypoint check can only promote a weak sighting, so it does not change the recall.

### The new signals (`collect.py`)

The boxes are the union of the candidates of today and the YOLOE boxes (prompt-free, the settings of `[open_detection]`).
A candidate takes the mask of the YOLOE box that fits it (IoU of 0.5 or more). For each box and each reference photo:

| Signal | Content |
|--------|---------|
| `pooled` | Today's region vector: DINOv2 patches of the whole frame, pooled in the box |
| `sig` | The SigLIP image vector of the crop with `matching.crop_padding` |
| `cls` | The DINOv2 CLS token of the tight crop at 224 px |
| `pool_mask` | The DINOv2 patches of the tight crop, pooled inside the mask (all patches if no mask) |
| `pool_attn` | The DINOv2 patches of the tight crop, pooled with the attention of the CLS token (a simple form of Object-DINO) |
| `word` | The highest detector score of a candidate with a phrase of the object on this box (IoU of 0.5 or more) |

### The negatives of the reference photos (`negatives.py`)

A reference photo also shows the place of the object. Each other box of the photo is a negative from the home.
The script uses the same two detectors. It leaves out a box that covers a half or more of the object box.
A negative belongs only to the object of its photo.

The agreement of a box with an object is the mean of its 2 best reference photos.

### The report (`report.py`)

| Section | Content |
|---------|---------|
| 1. Boxes | For each truth instance: did today's detector (G) or YOLOE (Y) give a box with an IoU of 0.5 or more |
| 2. Today | For each truth instance and each mode: `STRONG`, `weak`, or the step that lost it: `nobox`, `quality` (the crop gate), `notchosen` (`candidate_min` or `max_regions`), `below` (scored, no sighting). Also the false sightings |
| 3. Signals | For each signal: the bar is the lowest value of the true instances. The table counts the negatives at or above it, in the home frames and in the cohort |
| 4. Cohort | For each object: the value that only 1 % or 0.1 % of the cohort boxes reach. It needs no truth |
| 5. Replay | The rule of the plan at frame level. Strong: SigLIP and DINOv2 at their strong bars. Weak: SigLIP at its weak bar, and DINOv2 above the 1 % cohort bar or a word vote. A crop that fails the quality gate is weak at most. One box is one object. One sighting for each object in a frame |

A sighting counts as found if its box is the truth box or a part of it. The place is the same for the patient.

## Results (2026-11-12)

The set: 4 objects enrolled together (Watch, Book, Bottle, Headset; 2 to 4 reference photos each),
31 home frames with 22 true and 8 hard instances, and a cohort of 309 frames of other homes (4619 boxes).
The collection took 93 minutes on the development CPU (340 frames, 5002 boxes).

### Today's pipeline against the two-key rule

The bars of the replay are the start bars of the plan: SigLIP 0.77 and CLS 0.42 for strong, SigLIP 0.72 for weak.

| Pipeline | True found (22) | Strong | Weak | False sightings, home frames | False sightings, cohort frames |
|----------|-----------------|--------|------|------------------------------|--------------------------------|
| Today, mode `tau` | 8 | 5 | 3 | 0 | 6 |
| Today, mode `evidence` (default) | 13 | 8 | 5 | 0 | 3 |
| Two keys, union boxes | 20 | 14 | 6 | 0 | 15 (1 strong, 14 weak) |
| Two keys, Grounding DINO boxes only | 20 | 12 | 8 | 0 | 13 |
| Two keys, weak bar 0.74 | 19 | 14 | 5 | 0 | 8 (1 strong, 7 weak) |

- Today, 8 of the 9 lost true instances are `below`: the detector gave a box and the box was scored, but no sighting came out.
  The pooled DINOv2 vector (`pooled`) cannot separate them. At full recall it accepts 473 home boxes and 5555 cohort boxes.
- The two-key rule finds all 5 watches and all 3 books. It loses 2 of 4 headsets. The reference photos of the headset do not look like the frames.
- 12 of the 15 false cohort sightings of the two-key rule are weak sightings of the watch and the headset.
  A weak sighting changes the last-seen record only after a second one within 60 s. The replay does not count this rule.
- The false sightings in the cohort show that the bars must come from a cohort, as the plan says. They are not final.

### Without a cohort, and with the negatives of the reference photos (phase R1a)

The owner decided to build the new matching without a cohort first. So the weak rule takes one fixed DINOv2 bar for all objects.
The 12 reference photos gave 33 negatives (0 to 6 for each photo: the photos are close views).

| Rule (start bars, CLS key) | True found (22) | False, home / cohort |
|----------------------------|-----------------|----------------------|
| Cohort bar for each object (R0) | 14 + 6 = 20 | 0 / 15 |
| Fixed DINOv2 weak bar, any value from 0.36 to 0.42 | 14 + 6 = 20 | 0 / 15 |
| Fixed bar 0.40, negatives on both keys | 14 + 6 = 20 | 0 / 14 |
| `--dino pool_mask` in place of `cls` | 14 + 6 = 20 | 0 / 15 |

A lower SigLIP weak bar, fixed DINOv2 bar 0.40:

| SigLIP weak bar | True found, no negatives | False, no negatives | True found, negatives | False, negatives |
|-----------------|--------------------------|---------------------|-----------------------|------------------|
| 0.72 | 20 | 0 / 15 | 20 | 0 / 14 |
| 0.68 | 20 | 2 / 37 | 20 | 2 / 31 |
| 0.65 | 20 | 4 / 68 | 20 | 2 / 41 |
| 0.62 | 21 | 8 / 151 | 20 | 3 / 62 |

- The fixed bar loses nothing against the cohort bar on this set.
- The CLS token and masked pooling give the same result. The CLS token needs no mask, so it is the simpler key.
- At the start bars the negatives remove 1 false sighting. At lower bars they remove about a half of the false sightings.
  A lower bar finds no more true objects (except 1 headset at 0.62), so the start bars stay.
- The new rule keeps every true instance that the mode `evidence` finds (the exit test of phase R1), and it gives no false strong sighting in the home frames.

### Check against Revision 3 of the plan (the exit test of phase R0)

The plan scripts found 14 strong, 6 weak and 2 lost true instances with the same bars. The replay gives the same numbers.
The signal table (section 3) has the same order: `pool_mask` (8 / 251) is the best single DINOv2 signal, then `cls` (16 / 429), then `pool_attn` (21 / 549).
The numbers are not equal, for 2 reasons:

- `collect.py` takes the boxes from `AiEngine.detect_candidates` with the phrases of all 4 objects. The plan scripts used one phrase for each run.
  So the boxes are not the same. Example: the Grounding DINO box of the watch in frame D14 has an IoU below 0.5 with the truth box (21 of 22 in section 1).
  It is a part of the watch, so the sighting counts.
- The headset counts differ the most (SigLIP 56 / 1242 here, 103 / 2068 in the plan), because its lowest true value (0.56) sets the bar in a crowded range.

## Limits

- The set is small. Read the numbers as a direction.
- The truth boxes of the stored frames come partly from Grounding DINO. An object that it missed and that nobody saw is not in the truth.
- The cohort comes from other homes and cameras. The plan also adds the boxes of the home to it later.
- The replay is per frame. The memory rule over time is not replayed. It needs a recorded sequence.
