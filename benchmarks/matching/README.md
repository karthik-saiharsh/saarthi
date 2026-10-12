# Matching benchmark

The script `evidence.py` measures the signals of the evidence-based matching (`ai/evidence.py`) on stored data.
It is not part of the runtime. It found the start values of `[matching]` in `ai/config.toml`.

## How to run

Run it on a **copy** of the data, never on the data of a user that the application uses.

```
# 1. The signals of every region against every object (SigLIP and LightGlue, about 2 minutes for 15 frames)
uv run python benchmarks/matching/evidence.py collect OUT_DIR --keypoints
# 2. A table for each object. TRUTH_JSON names the frames that show each object: {"Watch": [1, 2, 4, 15]}
uv run python benchmarks/matching/evidence.py report OUT_DIR TRUTH_JSON
```

The script reads `db/data` and `storage/uploads` of the project folder that contains it. It opens the database read-only.

## Data

| Set | Content | Truth |
|-----|---------|-------|
| Own frames | 15 frames of a room. Registered: a watch, a headset, a book. All 11 reference photos were taken on the same grey table | Watch: frames 1, 2, 4, 15 (on the wrist). Headset: 5, 9 (on a table), 10 (in a hand). Book (a red book): 7, 10. Frames 1 and 6 show **another** book |
| Hold-out | The 14 photos of the speckled bottle (`benchmarks/calibration`), enrolled from photos 2, 3 and 9. Not used to set the values | Bottle in photos 8 (blurred, not a good crop), 10 (in a hand), 14 (small, far) |

A person looked at each frame to set the truth.

## Method

- A frame counts for an object if its best region for that object is accepted.
- The signals are those of `ai/evidence.py`: `dino_agree`, `margin`, `visual` (SigLIP), `meaning` (SigLIP). The script also counts keypoints.
- Several weightings were compared on the own frames. The chosen one was then tested once on the hold-out set, without a change.

## Results

Scores of the chosen formula (the best region of each frame):

| Object | True frames | Best false frame |
|--------|-------------|------------------|
| Watch | 4.1, 4.4, 6.1, 6.4 | -4.2 |
| Book | 4.2 (frame 10), 6.9 (frame 7) | 2.6 (another book, frame 1) |
| Headset | 0.5, 1.9 (frame 10: no candidate) | 0.3 |
| Bottle (hold-out) | 5.9 (photo 14), 8.1 (photo 10) | 1.8 |

With `score_maybe = 3` and `score_sure = 6`:

| Object | Mode `tau` | Mode `evidence` |
|--------|------------|-----------------|
| Watch | 0 of 4 | 4 of 4 |
| Book | 1 of 2 | 2 of 2 |
| Headset | 0 of 3 | 0 of 3 |
| Bottle (hold-out) | 2 of 2 | 2 of 2 (photo 14 is weak and passes the keypoint check with 153 matches) |
| False sightings | 0 | 0 |

## Findings

- **DINOv2 alone failed on the watch and the headset.** Their crops were mostly table, so DINOv2 found the two objects similar (0.64),
  and the calibration raised both thresholds to 0.674. The true watch frames scored 0.26 to 0.52.
- **SigLIP separated them.** The SigLIP agreement with the reference crops and the meaning ("a wrist watch" against "a hand", "a table" and the other objects)
  carried most of the decision.
- **The meaning knows the kind, not the instance.** The other book scored 0.98 for "book". So the positive part of the meaning is capped at +1.
  Without the cap, the other book scored almost as high as the true book of frame 10 (5.0 against 5.7), and with equal weights it scored higher.
- **One lucky view.** 14 of the 15 regions where the watch scored 0.45 or more matched only one of its 3 photos. The agreement of 2 photos removes them.
- **Keypoints help only textured objects.** The book had 383 to 389 matches. The watch had at most 22, and the headset at most 14. So keypoints are not in the score.
  They stay as the check of stage I5, which can only promote a weak sighting.
- **The detector phrase is not a safe gate.** The watch on the wrist in frame 1 got the label "a headset".
- **The headset was not found.** Its reference photos are close-ups on one table. The frames show it from the side or in a hand. Photos in the places where it is used are needed.

## Limits

- The data is small: 4 objects, 26 frames, one camera. All values are start values.
- The truth is per frame, not per box.
- The keypoint counts use the crops of the benchmark script, not the crops of stage I5.
