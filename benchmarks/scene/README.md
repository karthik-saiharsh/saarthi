# Scene description test

This test compares ways to describe the place of an object in a frame (the planned stage Q2, see `docs/backend/Voice.md`).
The voice must say where the object was. A wrong description can mislead a patient, so the test looks at the errors first.
The scripts are not part of the runtime.

## Data

The 14 photos of the folder `Dataset`. In 6 photos, the speckled bottle is labelled (`../calibration/labels.json`).
`labels.json` of this folder holds the rooms that count as right for each photo.
These judgements are subjective. A balcony and a corridor look alike, so a photo can have more than one right room.

## How to run

```
PYTHONPATH=.:benchmarks/calibration uv run python benchmarks/scene/compare.py DATASET_DIR OUT_DIR --method lists
PYTHONPATH=.:benchmarks/calibration uv run python benchmarks/scene/compare.py DATASET_DIR OUT_DIR --method detector
PYTHONPATH=.:benchmarks/calibration uv run python benchmarks/scene/compare.py DATASET_DIR OUT_DIR --method vlm256 --style full
uv run python benchmarks/scene/analyse.py OUT_DIR
```

The methods need their models in the Hugging Face cache (see the revisions in `compare.py`).

## The methods and the results

| Method | Model | Result |
|--------|-------|--------|
| Free text | SmolVLM-256M (513 MB) | With a short prompt, the answers have one word ("Floor.", "White."). With a full prompt, they are sentences, but a part is invented (see below). 2 to 5 seconds for each photo |
| Fixed lists | SigLIP base (813 MB) | The room is right for 11 of 14 photos. The surface is mostly right. About 2 seconds for each photo |
| Fixed lists and the neighbours in time | The same | The room is right for 13 of 14 photos (an illustration, see "Limits") |
| Colour from the pixels | None | Clearly right for 2 of 6 photos |
| Furniture from the detector | Grounding DINO tiny (already in the application) | Not usable (see below) |

**Free text.** Some answers are right: a backpack on a chair, a bottle on a window sill, a laptop on a table.
Some parts are invented: a "fireplace" on a balcony, "high-rise buildings" in a corridor, and a "painting" for a notebook on a railing.
A window sill with a soap dispenser became a "table with two bottles".
The invented parts have the same confident tone as the right parts. The model of 500 MB (SmolVLM-500M, 1015 MB) was not tested.

**Fixed lists.** A wrong word comes from a closed list. The errors are three photos of a marble floor or a window sill that the model calls "a bathroom",
with the scores 0.71, 0.93 and 0.83. The right rooms have scores from 0.47 to 1.00.
So a score cutoff cannot remove the wrong rooms. The reason is in the photo: a first-person photo of a floor does not show the room.

**The neighbours in time.** The person stays in one place for some minutes. The average of the room scores of the photo and its neighbours fixes the floor photos.
A window of 1 neighbour on each side gives 13 of 14. A window of 2 gives 11 of 14, because it mixes two scenes.

**The colour.** The colour of the surface came from the pixels below and beside the object. It was right for a grey table and a beige marble floor.
It was borderline twice (a white ledge became "beige" in warm light). It was wrong twice (a marble floor became "yellow", and a grey desk became "brown" because of a red book).

**The detector.** The detector looks for furniture, and a room follows from it (a bed means a bedroom). The detector finds furniture that is not there.
At the score 0.30, it claims a room for 9 photos. Eight claims are wrong.
Examples: a toilet and a bathtub on a balcony, and a television in a corridor.
At the score 0.40 or higher, it makes no claim, because even real furniture scores less. It did not find the desk and the laptop in photo 14.

## Limits

- **A small set.** 14 photos, one camera, and my own labels. Read the numbers as a sign.
- **The time order is an assumption.** The test takes the photo numbers as the order in time. They may not be.
- **No model on the Raspberry Pi.** The times are from the development machine (a CPU, no GPU). The speed on the Raspberry Pi is not measured.
- **The model size.** SigLIP base is 813 MB. The application already loads four models.
- **SmolVLM-500M and other models** were not tested.
