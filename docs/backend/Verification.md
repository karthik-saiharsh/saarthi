# Verification (I5)

## Purpose

A weak sighting is a match that clears the threshold of the object but is not strong.
It can be right, but it can also be a wrong thing that looks like the object.
The verification checks such a sighting against the reference photos of the object.
It is stage I5 of `docs/guide/DINO-Based Object Recall Pipeline.md`.

The check counts the keypoints that match between two pictures.
Many matches mean that both pictures show the same textured thing. A few matches mean that they do not.
A weak sighting that passes the check counts as a **strong** sighting.

## Rules of the check

- The check **only promotes**. It never rejects a sighting. A sighting that fails stays weak, as it was without the check.
  This is the decision of the owner. A plain object (for example keys) has few keypoints, so a rejection would hide true sightings.
- It runs on **weak** sightings only. A strong sighting needs no check.
- An error of the check changes nothing. If the verifier fails, or a reference file cannot be read, the sighting stays as it is.
  The image does not fail, and the search does not fail.
- A switched-off check does nothing. With `verification.enabled = false`, the verifier is not loaded.

## Where it fits

```
worker: frame ──▶ MemoryService.remember ──▶ weak sighting ──▶ ReferenceVerifier ──▶ strong, or still weak
search: archive hit ──▶ SearchService._checked ──▶ weak hit ──▶ ReferenceVerifier ──▶ strong, or still weak
                                                                    │
                                              ai: AiEngine.verify_match ──▶ KeypointVerifier (LightGlue)
```

- `backend/verification.py` has `ReferenceVerifier`. The memory and the search use it. They share one code for the check.
- The worker gives the decoded frame to the memory, so the memory can cut the candidate from it.
- The search reads the frame of the hit from the storage.

## The steps of one check

1. **Choose the references.** The object has confirmed reference photos. Each photo has views (vectors).
   A photo is as similar to the candidate as its most similar view. The photos are sorted by this value, the most similar first.
   At most `verify_max_references` photos are used (3 by default).
2. **Cut the pictures.** The module cuts the box of the object from each reference photo and from the frame.
   Each cut has `verification.crop_padding` of space on each side (0.15). `AiEngine.verification_crop` does this for both, so the cuts are alike.
   A cut stops at the edge of the picture.
3. **Compare.** The verifier compares the candidate with the references, one by one.
   The check stops at the first reference with at least `verification.min_matches` matching keypoints (30 by default).
4. **Result.** `VerifyResult` has `verified`, `matches` (the highest count) and `compared`.
   Without a readable reference, the result is `verified = False`.

A photo that cannot be read is left out. If all photos are left out, the sighting stays weak.

## The verifier

`ai/verifier.py`, `KeypointVerifier`. The model is LightGlue with SuperPoint (`ETH-CVG/lightglue_superpoint`, about 55 MB).
SuperPoint finds keypoints in both pictures. LightGlue matches them. The module counts the matches with a confidence of at least `verification.match_threshold` (0.2).

- The model is loaded one time, with a fixed revision, from safetensors files only. No code of the model repository runs.
- One lock protects the model. The processor resizes each picture to 640 x 480 pixels.
- The weights are in the Hugging Face cache after the first download (about 55 MB).

### Licence

The weights of this model are for **academic or non-profit non-commercial research use only**.
The owner confirmed that this project is for academic research.
If the use changes, for example to a product that is sold, the verifier must change too.
A licence-clean choice is SIFT features with a RANSAC filter (OpenCV). The check has one function (`count_matches`), so a new verifier is a small change.

## Why 30 matches

The start value in the pipeline document is 15 matches that survive a RANSAC filter. This module has no RANSAC filter.
The value was checked with the photos of a speckled bottle (see `benchmarks/calibration/README.md`).
For each candidate, the best count over the 3 most similar reference photos counts. The reference photos are other photos than the candidate.

| Candidates | Matches |
|------------|---------|
| The bottle in a sharp crop (8 candidates in 4 photos, one of them a small bottle on a desk) | 81 to 264 |
| False candidates that look like the object (20 candidates with a similarity of 0.35 or more) | 1 to 17 |
| The bottle in a blurred crop (photos 3 and 8) | 4 to 11 |

Any limit from 18 to 80 separates the sharp bottles from the false candidates. The value 30 is inside this gap.
At 15, a false candidate (17 matches) would pass. A blurred crop cannot be verified: blur removes the keypoints.
The value is a start value. A set with more objects, and with look-alikes, can move it.

## Configuration

`ai/config.toml`:

| Key | Default | Meaning |
|-----|---------|---------|
| `models.verifier_id`, `models.verifier_revision` | `ETH-CVG/lightglue_superpoint`, a fixed revision | The model |
| `verification.enabled` | true | false: the model is not loaded, and no sighting is checked |
| `verification.min_matches` | 30 | A reference passes with this number of matches or more |
| `verification.match_threshold` | 0.2 | A match counts if its confidence is at least this value |
| `verification.crop_padding` | 0.15 | Space around the box in each cut, as a part of the box |

`backend/config.toml`:

| Key | Default | Meaning |
|-----|---------|---------|
| `memory.verify_weak` | true | The memory checks a weak sighting |
| `memory.verify_max_references` | 3 | The most reference photos for one check in the memory |
| `search.verify_weak` | true | The search checks a weak hit of the archive |
| `search.verify_max_references` | 3 | The most reference photos for one check in the search |

## What changes in the data

Nothing in the tables. A sighting that passes the check is stored with `strong = true`, and the record `last_seen` gets `strong = true`.
In the data, "strong" means "strong, or verified". The search shows such an answer without the word "maybe".

## Security and privacy

- The pictures stay on the device. The module sends nothing to another service.
- The log shows ids, counts and the words passed or failed. It never shows a name, a phrase or a path of a reference photo.
- The only network use is the first download of the model.

## Known limits

- **Small test.** The test had one object and photos of one place. It shows that the model separates this bottle from other things.
  It does not show how the check works with two objects that look alike.
- **Textured objects only.** A plain object has few keypoints. Such an object seldom passes the check, so its weak sightings stay weak.
  A check with a vision-language model (the other idea of the pipeline document) is not built.
- **No RANSAC filter.** The count comes from the confidence of the model only. A geometric filter can reduce wrong matches.
- **Distorted cuts.** The processor resizes each cut to 640 x 480 pixels. A tall cut, for example a bottle, is squeezed. The test still worked.
- **A check costs time.** One comparison takes about 1.7 seconds on the development machine (a CPU, no GPU). A check uses up to 3 comparisons.
  The worker waits for the check, so a frame with a weak sighting takes longer. The speed on the Raspberry Pi is not measured.
- **Reference photos.** The check needs confirmed reference photos with their files. If the files are gone, the sighting stays weak.
- **The newest hit wins.** The search checks the newest weak hit only. If it fails, the search does not look for an older strong hit.

## Tests

| File | What it tests |
|------|---------------|
| `tests/test_ai_verification.py` | The engine: the first pass stops the check, the best count, the limit, no reference, the crop at the edge, the switch-off |
| `tests/test_backend_verification.py` | The memory: promotion, failure, the order of the references, the limit, the cases without a check, errors, missing files, the log |
| `tests/test_backend_search_verification.py` | The search: a firm answer after a pass, a "maybe" after a fail, strong and fresh answers without a check, errors, missing files |
| `tests/test_backend_worker.py` | The worker gives the decoded frame to the memory |
| `tests/test_ai_verifier_model.py` (slow) | The real model on real photos: the bottle passes, other parts of the photos do not |
| `tests/test_ai_settings.py`, `tests/test_backend_settings.py` | The keys of the configuration |

The fast tests use a fake verifier that gives the number of matches that the test sets.
The slow tests need the model download and the photos (`SAARTHI_TEST_DATASET`).
