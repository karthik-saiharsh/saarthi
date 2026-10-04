# Manual Test Guide

## Purpose

This guide shows you how to run the application with `main.py` and follow one picture through all components.
You play three roles: the carer (adds an object), the wearable (sends pictures) and the patient (asks where the object is).
The expected results come from a real run with the real models. The run used the 14 photos in the folder `Dataset`.

## What you need

- The models in the Hugging Face cache. They are on this machine. A new machine downloads about 925 MB at the first start.
  If the download does not start, set `HF_HUB_DISABLE_XET=1`.
- The test photos: `/home/ks/Desktop/Project Saarthi/Dataset/` (`1.jpeg` to `14.jpeg`).
- A browser. Two terminals help: one for the application, one for the commands.
- About 8 seconds for each picture on this machine (a CPU, no GPU).

## 1. Start the application

```
cd "/home/ks/Desktop/Project Saarthi/saarthi-primus"
HF_HUB_OFFLINE=1 uv run python main.py
```

`HF_HUB_OFFLINE=1` stops the application from asking the internet for models that it already has.
The application is ready when you see `Serving on http://0.0.0.0:8080`. The start takes a few seconds.

The log shows the start-up order. Each line starts with the name of a component:

| Log line starts with | Component | What it did at the start |
|----------------------|-----------|--------------------------|
| `storage.` | Storage | Prepared the folder `storage/uploads` for the pictures |
| `db.` | Database and vector index | Opened `db/data/saarthi.db` and the index. The recovery repaired nothing |
| `ai.detector`, `ai.embedder`, `ai.query`, `ai.verifier` | The four models | Detector, region embedder, query encoder, verifier |
| `backend.` | Control flow | Started the upload step, the worker and the maintenance thread |
| `main` | Entry point | Held the order, and printed the address |

The pages:

| Address | Who uses it |
|---------|-------------|
| `http://localhost:8080/enroll` | The carer: add objects and photos |
| `http://localhost:8080/upload` | The wearable (or you): send pictures |
| `http://localhost:8080/find` | The patient: ask where an object is |

## 2. Add an object (the carer)

1. Open `/enroll`. Fill in the form:
   - Name: `Speckled bottle`
   - Kind of object: `a bottle`
   - Other words: `a water bottle`
2. Tap **Add object**. Add the photos `2.jpeg`, `3.jpeg` and `9.jpeg` in the photo box.
3. Watch the log. For each photo, the worker finds the object and makes four vectors. The three photos take about 20 seconds.
4. The cards show the crop of each photo. All three crops show the bottle. Tap **Accept** on each card.
5. The object badge changes to **Ready**.

What happened inside: the worker (`backend.worker`) asked the detector (`ai.detector`) for "a bottle" in each photo.
The embedder (`ai.embedder`) made four vectors for the box. After you accepted three photos, `backend.enrollment` calculated the thresholds.
The values from the run: `tau` 0.45 and `tau_strong` 0.55. (The formula alone gave 0.418. The floor `tau_min` raised it.)

## 3. Send pictures (the wearable)

Open `/upload`. Choose the photos in this order: `1, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14`.
You can also use the command line, one picture at a time:

```
curl -i -X POST --data-binary @"/home/ks/Desktop/Project Saarthi/Dataset/10.jpeg" http://localhost:8080/api/images
```

Watch the log. For each picture you see these steps: stored, claimed, candidates found, vectors made, matching, and done.
The matching line says how many sightings were found. A line from `backend.memory` tells you what happened to the last-seen record.

What you should see (the photo numbers are the file names):

| Photo | What it shows | Expected result |
|-------|---------------|-----------------|
| 1 | A notebook on a railing | One region, marked not good. No sighting |
| 4 | A balcony | One region. No sighting |
| 5 | A backpack on a chair | 3 regions. No sighting |
| 6, 7, 12 | A corridor, a black bag | No region |
| 8 | The bottle, partly out of the picture and blurred | One region, marked not good (the crop is blurred). No sighting. This is a known miss |
| 10 | The bottle in a hand | **Strong sighting**, similarity about 0.63, "in hand" |
| 11 | A soap dispenser | One region. **No sighting** (its similarity of 0.43 is below `tau`) |
| 13 | A shelf with shampoo and other bottles | 7 regions. No sighting |
| 14 | The bottle, small, on a desk | A **weak** sighting (similarity about 0.45). The check passes (`Verification: passed (best 153 of 30 needed ...)`). It counts as strong |

Photo 14 shows stage I5: the sighting is weak, the verifier compares it with a reference photo, and it passes.
The similarity of photo 14 is only 0.001 above `tau`. A small change can make it disappear.

Two more tests:
- Send `1.jpeg` again. The answer is `duplicate`. Nothing is stored twice.
- Send a text file. The answer is 415 (not an image).

## 4. Look inside the data

The pictures are in `storage/uploads/YYYY/MM/DD/`. The tables are in `db/data/saarthi.db`. Run this while the application runs (it only reads):

```
uv run python - <<'EOF'
import sqlite3
c = sqlite3.connect("file:db/data/saarthi.db?mode=ro", uri=True)
print("images (id, status, blur score):")
for row in c.execute("select id, status, round(blur_score) from image order by id"): print("  ", row)
print("regions per image (good / all):")
for row in c.execute("select image_id, sum(quality_ok), count(*) from region group by image_id"): print("  ", row)
print("sightings (image, similarity, strong):")
for row in c.execute("select image_id, round(similarity, 3), strong from sighting order by id"): print("  ", row)
print("last seen (image, similarity, strong, in hand):")
for row in c.execute("select image_id, round(similarity, 3), strong, in_hand from last_seen"): print("  ", row)
EOF
```

The image numbers here are the numbers in the table, not the file names. The first picture that you sent is image 1.
The vector index is a file (`db/data/regions.faiss`). The application saves it at shutdown, every hour, and after 200 changes.

## 5. Ask where the object is (the patient)

Open `/find`.

1. Tap **Speckled bottle**. The card at the top shows the name and "Seen N minutes ago". It also shows a picture of the bottle with its surroundings.
   The answer comes from the last-seen record. The record is the sighting of photo 14.
2. Write `where did I leave my water flask` in the box and tap **Find it**. The result is the same object.
3. Write `where are my keys`. The page says that it does not know the object.

The same through the command line. The number 1 is the id of the first object. A new object has the next number:

```
curl -X POST -d '{"query": "where is my speckled bottle"}' http://localhost:8080/api/search
curl http://localhost:8080/api/objects/1/last-seen
curl -o answer.jpg "http://localhost:8080/api/objects/1/last-seen/picture?view=frame"
```

What happened inside: `backend.query` turned your words into a vector (`ai.query`) and compared it with the description of each ready object.
`backend.search` read the last-seen record. If the record is older than 24 hours, it looks through the stored frames, newest first.

## 6. More things to try

| Test | Expected result |
|------|-----------------|
| Start a second copy while the first runs | The second copy stops at once: `Another copy of Saarthi is running` |
| Press Ctrl+C while a picture is processed | The worker finishes the picture. Shutdown ends with `Shutdown is complete` |
| Start again after that | `Start-up recovery` finds nothing to repair. The vectors load again |
| Stop with `kill -9` (use the process id), then start | The recovery repairs the data. The log says what it repaired |
| Set `verification.enabled = false` in `ai/config.toml` | The verifier is not loaded. The log says `verifier off` |
| Set the variable `SAARTHI_API_TOKEN=secret` before the start | The API routes need `Authorization: Bearer secret`. The pages stay open |

Restore every change of a config file afterwards.

## 7. Stop and clean up

Press Ctrl+C in the terminal of the application. Look for `Saved index`, `Instance lock released` and `Shutdown is complete`.

The test data stays on the disk. To remove it, stop the application and run:

```
uv run python - <<'EOF'
import db.api as db
import storage.api as storage
from db.schema import Image

storage.init_storage()
db.init_db()
for obj in db.gallery.list_objects():
    storage.delete_images(db.gallery.delete_object(obj.id))
ids = [i.id for i in Image.select()]
if ids:
    storage.delete_images(db.delete_images(ids))
db.checkpoint_vectors()
print("vectors left:", db.vector_count(), "| files left:", len(storage.list_stored_paths()))
EOF
```

## What you will notice that is not right yet

- **A weak false positive can still replace the last-seen record.**
  The rule of the hysteresis counts a strong sighting as the first hit.
  In an earlier test with the old thresholds, photo 11 did this. With the calibrated thresholds, photo 11 gives no sighting,
  so this test no longer shows it. The rule is not changed.
- **The blur check hides a true sighting.** In photo 8, the bottle is blurred by motion. Its similarity is above the threshold, but the crop fails the blur check, so the region is not used.
- **`/` opens the upload page**, not the find page.
- **The pages show all links to all users.** There is no separation between the carer and the patient.
- **The first answer can be old.** A picture takes about 8 seconds to process. The answer does not show how fresh the memory is.
