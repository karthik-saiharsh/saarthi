# Manual Test Guide

## Purpose

This guide shows you how to run the whole pipeline with `main.py`, from the first object to the spoken answer.
You play three roles: the carer (adds an object), the wearable (sends pictures) and the patient (asks where the object is, by voice or by text).
The pipeline: enrollment (E1 to E4), ingestion and memory (I1 to I6), the search (Q1), the description (Q2), the voice (Q3) and the listening (Q0).
The expected results come from a real run with the real models. The run used the 14 photos in the folder `Dataset`.

## What you need

- The models in the Hugging Face cache. They are on this machine. A new machine downloads about 3.8 GB at the first start.
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
| `ai.detector`, `ai.embedder`, `ai.query`, `ai.verifier`, `ai.scene`, `ai.speech`, `ai.open_detector`, `ai.language` | The models | Detector, region embedder, query encoder, verifier, scene classifier, speech recognizer, broad detector, language model |
| `backend.` | Control flow | Started the upload step, the worker and the maintenance thread |
| `main` | Entry point | Held the order, and printed the address |

The pages:

| Address | Who uses it |
|---------|-------------|
| `http://localhost:8080/enroll` | The carer: add objects and photos |
| `http://localhost:8080/upload` | The wearable (or you): send pictures |
| `http://localhost:8080/find` | The patient: ask where an object is |
| `http://localhost:8080/memory` | The carer or you: see every frame, and what the pipeline found in it |

### Quick path: the whole run from the command line

You can run steps 2, 3 and 5 without a browser. Start the application (step 1), then use a second terminal.
`D` is the folder of the photos. The commands do not wait for the worker, so keep the `sleep` lines.

```
D="/home/ks/Desktop/Project Saarthi/Dataset"
# Step 2: the carer adds the object and three reference photos, and waits for the worker (about 40 s)
curl -s -X POST -d '{"name":"Speckled bottle","category":"a bottle","synonyms":["a water bottle"]}' localhost:8080/api/objects
for n in 2 3 9; do curl -s -X POST --data-binary @$D/$n.jpeg localhost:8080/api/objects/1/photos; echo; done
sleep 40
for p in 1 2 3; do curl -s -X POST localhost:8080/api/objects/1/photos/$p/confirm -o /dev/null -w "%{http_code} "; done; echo
curl -s localhost:8080/api/objects/1          # "status":"ready", "tau":0.45, "tau_strong":0.55
# Step 3: the wearable sends the pictures. Wait until the log stops (about 90 s)
for n in 1 4 5 6 7 8 10 11 12 13 14; do curl -s -X POST --data-binary @$D/$n.jpeg localhost:8080/api/images -o /dev/null -w "$n:%{http_code} "; done; echo
# Step 5: the patient asks
curl -s -X POST -d '{"query": "where is my speckled bottle"}' localhost:8080/api/search
```

The numbers 1, 2 and 3 are the ids of the first object and its photos on an empty database.
The route `/api/search` gives the answer data. The sentence and the voice are made by the page `/find` (step 5).

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

### The memory page

Open `/memory` after the pictures are processed. You see one card for each frame, with the status and a line such as `1 match, 1 region, 14 other boxes`.
Tap a card. The details show the picture with the boxes: green for a match of a registered object, orange for a box without a match,
dashed for the other objects that the broad detector named. Photo 10 (the bottle in a hand) shows the match "Speckled bottle - match 0.63 - strong - in a hand".
Use the filter to see only rejected or failed frames. The page shows every picture, so open it only on a network that you trust.

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
print("room scores that the description cached (image, model):")
for row in c.execute("select image_id, model from frame_scene order by image_id"): print("  ", row)
EOF
```

The table `frame_scene` is empty until the first answer. The description fills it for the answer frame and its neighbours.
The image numbers here are the numbers in the table, not the file names. The first picture that you sent is image 1.
The vector index is a file (`db/data/regions.faiss`). The application saves it at shutdown, every hour, and after 200 changes.

## 5. Ask where the object is (the patient)

Open `/find`. The page has four parts, from top to bottom: the result, the **Talk** button, the object buttons and the box for your own words.

1. Tap **Speckled bottle**. The card at the top shows the name and "Seen N minutes ago". It also shows a picture of the bottle with its surroundings.
   The answer comes from the last-seen record. The record is the sighting of photo 14.
2. Write `where did I leave my water flask` in the box and tap **Find it**. The result is the same object.
3. Write `where are my keys`. The page says that it does not know the object.

### What the page says and shows

The answer has the time, the picture and a sentence. The browser also says the sentence with a voice of the device.
In the run with the real models (all 14 photos sent within about 2 minutes), the sentence was:

> Your Speckled bottle was last seen 2 minutes ago. It was on a desk.

The surface "a desk" comes from the scene model (`ai.scene`) that looks at the box of the bottle with space around it (photo 14).
The room is **not** named in this run: the log says `Room rule: 5 frame(s), 2 agree, no room named`.
The rule names a room only if at least 3 frames have scores, 60 % of them agree, and the average score is at least 0.4.
A frame that was sent more than 2 minutes before or after the answer frame does not count. This silence is correct behaviour, not an error.

| Test | Expected result |
|------|-----------------|
| Tap **Say it again** | The browser repeats the sentence. If the device has no local English voice, it stays silent. The text stays on the page |
| Ask for an object that was never seen | "The computer has not seen it lately." |
| Wait more than 6 hours, then ask | The sentence says a part of the day ("this morning", "yesterday evening") instead of "N hours ago". See `describe.time_zone` in `backend/config.toml` |

The time zone: the setting `describe.time_zone` is empty, so the application uses the time zone of the computer. Set an IANA name (for example `Asia/Kolkata`) if the computer runs on UTC.

### The Talk button (listening)

1. Open `http://localhost:8080/find` **on this computer**. `localhost` lets the browser use the microphone without HTTPS.
2. Tap **Talk** and allow the microphone. The button says **Stop**.
3. Say: "where is my water flask". Tap **Stop** (or wait 15 seconds).
4. The page shows **I heard: ...** and then the answer, with the picture, the sentence and the voice. The recognition takes about 1 second.

Also try: a silent recording (the page says that the computer did not hear you), and "where are my keys" (the page says that it does not know the object).
The audio is never stored and never logged. The log line for a recording says only `Speech recognized: N characters`.
You can test the route without a microphone: `curl -s -X POST --data-binary @recording.wav -H 'Content-Type: audio/wav' localhost:8080/ui/voice`
(a WAV file with 16 kHz, 16 bits and one channel). A file of another kind gives `The recording is not a usable WAV file.`

### From a phone or tablet (HTTPS)

The microphone on a phone needs HTTPS. Follow `docs/backend/HTTPS.md` (make a local certificate, install it on the phone, set `server.tls_certfile` and `server.tls_keyfile`),
then open `https://<address>:8080/find` on the phone. This path is **not tested on a real phone yet**. Tell the developer what you see.

The same through the command line. The number 1 is the id of the first object. A new object has the next number:

```
curl -X POST -d '{"query": "where is my speckled bottle"}' http://localhost:8080/api/search
curl http://localhost:8080/api/objects/1/last-seen
curl -o answer.jpg "http://localhost:8080/api/objects/1/last-seen/picture?view=frame"
```

What happened inside: `backend.query` turned your words into a vector (`ai.query`) and compared it with the description of each ready object.
`backend.search` read the last-seen record. If the record is older than 24 hours, it looks through the stored frames, newest first.

### The Talk button

Open `http://localhost:8080/find` on this computer (`localhost` allows the microphone without HTTPS).
Tap **Talk**, allow the microphone, say "where is my speckled bottle", and tap **Stop**.
The page shows "I heard: ..." and then the answer, with the sentence and the voice.
On a phone, you need HTTPS first (`docs/backend/HTTPS.md`).
Try also a silent recording: the page says that the computer did not hear you.

## 6. More things to try

| Test | Expected result |
|------|-----------------|
| Start a second copy while the first runs | The second copy stops at once: `Another copy of Saarthi is running` |
| Press Ctrl+C while a picture is processed | The worker finishes the picture. Shutdown ends with `Shutdown is complete` |
| Start again after that | `Start-up recovery` finds nothing to repair. The vectors load again |
| Stop with `kill -9` (use the process id), then start | The recovery repairs the data. The log says what it repaired |
| Set `verification.enabled = false` in `ai/config.toml` | The verifier is not loaded. The log says `verifier off` |
| Set only `server.tls_certfile` in `backend/config.toml` | The start stops: both keys must be set, or both empty |
| Set `server.tls_certfile` and `server.tls_keyfile` to missing files | The start stops with `server.tls_certfile is not a file` |
| Set `speech.enabled = false` in `ai/config.toml` | The speech model is not loaded. The page has no **Talk** button |
| Set `scene.enabled = false` in `ai/config.toml` | The sentence has the time only. It names no surface and no room |
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

- **The voice and the Talk button are not tested on a phone.** The sound depends on the voices that the device has.
- **Noise can give wrong words.** The speech model can write words for a noisy recording. The silence check only catches near-silence.
- **The room is often not named.** The rule prefers silence to a wrong room (see step 5).
- **The blur check hides a true sighting.** In photo 8, the bottle is blurred by motion. Its similarity is above the threshold, but the crop fails the blur check, so the region is not used.
- **`/` opens the upload page**, not the find page.
- **The pages show all links to all users.** There is no separation between the carer and the patient.
- **The first answer can be old.** A picture takes about 8 seconds to process. The answer does not show how fresh the memory is.
