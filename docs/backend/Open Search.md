# Open Search (broad detection)

## Status

This feature is **partly built**. Built (phase 1): the broad detector in the `ai` component (`ai/open_detector.py`, `AiEngine.detect_open`).
Built (phases 2 and 3): the SigLIP vectors, and the table and the vector index for the boxes (`db/open_regions.py`). The worker step (phase 4) and the search (phase 5) are built. The sentence for an open result is the sentence of the registered path with "a laptop" in place of "your laptop". The query rewriter (phase 6) is built. Not built: the scene description by the language model, the page text for an open result, and the calibration.
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
| 2 | SigLIP vectors for the crops, and a text vector for a phrase (`ai/`) | Built |
| 3 | The tables for the boxes and a second vector index (`db/`) | Built |
| 4 | The worker step: detect, link to registered boxes, embed, store | Built |
| 5 | The search: the phrase of the user, the match with the stored names, then the SigLIP retrieval | Built |
| 6 | The sentence and the page | Planned |
| 7 | The calibration on real frames | Planned |

The search of phase 5, in this order:
1. Make a phrase from the words ("where is my laptop" gives "laptop").
2. Compare the phrase with the stored names with the sentence encoder (MiniLM). A score above a cutoff gives a hit.
3. If there is no hit, compare the SigLIP text vector of the phrase with the stored crop vectors.
4. Show the best hit as a weak result.

## The search (phase 5)

`backend/open_search.py`, `OpenSearchService.find(words)`. It runs only when no enrolled object fits the words
(`UiFindService` and the route `POST /api/search` call it when the resolver says `none`). The steps:

1. **The phrase.** `phrase_from_words` removes the question words and the pointing words: "where did I leave my laptop?" gives "laptop".
   A description keeps its words ("that thing I check my mails on" gives "thing i check my mails on").
2. **The names.** The sentence encoder (MiniLM) compares the phrase with each name that the detector gave to a stored box.
   A name with a score of at least `label_min_score` (0.6) fits. The boxes of the names within `label_tier` (0.15) of the best score can be the answer.
   The newest frame wins.
3. **The pictures.** If no name fits, the SigLIP vector of the phrase is compared with the vector of each stored box.
   The best box must reach `visual_min_probability` (0.8) and beat the best neutral phrase ("a wall", "an object") by `visual_margin`.
4. The answer is weak (`strong` is false, `source` is `open`). The sentence says "I am not sure, but I may have seen a laptop 12 minutes ago. Please look at the picture."
   The surface and the room come from the description (`docs/backend/Voice.md`). The words of the user are never logged.

The search reads the boxes of `done` frames of the last `max_scan_hours` (72), at most `max_boxes` (5000), and compares them in memory.
The route gives `status: "open"` with the answer, its `label` and `via` (`label` or `visual`).

### Test on 18 frames (7 of the owner, 11 of the dataset; 158 boxes)

| Question | Result |
|----------|--------|
| "where is my laptop" | `laptop` through the names (score 1.0). Correct |
| "where is my phone" | `smartphone` through the names (0.75). A phone is in the frames |
| "what I wear on my feet" | `footwear` through the names (0.76) |
| "where is my water bottle" | `bottle cap` through the names (0.74). The small bottle on the desk was named "bottle cap" by the detector |
| "where is my piano", "my umbrella" | No result (correct, there is none) |
| "that thing I check my mails on" | `passbook` through the pictures (0.12). **Wrong** (it is a laptop) |
| "something to sit on", "where are my headphones", "my charger", "the thing I drink water from" | A wrong box through the pictures (0.18 to 0.63). **Wrong** |

What this shows:
- The step with the names works when the words name the object or a near word.
- The step with the pictures was wrong in every case of this test, also with a probability of 0.63. So the start value of `visual_min_probability` is 0.8:
  the step answers only when it is very sure. A wrong answer is worse than no answer for the patient. Phase 7 sets this value on the new images.
- A description that needs world knowledge ("check my mails" means a laptop) fails in both steps. A small language model must rewrite such a question into object names (see below).

### The query rewriter (built)

`ai/language.py`, `LanguageModel.rewrite_query`, with Qwen3.5-0.8B (`AiEngine.rewrite_query`). Step 3 of the search runs it when the names and the pictures fail and the words are a description (3 words or more).
The model writes up to 3 common names ("laptop, computer, tablet"). The code keeps short names that have letters only. Step 1 runs again with each name, and a name must fit the name of a stored box almost exactly (`rewrite_min_score` 0.9).
The best score wins. The model runs for 3 to 5 s on this CPU. The instruction and the two examples are in `ai/config.toml`, `[language]`.

**A correction of the first test.** The first test gave "that thing I check my mails on" the answer "laptop, computer, tablet". That instruction had the example "laptop, computer, tablet", so the model repeated it.
Without such an example, the model answers "mail, inbox, folder" (5 prompts tested). It does not know that mail is read on a computer. The first count of 16 of 20 right answers was too high for the same reason.
The final instruction has no example that is a right answer. On 30 questions it gives 28 right answers. It misses "that thing I check my mails on" and "what I eat soup with".
A larger model may know more: Qwen3.5-2B is being downloaded for the same test.

**A wrong answer in the real test, and the fix.** The question "where is my piano" gave the name "keyboard", and the search showed a "laptop keyboard" (score 0.83). The first version of step 3 ran for every question.
Now it runs only for a description (3 words or more) and a name from the model must score 0.9. The question "the thing I carry my stuff in" still finds the bag (name "bag", score 1.0).

### The old text of this section

The first plan of the rewriter, kept for the record:

A small model rewrites a question that both steps could not answer into two or three object names ("laptop, computer, tablet").
The steps run again with those names. The names only feed the search. They never reach the voice, and a name still needs a stored box that fits it.
The owner asked to test one vision-language model that does this job and the scene description (see "The scene description").

## The scene description by one model (plan, not built)

The owner wants the sentence at the end of the pipeline to be a full description of the scene, not only the time. A model reads the full frame
and the facts that the system knows: the object, the time and the room (see `docs/backend/Voice.md`). The owner also wants to know if
**one model** can do this job and the query rewriting.

A multimodal model has one language model inside. So the same weights can take a text-only prompt (the rewriting) and an image with facts (the description).
The candidates below were read from their model cards on 2026-11-10. All have the licence Apache-2.0, need no code from the repository, and have safetensors files.
The installed `transformers` (5.14.1) has the code for `qwen3_5`, `qwen3_vl` and `smolvlm`.

| Model | Size of the weights | Notes |
|-------|---------------------|-------|
| `Qwen/Qwen3.5-0.8B` | 1.75 GB | Multimodal from the start. Newest (2026-03). The first to test |
| `Qwen/Qwen3.5-2B` | 4.55 GB | The next size, if 0.8B is not good enough |
| `Qwen/Qwen3-VL-2B-Instruct` | 4.26 GB | A vision model of 2025 |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 2.03 GB | The model family of the first test (see `benchmarks/scene/README.md`) |
| `OpenGVLab/InternVL3_5-1B-HF` | 2.12 GB | |

What the first test of a small vision model showed (SmolVLM, 14 photos): the text was fluent, but a part was invented ("a fireplace", "a painting"),
with the same confident tone. The patient may believe the voice. So the plan is a **grounded description**:
- The prompt contains only facts that the system checked: the object, the time, the room (from the room rule), the surface, and the names of the boxes of the broad detector in the frame.
- The prompt tells the model to use only these facts and to describe nothing else.
- A check compares the nouns of the answer with the facts. If a noun is not in the facts, the system says the template sentence of today.
- The picture stays on the page, and a weak result keeps "I am not sure".

### Result of the test of Qwen3.5-0.8B (2026-11-10)

The model has 853 million parameters (the vision part included). It loads in 1.8 s. The times below are from this development CPU (no GPU, 32-bit numbers).

| Job | Result | Time |
|-----|--------|------|
| Rewrite a description into 3 object names (20 questions) | 16 of 20 right without examples, 18 of 20 with 2 examples. "that thing I check my mails on" gives "laptop, computer, tablet" | 3 to 5 s |
| Describe a frame (image and facts) | Fluent, and right for 3 of 6 frames, with the errors below | 10 to 17 s (330 to 560 prompt tokens) |
| Describe with facts only (no image) | Worse: it invents a table, a lamp and a clock | 6 to 8 s |

The model copies what it gets. The errors in the description test:
- It repeats a wrong name of the broad detector: "on a bed" (the detector named a soap tray "bed") and "with a cork nearby" (the detector named a cable "cork").
- It adds details that nobody checked: "on a ledge", the colours "black" and "light grey", "a wooden handle". The prompt told it not to.
- Wrong names of the detector are common (see "Test on 18 frames"), so the facts in the prompt must not contain names that nobody checked.

What this means for the plan:
- The **rewriting** works well enough as a fallback. It needs no image, and a wrong guess still needs a stored box that fits.
- The **description** is not safe without gates: only checked facts in the prompt (the object, the time, the room, the surface, and detector names above a high confidence that SigLIP confirmed),
  a check of the nouns in the answer, and the template sentence as the fallback.
- The description takes 10 to 17 s here, and a Raspberry Pi is slower. So it must be made in the background, when a last-seen record changes, and stored. The answer then uses the stored text.

The plan of the test (done for Qwen3.5-0.8B):
1. Rewriting: 20 questions, for example "that thing I check my mails on". Count the answers that name the right object.
2. Description: the 14 photos and the frames of the owner. Count the invented nouns, and read the sentences.
3. Time: seconds for one rewrite and for one description on this CPU. A Raspberry Pi 5 is slower. A report on text-only models of 1 to 3 billion parameters
   gives about 2 to 11 tokens a second there, and the image adds time before the first word. The description can run when the answer is found, or in the background.
4. Choose the smallest model that passes. If no small model passes the safety test, the template sentence stays and the model is used for the rewriting only.

## The worker step (phase 4)

After the registered path of a frame (detect, embed, match, memory), the worker runs the broad detector, makes a SigLIP vector for each box,
and stores the boxes (`Worker._store_open_boxes`, `docs/backend/Worker.md`, step 10).
- The step needs the broad detector and the scene model. If one is off, the step does nothing.
- An error of the detector, of the scene model or of the database is logged. The frame is still `done`: the open search is an extra.
- A box is linked to the region of a registered candidate when the IoU is at least `worker.open_link_iou` (0.5). A box with a link is kept.
  The owner agreed to this rule on 2026-11-10. The link tells the search that the registered path has seen the same thing.
- The backend refuses to start if the scene model makes vectors of another size than `vectors.open_dimension` in `db/config.toml`.

Run on the 14 photos of the dataset, with the real models and an empty database (all 11 frames `done`, no error):

| Result | Value |
|--------|-------|
| Boxes stored | 108 in 11 frames (4 to 20 a frame), 9 of them linked to a registered candidate |
| Laptop | Found in photo 14 (0.98) |
| Bottle | The bottle in the hand (0.82) and the bottle on the shelf (0.68) are linked to registered candidates |
| Time | A frame took 10 to 13 s in total (Grounding DINO is most of it). The detector took about 0.5 s, and SigLIP about 0.25 s a box |

## The vectors (phase 2)

- `AiEngine.embed_open_crops(image, boxes)` makes a unit-length SigLIP vector for each box (the box with `open_detection.crop_padding` around it).
- `AiEngine.embed_open_phrases(phrases)` makes a vector for each phrase. A phrase without an article gets one ("laptop" gives "a laptop").
- `AiEngine.open_pair_scores(crops, phrases)` gives the SigLIP probability of each pair, shape (crops, phrases). The pairs are judged alone, so a row does not add up to 1.
- `open_vector_dimension` is the size of the vectors (768 for the model that is in use). `open_vector_model` is the name to store with the vectors.
- The vectors come from the scene model (SigLIP base), so the scene model must be on.
- **Finding:** the size of the score changes a lot from crop to crop. A real laptop got 0.89 and 0.66 in two frames of the owner, and 0.08 in
  one dataset photo, while the other phrases got 0.00. The order is right, but a fixed cutoff of 0.5 would miss the 0.08 case.
  So phase 5 ranks the crops and compares the best score with the other phrases. Phase 7 sets the cutoffs on real frames.

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
