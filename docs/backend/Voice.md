# Voice and Description (planned)

## Status

This feature is **partly built**. Built: the scene classifier in the `ai` component (`ai/scene.py`, phase 1a),
the cache of the room scores and the lookup of neighbouring frames in the `db` component (`db/scenes.py`, phase 1b),
and the room rule, the sentence and the service in `backend/describe.py` (phase 1c).
The sentence is on the `/find` page (phase 1d), and the browser says it with a local voice (phase 2). Not built: the listening.
This document records the flow that the owner described and the stages that it adds to the pipeline.
It also lists the decisions that are still open.

## Purpose

The patient may have severe dementia. Such a patient cannot always read a screen or understand a photo.
So the application must listen to the patient, and it must answer with a voice that describes the place.

The flow:

1. The carer registers the belongings of the patient. They become the targets of the retrieval pipeline (stage E1 to E4, built).
2. The patient speaks to the application, for example: "I can't find my bottle." The first version does this in the UI.
3. The retrieval pipeline finds the object (stage Q1, built).
4. The application shows the picture of the answer. It also says what the picture shows, with a voice.
   Example: "Your bottle was last seen on a green table, about 20 minutes ago. The room looks like a bedroom."

## The new stages

| Stage | Name | Task | Status |
|-------|------|------|--------|
| Q0 | Listen | Turn the speech of the patient into text | Planned |
| Q1 | Answer | Find the object, and where and when it was seen | Built (`docs/backend/Search.md`) |
| Q2 | Describe | Describe the place in the frame of the answer: the surface, its colour and the kind of room | Planned |
| Q3 | Speak | Say the answer with a voice. The page also shows the picture | Built (phase 2) |

```
patient speaks ─▶ Q0 Listen ─▶ words ─▶ resolver (built) ─▶ object ─▶ Q1 Answer: frame, box, time (built)
                                                                          │
              the picture on the page ◀───────────────────────────────────┤
                                                                          ▼
                              Q2 Describe: surface, colour, room ─▶ sentence ─▶ Q3 Speak ─▶ voice
```

The resolver already accepts any words, for example "I can't find my water bottle". Q0 only has to give it the text.

## Rules

- **Audio stays on the device.** The application never stores it, never logs it and never sends it to another service.
  The only network use is the first download of a model.
- **The transcript is a private question.** The log never shows it, as for the words that a user types.
- **No browser speech service.** The speech service of a browser normally sends the audio to a cloud service. The application does not use it.
- **The voice says only what the system knows.** If the match is weak, the sentence says so.
  If the scene is not clear, the sentence leaves that part out.
- **A wrong description can mislead the patient.** The description must be careful. A fixed template and fixed word lists are safer than free text.
  The choice is open (see "Open decisions").
- **The picture stays on the page.** The voice adds to the picture. It does not replace it.

## The spoken sentence

The application builds the sentence from a template with slots. The slots are the object, the time, the surface, its colour and the kind of room.

| Case | Example |
|------|---------|
| Found | "Your bottle was last seen about 20 minutes ago. It was on a table. The room looks like a bedroom." |
| In hand | "You were holding your bottle about 20 minutes ago. You were in a room that looks like a bedroom." |
| Weak match | "Your bottle may have been on a table. I am not sure. Please look at the picture." |
| Old sighting | "The last time that I saw your bottle was a long time ago, yesterday evening." |
| Not seen | "I have not seen your bottle lately." |
| Not sure which object | "Which one do you mean? The blue bottle, or the green cup?" |
| Unknown object | "I do not know that object." |

**The time.** The sentence uses a relative time ("20 minutes ago").
For older sightings, it also uses a part of the day ("this morning", "yesterday evening").
The part of the day needs the local time zone. The server stores UTC. A new setting for the time zone is needed.
The time is the time at which the server received the frame (see `docs/backend/Memory.md`). A frame that arrives late has a wrong time.

## The room rule (built, `backend/describe.py`)

The room comes from the answer frame and its neighbouring frames in time (settings: `[describe]` in `backend/config.toml`).
The service names the room only if all these are true:

- At least `room_min_frames` frames have room scores (3 by default).
- At least `room_agree_fraction` of them have the room as best room (0.6).
- The average score of the room is at least `room_min_score` (0.4).

If one is false, the sentence says nothing about the room. The surface comes from the box with its surroundings.
It needs a score of at least `surface_min_score` (0.4). There is no surface when the object is in a hand.
On the 14 photos these settings gave 6 right rooms, 0 wrong rooms and 8 silent answers.
The values are start values. A test on real frame sequences must confirm them.

The sentences that the code builds:

| Case | Sentence |
|------|----------|
| Sure | "Your bottle was last seen about 20 minutes ago. It was on a table. The room looks like a bedroom." |
| In hand | "You were holding your bottle about 20 minutes ago." |
| Weak | "I am not sure, but I may have seen your bottle about 20 minutes ago. It may have been on a table. Please look at the picture." |
| Old | "The last time that I saw your bottle was 2 days ago. That was a long time ago." |
| Not seen | "I have not seen your bottle lately." |

An error of the scene model, of a file or of the database gives a sentence with the time only.

## The planned parts

| Part | Task |
|------|------|
| `ai/speech.py` (new) | The speech recognizer (Q0): audio in, text out |
| `ai/scene.py` (new) | The scene describer (Q2): a frame and a box in, the slots out (surface, colour, room) |
| A speech synthesizer (Q3) | Text in, audio out. The place is open: in the `ai` component, or in the browser |
| `backend/voice.py` (new) | `VoiceService`: audio in, the words and the result of the search out |
| `backend/describe.py` (new) | Builds the sentence from the answer and the slots. The result is cached for each frame |
| A route for the audio | A new HTTP route with a size limit, the optional token and fixed messages, as the other routes |
| The UI (`/find`) | A large "Talk" button, the words that the system heard, the picture, the text of the sentence and a "Say it again" button |
| `[voice]` and `[scene]` in the config | The models, the time zone and the limits |

**When to describe a frame.** A description is only needed for the frame of an answer. Describing every frame would cost too much.
There are two ways. The first: describe the frame when the last-seen record changes, and store the text with the record.
The second: describe it when the patient asks, and keep the text.
Both need a place to store the text. This is open.

## Limits of the browser

- **The microphone.** A browser allows it only on a page with HTTPS, or on `localhost`.
  A phone that opens `http://<address>:8080` cannot use the microphone. The owner chose phones and tablets over Wi-Fi, so the application needs HTTPS.
- **The sound.** A browser plays sound only after the user taps the page. The tap on "Talk" unlocks the sound for the answer.
- **The recording.** A browser records in a compressed format (for example WebM). The server needs a decoder for it.
  A simple way is to record raw samples at 16 kHz in the browser and send a WAV file. Then the server needs no new tool.

## The candidates

The licences below were read on the model pages. Check the licence of every model before use. The weights must come from a source that the module README names.

| Task | Candidate | Licence and notes |
|------|-----------|-------------------|
| Listen (Q0) | Whisper tiny, base, small (`openai/whisper-*`) | Apache-2.0, not gated, safetensors. Speaks many languages. About 39, 74 and 244 million parameters. The speed on the Raspberry Pi is not measured |
| Listen (Q0) | Other small recognizers (for example Vosk, Moonshine) | Not checked |
| Describe (Q2), fixed lists | SigLIP (`google/siglip-base-patch16-224`) | Apache-2.0. Chooses the best of a fixed list of words, for example the kinds of room. The colour can come from the pixels near the object |
| Describe (Q2), fixed lists | CLIP (`openai/clip-vit-base-patch32`) | The model card gives no licence. Check it |
| Describe (Q2), free text | SmolVLM 256M or 500M (`HuggingFaceTB/SmolVLM-*`) | Apache-2.0. Writes free text. A small model can invent details |
| Describe (Q2), free text | Florence-2 base (`microsoft/Florence-2-base`) | MIT. It may need code from its repository. The rules do not allow that. Check |
| Speak (Q3) | Kokoro 82M (`hexgrad/Kokoro-82M`) | Apache-2.0. Languages: check |
| Speak (Q3) | Piper (`rhasspy/piper-voices`) | The repository has the MIT licence. Each voice has its own licence. Check the voice |
| Speak (Q3) | The speech synthesis of the browser | No model on the server. The voices belong to the device. Some browser voices use a cloud service, so only local voices are allowed |

The scene describer is the risky part. The test on the 14 photos of `../Dataset` is done. Its results are in the next section.

## The test of the description

The scripts and the numbers are in `benchmarks/scene/README.md`. The test used the 14 photos of `../Dataset` on the development machine.

| Method | Result |
|--------|--------|
| Free text (SmolVLM-256M, 513 MB) | Real sentences with a full prompt. But a part is invented, with the same confident tone: a "fireplace", "high-rise buildings", a "painting". 2 to 5 seconds a photo |
| Fixed lists (SigLIP base, 813 MB) | The room is right for 11 of 14 photos, and the surface is mostly right. The wrong rooms have a high score (0.71, 0.93, 0.83). About 2 seconds a photo |
| Colour from the pixels | Clearly right for 2 of 6 photos. The light and the objects near the object change it |
| Furniture from the detector (no new model) | Not usable. It finds furniture that is not there, and it makes wrong room claims |

What the test shows:

- **Free text from a small model is not safe** for a patient who may believe the voice.
- **Fixed lists give a safer kind of error**, a wrong word from a closed list. But the room is wrong when the photo shows only a floor.
  A wrong room has a high score, so a score cutoff cannot remove it.
- **The neighbours in time can fix this.** The average of the room scores of a photo and its 2 neighbours gave 13 of 14 rooms.
  This is an illustration. It assumes that the photo numbers are the order in time. A test on real frame sequences must confirm it.
- **The colour is the weakest part.** The sentence should not state a colour until a better method is found.
- **The model size is a cost.** SigLIP base is 813 MB. The application already loads four models.

The owner chose this on 2026-11-09: use fixed lists for the room and the surface.
Take the room from the neighbouring frames in time. Say nothing about the room when the frames do not agree.
Leave the colour out for now. Say "looks like", and keep the picture on the page.
The sentence has no colour, so the example of a "green table" in the first plan is now "a table".

## Decisions

The owner decided these on 2026-11-09:

| # | Question | Decision |
|---|----------|----------|
| 1 | Where does the UI run? | On a **phone or tablet over Wi-Fi**. So the application must serve **HTTPS** (see "HTTPS" below) |
| 2 | How does listening start? | The patient or carer **taps a large button**. There is no wake word and no always-on microphone |
| 3 | Which languages? | **English only** to start. The object names, the recognizer and the voice are English |
| 4 | How is the place described? | **Fixed lists, with the room taken from neighbouring frames in time.** No colour. The test is in "The test of the description" |
| 5 | Where does the voice come from? | **The local voices of the device**, through the browser. No model on the server. A device without a local English voice stays silent |

These decisions are still open:

| # | Question | Why it matters |
|---|----------|----------------|
| 6 | Which time zone? | The part of the day in the sentence |
| 7 | How much memory and which CPU does the target device have? | Every new model adds memory and time. The application already loads four models |
| 8 | How does a patient who cannot tap answer "Which one do you mean?" | A yes or no by voice needs a dialogue of more than one turn |

## HTTPS

A phone that opens `http://<address>:8080` cannot use the microphone. So the application must serve HTTPS. The plan:

- New settings in `backend/config.toml`: the paths of a certificate file and of a key file. Without them, the server stays on HTTP, as now.
- The certificate must name the address that the phone uses (an IP address, or a name such as `saarthi.local`).
  An address that changes needs a new certificate. A fixed address or name helps.
- The phone must trust the certificate. A small local authority, installed once on each phone, is the clean way.
  A certificate with a warning that the user accepts may or may not give a microphone. This must be tested on the real phone and browser.
- The optional token and all routes stay as they are. HTTPS also protects the audio on the local network.
- The guide for the carer must say how to make the certificate and how to install it.

## The planned phases

Each phase is one commit.

0. **The test of the description.** Fixed lists against free text, on the 14 photos. The owner chooses. (No application code.)
1. **Describe (Q2).** The scene describer, the sentence, the cache, and the sentence as text on the page.
2. **Speak (Q3).** The voice and a "Say it again" button.
3. **HTTPS.** The certificate settings, and the guide to make and to install a certificate.
4. **Listen (Q0).** The audio route, the recognizer, and the "Talk" button.
5. **The dialogue.** The spoken answers for the cases without an object, and for the choice between objects.
6. **The docs.**

## Risks

- **A wrong description.** The system could say "bedroom" for a bathroom, or name a wrong colour. A patient may believe it.
  The sentence must be careful, and the picture must stay on the page.
- **Time and memory.** The recognizer, the describer and the voice add three models to four. The speed on the Raspberry Pi is not measured.
- **Speech of an older patient.** A recognizer can make more mistakes with a quiet or slow voice. This is not tested.
- **Noise.** A home has a TV and other voices. The recognizer can hear the wrong person.
