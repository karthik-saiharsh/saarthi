# Voice and Description

## Status

This feature is **partly built**. Built: the scene classifier in the `ai` component (`ai/scene.py`, phase 1a),
the cache of the room scores and the lookup of neighbouring frames in the `db` component (`db/scenes.py`, phase 1b),
and the room rule, the sentence and the service in `backend/describe.py` (phase 1c).
The sentence is on the `/find` page (phase 1d), the browser says it with a local voice (phase 2), the server can serve HTTPS (phase 3), and the Talk button listens (phase 4).
Not built: the dialogue by voice (phase 5).
The wearable (an ESP32-CAM with a microphone and a speaker) is built on the server side (2026-11-14): a spoken question through
`POST /api/talk`, and a spoken answer that a local voice on the server makes (Piper). See `docs/backend/Wearable.md`.
A checked sentence of the language model about the place is also built (see "The description of the place by the language model").
This document records the flow that the owner described and the stages that it adds to the pipeline.
It also lists the decisions that are still open. The sections "The planned parts" and "The candidates" keep the first plan for the record.

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
| Q0 | Listen | Turn the speech of the patient into text | Built (phase 4) |
| Q1 | Answer | Find the object, and where and when it was seen | Built (`docs/backend/Search.md`) |
| Q2 | Describe | Describe the place in the frame of the answer: the surface and the kind of room, from fixed lists. No colour | Built (phase 1) |
| Q3 | Speak | Say the answer with a voice. The page also shows the picture | Built (phase 2) |

```
patient speaks ─▶ Q0 Listen ─▶ words ─▶ resolver (built) ─▶ object ─▶ Q1 Answer: frame, box, time (built)
                                                                          │
              the picture on the page ◀───────────────────────────────────┤
                                                                          ▼
                                     Q2 Describe: surface, room ─▶ sentence ─▶ Q3 Speak ─▶ voice
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
  The owner chose fixed word lists (see "Decisions").
- **The picture stays on the page.** The voice adds to the picture. It does not replace it.

## The spoken sentence

The application builds the sentence from a template with slots. The slots are the object, the time, the surface and the kind of room.
There is no colour slot (see "Decisions"). The sentences that the code builds are in "The room rule".
A choice between objects and an unknown object are shown as text on the page. They have no spoken sentence yet (phase 5).

**The time.** The sentence uses a relative time ("20 minutes ago").
For a sighting that is at least `describe.day_part_min_hours` old (6), it uses a part of the day instead: "this morning", "this afternoon", "this evening",
"during the night" for today, and "yesterday morning" and so on for yesterday. A sighting that is older than yesterday keeps "N days ago".
The parts: night 22:00 to 04:59, morning 05:00 to 11:59, afternoon 12:00 to 16:59, evening 17:00 to 21:59.
The day and the part come from the time zone. The server stores UTC. The setting `describe.time_zone` is an IANA name (for example `Asia/Kolkata`).
If it is empty (the default), the application uses the time zone of the computer that runs it. A wrong time zone gives a wrong part of the day, so set it on a device that has no zone.
The time is the time at which the server received the frame (see `docs/backend/Memory.md`). A frame that arrives late has a wrong time.

## The description of the place by the language model (built)

A sure answer can use a checked sentence of the language model about the place, in place of the surface and the room.
The model gets the picture and the checked facts, and the backend checks every word of its sentence. The sentence is made in the background and stored.
See `docs/backend/Open Search.md`, "The scene description by one model".

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
| Sure, with a checked sentence of the language model | "Your bottle was last seen 2 minutes ago. The bottle is on the desk." (the sentence replaces the surface and the room) |
| In hand | "You were holding your bottle about 20 minutes ago." |
| In hand, weak | "I am not sure, but you may have been holding your bottle about 20 minutes ago. Please look at the picture." |
| Weak | "I am not sure, but I may have seen your bottle about 20 minutes ago. It may have been on a table. The room may look like a bedroom. Please look at the picture." |
| Old | "The last time that I saw your bottle was 2 days ago. That was a long time ago." |
| Not seen | "I have not seen your bottle lately." With an incomplete search: "I could not look at all the pictures." is added |
| Open search | "I am not sure, but I may have seen a laptop 12 minutes ago. ... Please look at the picture." ("a laptop", not "your laptop") |
| A newer weak sighting in another frame | The sentence of the answer, then: "I am not sure, but I may have seen it again 5 minutes ago, in another place. Please look at the second picture." |

An error of the scene model, of a file or of the database gives a sentence with the time only.
The code is `build_sentence` in `backend/describe.py`.

## The listening (built, phase 4)

- **Recognizer.** Whisper base (`openai/whisper-base`, Apache-2.0, 74 million parameters, 290 MB), English only. `ai/speech.py`, `AiEngine.transcribe`.
  The owner chose it on 2026-11-09. It ran in about 1 second for 2 seconds of audio on the development machine. The Raspberry Pi is not measured.
- **Recording.** The browser records and makes a WAV file (16 kHz, 16-bit, mono). The server needs no audio tool. `ui/static/js/talk.js`.
- **Route.** `POST /ui/voice` (`ui/voice_route.py`) reads the body with a size limit and calls `ui.ports.VoiceService`. The route has no token, as the pages have none.
- **Service.** `backend/voice.py`, `VoiceService`, checks the WAV header (mono, 16-bit, 16 kHz), the length (`speech.max_seconds`, `voice.min_seconds`) and the sound level.
  A silent recording (`voice.silence_rms`) gets no text and no model call, because a speech model writes words for silence.
- **Privacy.** The audio stays in memory. The log shows the length of the text only. The browser speech services are not used.
- **Limit.** A noisy recording still passes the silence check, and the model can write words for it. Then the resolver finds no object, and the page says so.
  A test with real speech of an older person is still to do.

## The planned parts

| Part | Task | Status |
|------|------|--------|
| `ai/speech.py` | The speech recognizer (Q0): audio in, text out | Built |
| `ai/scene.py` | The scene classifier (Q2): a frame and a box in, the room and the surface out | Built. No colour |
| The speech synthesizer (Q3) | Text in, audio out | The page: built in the browser (`ui/speech.py`). The wearable: Piper on the server (`ai/tts.py`), see `docs/backend/Wearable.md` |
| `backend/voice.py` | `VoiceService`: checks the WAV recording and turns it into text | Built |
| `backend/describe.py` | Builds the sentence from the answer, the surface and the room | Built |
| The route for the audio | `POST /ui/voice` (`ui/voice_route.py`), with a size limit and fixed messages. It has no token, as the pages have none | Built |
| The UI (`/find`) | A large "Talk" button, the words that the system heard, the picture, the text of the sentence and a "Say it again" button | Built |
| The config | `[speech]` and `[scene]` in `ai/config.toml`. `[describe]` and `[voice]` in `backend/config.toml` | Built |

**When to describe a frame.** A description is only needed for the frame of an answer. Describing every frame would cost too much.
- The room scores are made when an answer needs them, and kept for each frame in the table `frame_scene` (`db/scenes.py`).
- The sentence of the language model is made in the background, when the last-seen record changes and no frame waits.
  It is kept in the table `frame_description` (see `docs/backend/Open Search.md`).

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
| 5 | Where does the voice come from? | **The local voices of the device**, through the browser. No model on the server. A device without a local English voice stays silent. **Changed 2026-11-14 for the wearable only:** a local voice on the server (Piper), because the wearable has no voice. The page keeps the browser voice |
| 6 | Which time zone? | **A setting**, `describe.time_zone`. Empty (default) uses the time zone of the computer that runs the application |

These decisions are still open:

| # | Question | Why it matters |
|---|----------|----------------|
| 7 | How much memory and which CPU does the target device have? | Every new model adds memory and time. The application loads eight models. A Pi 5 with 8 GB used up to 5.8 GB (see `docs/Raspberry Pi.md`) |
| 8 | How does a patient who cannot tap answer "Which one do you mean?" | A yes or no by voice needs a dialogue of more than one turn. **For the wearable (2026-11-14):** the answer names the objects and asks the user to press the button again and say the name. The page is not changed |

## HTTPS

A phone that opens `http://<address>:8080` cannot use the microphone. So the application must serve HTTPS. The plan:

- Built (phase 3): the settings `server.tls_certfile` and `server.tls_keyfile` in `backend/config.toml`. Without them, the server stays on HTTP, as now. The guide to make and install a certificate is `docs/backend/HTTPS.md`.
- The certificate must name the address that the phone uses (an IP address, or a name such as `saarthi.local`).
  An address that changes needs a new certificate. A fixed address or name helps.
- The phone must trust the certificate. A small local authority, installed once on each phone, is the clean way.
  A certificate with a warning that the user accepts may or may not give a microphone. This must be tested on the real phone and browser.
- The optional token and all routes stay as they are. HTTPS also protects the audio on the local network.

## The planned phases

Each phase is one commit.

0. **The test of the description.** Fixed lists against free text, on the 14 photos. The owner chooses. (No application code.)
1. **Describe (Q2, built).** The scene classifier, the sentence, the cache, and the sentence as text on the page.
2. **Speak (Q3, built).** The voice and a "Say it again" button.
3. **HTTPS (built).** The certificate settings, and the guide to make and to install a certificate. The test on a real phone is still to do.
4. **Listen (Q0, built).** The audio route, the recognizer, and the "Talk" button. The test with real speech on a real phone is still to do.
5. **The dialogue.** The spoken answers for the cases without an object, and for the choice between objects.
   The wearable has the spoken answers for these cases (`backend/talk.py`). The page has not.
6. **The docs.**

## Risks

- **A wrong description.** The system could say "bedroom" for a bathroom, or name a wrong colour. A patient may believe it.
  The sentence must be careful, and the picture must stay on the page.
- **Time and memory.** The application loads eight models. On the Pi 5, the recognition of 2 s of speech took 2 to 5 s, and a sentence of the language model took 33 s (see `docs/Raspberry Pi.md`).
- **Speech of an older patient.** A recognizer can make more mistakes with a quiet or slow voice. This is not tested.
- **Noise.** A home has a TV and other voices. The recognizer can hear the wrong person.
