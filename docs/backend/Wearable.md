# The Wearable

## Purpose

This document is the interface between the server and the wearable (an ESP32-CAM).
It is for the person who writes the firmware.

The wearable does three tasks:

1. It sends photos to the server.
2. It records a question when the user presses a button (microphone INMP441).
3. It plays the spoken answer on a speaker.

The wearable has no screen. So the answer must be complete without a picture.

## Status

| Part | Status |
|------|--------|
| The upload of photos, `POST /api/images` | Built (it existed before the wearable) |
| The question, `POST /api/talk` and `GET /api/talk/{id}` | Built (phase W1) |
| The spoken answer, `GET /api/talk/{id}/speech` | Built (phase W2) |
| A test with a real ESP32 | Not done |
| A test on the Raspberry Pi | Not done |

On the development CPU, a question without an enrolled object took 2.2 s from the upload to the audio.
Whisper heard the question correctly. The question was spoken by the voice of the server, not by a person.

## Decisions of the owner (2026-11-14)

| # | Question | Decision |
|---|----------|----------|
| 1 | What makes the speech? | **A local voice on the server**: Piper, voice `en_GB-alba-medium`. The audio never leaves the home network. The page keeps the voice of the browser |
| 2 | The licence | The engine `piper-tts` is GPL-3.0-or-later and includes espeak-ng (GPL). The voice data is CC BY 4.0. The owner accepted both for this academic project |
| 3 | What does the answer say? | **The same checked facts as the page.** The words are for listening: no "look at the picture". No new kind of fact |
| 4 | The words fit several objects | **The answer names the objects and asks again.** "Do you mean your watch, or your headset? Please press the button and say the name." The next press is a new question. The server keeps no dialogue |
| 5 | How does the answer come back? | **The wearable polls.** The question gives an id at once. The wearable asks for the state with the id |
| 6 | Access | **The talk routes always need the token.** Without a token on the server, they answer 503. The upload route keeps its optional token |

## The flow

```
button pressed ─▶ record WAV ─▶ POST /api/talk ─▶ 202 {"id"}
                                                    │
        ┌───────────── poll every 0.5 s ◀───────────┘
        ▼
GET /api/talk/{id} ─▶ "pending" ─▶ poll again
                  └─▶ "done", "speech": true ─▶ GET /api/talk/{id}/speech ─▶ play WAV
```

On the server, one thread makes the answer in this order:
Whisper (the words), the resolver (the object), the search (the place), the sentence, and Piper (the audio).
A job is done only when its audio is ready. So one poll gives the text and the audio together.

## Prepare the server

### Set the token

The talk routes do not work without a token.

1. Make a long random token:

   ```
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. Set the token in the shell that starts the application:

   ```
   export SAARTHI_API_TOKEN="<the token>"
   ./start
   ```

3. Put the same token in the firmware.
4. Make sure that the log does not show "The talk routes of the wearable are switched off".

The token also protects the upload route and the routes `/api/objects` and `/api/search`.
So the wearable must also send the token with each photo.

Keep the token out of the repository. Put it in `.env` or in the secrets file of the firmware.

### Set HTTPS

`./start` serves HTTPS with a certificate of a local authority (see `docs/backend/HTTPS.md`).
The wearable must trust this authority:

1. Get `ca.pem` from `http://<address>:8081/ca.pem` while `./start` runs.
2. Put the text of `ca.pem` in the firmware.
3. Give it to the TLS client, for example `WiFiClientSecure::setCACert(ca_pem)`.

Do not use `setInsecure()`. Without the check, another device on the Wi-Fi can read the token and the answers.

The certificate names the IP addresses of the server and `hostname.local`.
The name check of the TLS library on the ESP32 with an IP address is not tested. If it fails, use the `.local` name.

## The routes

All routes need the header `Authorization: Bearer <token>`.
All answers of the talk routes have `Cache-Control: no-store`.

### 1. Send a photo

```
POST /api/images
Authorization: Bearer <token>
Content-Length: <bytes>

<the JPEG bytes>
```

- The body is the JPEG itself. Do not use a form (`multipart/form-data`).
- `201`: the photo is stored. `200`: the same photo was stored before.
- `503` with `Retry-After`: too many photos wait. Wait that number of seconds and send again.
- The full list of codes is in `backend/http.py`.

The time of a photo is the time at which the server receives it.
So send each photo when it is taken. Do not keep photos and send them later.

### 2. Ask a question

```
POST /api/talk
Authorization: Bearer <token>
Content-Length: <bytes>

<the WAV bytes>
```

The answer is `202`:

```json
{"id": "q1Xv...22 characters", "status": "pending", "outcome": null, "text": null, "detail": null, "speech": false}
```

| Code | Meaning | What the wearable does |
|------|---------|------------------------|
| 202 | The question is kept | Poll with the id |
| 400 | The body is empty or a header is wrong | Fix the firmware |
| 401 | The token is missing or wrong | Fix the token |
| 413 | The recording is longer than about 15 s | Record less |
| 503 | No token on the server, listening is off, or 4 questions wait (`Retry-After`) | Read `error`. If `Retry-After` is set, wait and send again |

### 3. Poll for the answer

```
GET /api/talk/{id}
Authorization: Bearer <token>
```

```json
{"id": "...", "status": "done", "outcome": "found",
 "text": "Your bottle was last seen 12 minutes ago. It was on a table. The room looks like a bedroom.",
 "detail": null, "speech": true}
```

- Poll every 0.5 s. Stop after 60 s, and tell the user that the server does not answer.
- `status` is `pending` or `done`.
- `text` is the sentence to say. It is never empty when `status` is `done`.
- `speech` is `true` when the audio is ready.
- `detail` is a fixed message when the recording is not usable (for example "The recording is not a usable WAV file."). It helps you to find a firmware fault.
- `404`: no job has this id, or the job was deleted. A job is deleted 300 s after the question (`[talk] keep_seconds`).

### 4. Get the spoken answer

```
GET /api/talk/{id}/speech
Authorization: Bearer <token>
```

- `200`: a WAV file, `Content-Type: audio/wav`, with a `Content-Length`. The server sends it in parts of 4 KB.
  The wearable can start to play the first part before the last part arrives.
- `409`: the job is still pending. Poll again.
- `404`: the job has no audio. The voice is off (`[tts] enabled = false`) or the voice failed.
  Then the wearable has only `text`. It cannot say the answer.

### The outcomes

The `outcome` tells what the sentence says. Use it, for example, for a light. Always play the sentence.

| Outcome | Meaning | Example of `text` |
|---------|---------|-------------------|
| `found` | A sure sighting | "Your bottle was last seen 12 minutes ago. It was on a table. The room looks like a bedroom." |
| `maybe` | A weak sighting, or a thing that nobody registered | "I am not sure, but I may have seen your bottle 5 minutes ago." |
| `not_seen` | The object is registered, but it was not seen lately | "I have not seen your bottle lately." |
| `choose` | The words fit 2 or 3 objects | "Do you mean your watch, or your headset? Please press the button and say the name." |
| `unknown` | No object fits the words | "I do not know which object you mean. Please press the button and say its name." |
| `not_heard` | Silence, or a recording that is not usable | "I did not hear a question. Please press the button and ask again." |
| `failed` | An error of the server | "Something went wrong. Please try again later." |

A sentence can also say that the user held the object, that the sighting is old, and that a newer weak sighting is in another place.
The sentence never asks the user to look at a picture.

## The audio formats

### The recording (to the server)

| Property | Value |
|----------|-------|
| Format | WAV, PCM, no compression |
| Channels | 1 (mono) |
| Sample rate | 16 000 Hz |
| Sample size | 16 bits, signed, little-endian |
| Length | 0.5 s to 15 s (`[speech] max_seconds` in `ai/config.toml`) |

The INMP441 gives 24-bit samples in 32-bit I2S slots. Change each sample to 16 bits before you send it.
Shift the 32-bit value 16 bits to the right. Measure the level: a quiet recording can be taken as silence (`[voice] silence_rms`).

Put a WAV header of 44 bytes in front of the samples. The server refuses raw samples without a header.
The firmware can send the header first and then stream the samples. It must set `Content-Length` for the header and all samples.

### The answer (from the server)

| Property | Value |
|----------|-------|
| Format | WAV, PCM, a header of 44 bytes |
| Channels | 1 (mono) |
| Sample rate | 22 050 Hz (the rate of the voice of today) |
| Sample size | 16 bits, signed, little-endian |

Read the sample rate from the header. Do not use a fixed value: another voice can have another rate.
Set the I2S amplifier (for example a MAX98357A) to this rate.

## Privacy

- The recording stays in memory. The server drops it when the words are made. It is never written to a file and never logged.
- The words, the sentence and the audio are never logged. The log shows the first 6 characters of a job id and the outcome.
- A job (the sentence and its audio) stays in memory only, and is deleted 300 s after the question.
- The job id is a secret: with the id and the token, a device can get the answer. Do not print it on a debug port that other people can read.

## Settings

| File | Key | Meaning |
|------|-----|---------|
| `backend/config.toml` | `[talk] max_jobs` | The most questions that wait at one time (4). Also the most answers that are kept |
| `backend/config.toml` | `[talk] keep_seconds` | A job is deleted this many seconds after the question (300) |
| `backend/config.toml` | `[api] token_env_var` | The name of the variable that holds the token (`SAARTHI_API_TOKEN`) |
| `ai/config.toml` | `[tts] enabled` | `false`: no audio. The wearable gets the text only |
| `ai/config.toml` | `[tts] length_scale` | Above 1 the voice speaks slower (1.1) |
| `ai/config.toml` | `[tts] max_chars` | The longest text that is spoken (600 characters) |
| `ai/config.toml` | `[models] voice_*` | The voice files, their source and their SHA-256 |
| `ai/config.toml` | `[speech] max_seconds` | The longest recording (15 s) |

## Limits and risks

- **Not tested on the hardware.** Nobody has tested the routes with a real ESP32, a real INMP441 or a real speaker.
- **Not tested on the Pi.** On the Pi, a 4:3 frame takes about 24 s (`docs/Raspberry Pi.md`). So send at most one frame every 25 to 30 s, or the queue grows.
  The talk thread and the worker share the CPU. So an answer on the Pi can take several seconds more than on the development CPU.
- **The pins of the ESP32-CAM.** The camera uses most of the pins and one I2S peripheral.
  The microphone and the amplifier must share the second I2S peripheral and the few free pins. Make the pin plan before you buy parts.
- **The memory of the ESP32.** A TLS connection needs about 40 KB to 50 KB of RAM. A board with PSRAM is recommended.
- **No voice dialogue.** A question that fits several objects needs a second press. The server does not remember the first question.
- **Speech of an older patient.** Whisper can make more mistakes with a quiet or slow voice. This is not tested.
- **The licence.** The engine of the voice is GPL. Ask the owner again if the use of the project changes.
