# Raspberry Pi 5

## Purpose

This guide shows how to run Saarthi on a Raspberry Pi 5 with 8 GB of memory, and how to open it on a phone on the same Wi-Fi.
The Pi is the target device of the project. The application ran only on a development computer so far (x86, no GPU).
**The speed and the memory on the Pi are not measured yet.** The list "To measure on the Pi" at the end says what to check first.

## What you need

- A Raspberry Pi 5 with 8 GB, a power supply of 5 V and 5 A, and **active cooling**. Without cooling, the processor slows down within minutes of work.
- Raspberry Pi OS **64-bit** (Bookworm or Trixie). The packages of this project do not exist for the 32-bit system.
- At least 15 GB of free space. The packages and the models need about 10 GB. A fast SD card, or better an SSD, helps: the models load from the disk.
- A network. The first start downloads the models (about 3.8 GB). Later starts work without internet.

## Setup (one time)

1. Copy the project to the Pi (for example with `git clone`). Use a plain path without a space, for example `/home/<user>/saarthi`.
2. Run `./deploy/pi_setup.sh`. It checks the machine (64-bit, memory, free space), installs the system packages
   (`curl`, `git`, `openssl`, `rsync`, `libgl1`, `libglib2.0-0` (`libglib2.0-0t64` on Trixie), `avahi-daemon`) and installs `uv`.
3. Run `./start`. The first start installs the Python packages, makes the local certificate, downloads the models and starts the application.

## Open it on the phone

`./start` prints the addresses. The phone must use the same Wi-Fi as the Pi.

1. On the phone, open `http://<address>:8081/ca.pem` and install the certificate (one time for each phone).
   - iPhone: Settings, General, About, Certificate Trust Settings. Turn on "Saarthi local CA".
   - Android: Settings, Security, Install a certificate, CA certificate.
2. Open `https://<address>:8080/find` (the patient), `/enroll` (the carer), `/upload` or `/memory`.
3. The name `https://<hostname>.local:8080` also works on most phones (the Pi OS has `avahi`). The certificate includes this name.

The microphone works only with HTTPS and a trusted certificate. If the Pi gets a new address, `./start` makes a new certificate. The phone keeps the trust, because the authority stays the same.

## Daily use

| Command | What it does |
|---------|--------------|
| `./start` | Start Saarthi. Stop it with Ctrl+C. The worker finishes its picture first |
| `./start --http` | Start without HTTPS (no microphone on a phone) |
| `./clear` | Remove all stored data. Saarthi must not run. It asks first. `--yes` does not ask. `--all` also removes the certificate |

`./clear` leaves the packages, the models and the certificate. The next `./start` is a first run with no object and no picture.

## Start at boot (the service)

**Decision of the owner (2026-11-10): do not install the service on the shared Pi.** The Pi runs other work, so Saarthi starts by hand with `./start`. The files stay in `deploy/` for a Pi that Saarthi uses alone.

1. Run `./deploy/install_service.sh`. It installs and starts `saarthi.service`.
2. Watch it with `journalctl -u saarthi -f`. Stop it with `sudo systemctl stop saarthi`. Remove it with `./deploy/install_service.sh --remove`.
3. Stop the service before `./clear`.

The service stops Saarthi with the same signal as Ctrl+C and waits up to 150 seconds for the worker. The installer refuses a project path with a space.

## Update

1. Stop Saarthi. 2. `git pull`. 3. `./start`. The script installs new packages if the lock file changed.

## What was prepared for the Pi

- `pyproject.toml` uses the CPU-only wheels of torch and torchvision on all Linux machines, also on the Pi (aarch64). The wheel of PyPI for Linux needs several GB of CUDA libraries.
  `uv.lock` has the aarch64 wheels of torch, torchvision, faiss, opencv, numpy, pillow, tokenizers and safetensors.
- `./start` and `./clear` use no tool that is new to the Pi OS: `bash`, `openssl`, `python3`, `hostname`.
- The Python version is fixed to 3.11.15. `uv` downloads it if the system has another version.

## Copy the models from another machine

The Pi may download slowly (about 0.35 MB/s on the test network, so 3.8 GB take about 3 hours). The models can come from a machine that has them:
1. Copy the folders `models--...` of the seven models of `ai/config.toml` from `~/.cache/huggingface/hub/` to the same place on the Pi (for example with `rsync -a`).
   Use `rsync --inplace --append-verify` for a weak Wi-Fi: a plain copy starts a big file again after each break.
2. Copy the folder `blobs/` of the same `hub` folder, too. The newer cache keeps some files there, and the links in `models--.../snapshots` point to it.
   Without it, a model has no weights, and the start stops with "does not appear to have a file named model.safetensors".
3. `./start` checks that every weight file opens. If one is missing, it does not use the offline mode, and the application downloads it.
4. The file `ai/models/yoloe-11s-seg-pf.pt` is in the project folder. Copy it with the project.

## Measured on a Pi 5 (2026-11-10)

Test: Raspberry Pi 5, 8 GB, Raspberry Pi OS Trixie (64-bit, with the desktop), an SD card of 29 GB, Wi-Fi, **no active cooler**, the same 14 photos and one object as on the development computer.

| Measure | Pi 5 | Development computer |
|---------|------|----------------------|
| Start until "All components are ready" (models on the disk) | 27 s (the language model 21 s) | 8 s |
| A reference photo (locate the object, 4 views) | 18 to 25 s | about 10 s |
| A frame (registered detector, broad detector, matching) | 20 to 26 s, 48 s for a frame with 7 regions | 10 to 13 s |
| The results of the 11 frames | The same as on the development computer: a strong sighting in photo 10, and photo 14 verified with 153 keypoints | Same |
| A question about a registered object | 1.3 s | under 1 s |
| A question with the open search (names) | 0.7 s | 0.5 s |
| A rewrite of a description (language model) | 17 s | 4 s |
| The description of a frame (language model) | 33 s | 9 s |
| The recognition of 2 s of speech (Whisper base) | 2 to 5 s | 1 s |
| Memory in use after the start (the desktop included) | 4.3 GB of 8 GB, up to 5.8 GB while it worked | not measured |

The Pi is about 2 times slower for a frame and about 4 times slower for the language model.
**The Pi got too hot.** During the frames the temperature went to 86 degrees C, and the processor ran at 1.5 GHz and not at 2.4 GHz (`vcgencmd get_throttled` gave `0xe0006`).
So the times above are with throttling. An active cooler is needed. Run the test again with one.
A frame takes 20 to 26 s, so the wearable must send at most one picture every 30 s, or the queue grows.

## Risks and open questions

- **Speed.** See the measurements above. The wearable must not send faster than the Pi can process. The queue holds up to 500 frames (`ingest.max_pending`).
- **Memory.** The application loads 7 models. It used 4.3 GB of 8 GB after the start (the desktop of the Pi OS included) and up to 5.8 GB at work. A Pi OS without the desktop leaves more room.
- **The description.** One sentence takes 33 s on the Pi. It runs in the background when no frame waits.
- **Heat.** Check the temperature with `vcgencmd measure_temp` during a long run.
- **The package `ultralytics`** needs `libgl1` (installed by the setup script). A start can fail with a message about `libGL.so.1` if the package is missing.

## To measure again, with a cooler

1. `./start` and note the time until "All components are ready", and `free -m` after the start.
2. Add an object (`/enroll`, 3 photos) and note the time for the photos.
3. Send 10 pictures (`docs/Manual Test Guide.md`) and note the time for each. Watch `vcgencmd measure_temp` and `free -m`.
4. Ask a question on the phone, with the voice (the time for the recognition) and with text.
5. Check that the answer, the picture and the voice work over HTTPS on the phone.
6. Stop with Ctrl+C. Look for `Shutdown is complete` in the log.
