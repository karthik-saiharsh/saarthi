# Raspberry Pi 5

## Purpose

This guide shows how to run Saarthi on a Raspberry Pi 5 with 8 GB of memory, and how to open it on a phone on the same Wi-Fi.
The Pi is the target device of the project. The application ran only on a development computer so far (x86, no GPU).
**The speed and the memory on the Pi are not measured yet.** The list "To measure on the Pi" at the end says what to check first.

## What you need

- A Raspberry Pi 5 with 8 GB, a power supply of 5 V and 5 A, and **active cooling**. Without cooling, the processor slows down within minutes of work.
- Raspberry Pi OS **64-bit** (Bookworm). The packages of this project do not exist for the 32-bit system.
- At least 15 GB of free space. The packages and the models need about 10 GB. A fast SD card, or better an SSD, helps: the models load from the disk.
- A network. The first start downloads the models (about 3.8 GB). Later starts work without internet.

## Setup (one time)

1. Copy the project to the Pi (for example with `git clone`). Use a plain path without a space, for example `/home/<user>/saarthi`.
2. Run `./deploy/pi_setup.sh`. It checks the machine (64-bit, memory, free space), installs the system packages
   (`curl`, `git`, `openssl`, `libgl1`, `libglib2.0-0`, `avahi-daemon`) and installs `uv`.
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

## Risks and open questions

- **Speed.** On the development computer one frame takes 10 to 13 s (most of it the detector of the registered objects), the broad detector 0.5 s, the speech model 1 s,
  and a rewrite 3 to 5 s. The Pi 5 is about 2 to 4 times slower for this kind of work (a guess). A frame may take 30 s or more.
  The wearable must not send faster than the Pi can process. The queue holds up to 500 frames (`ingest.max_pending`).
- **Memory.** The application loads 7 models (about 4 GB of weights in memory, 32-bit numbers). With the operating system this should fit in 8 GB. It is not measured.
- **Start time.** Loading the models takes about 8 s on the development computer. On a Pi from an SD card it can take much longer.
- **The description.** One sentence of the language model takes 8 to 17 s here. It runs in the background when no frame waits. On the Pi it may take a minute.
- **Heat.** Check the temperature with `vcgencmd measure_temp` during a long run.
- **The package `ultralytics`** needs `libgl1` (installed by the setup script). A start can fail with a message about `libGL.so.1` if the package is missing.

## To measure on the Pi (first test)

1. `./start` and note the time until "All components are ready", and `free -m` after the start.
2. Add an object (`/enroll`, 3 photos) and note the time for the photos.
3. Send 10 pictures (`docs/Manual Test Guide.md`) and note the time for each. Watch `vcgencmd measure_temp` and `free -m`.
4. Ask a question on the phone, with the voice (the time for the recognition) and with text.
5. Check that the answer, the picture and the voice work over HTTPS on the phone.
6. Stop with Ctrl+C. Look for `Shutdown is complete` in the log.
