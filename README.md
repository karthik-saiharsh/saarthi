# Saarthi

Edge Efficient Episodic Memory Aid

## What Saarthi does

Saarthi helps elderly people and dementia patients to find lost objects.
A wearable camera sends pictures. The application remembers where it saw each object.

1. The carer registers the belongings of the patient on the page `/enroll`.
2. The patient asks where an object is. The patient can type, choose the object, or speak (on the page, or on the wearable with a button).
3. The application finds the object in the stored pictures.
4. The application shows the picture. It also says what the picture shows: where and when the object was seen,
   and what the place looks like. A weak match is said with "I am not sure".

All pictures, questions and audio stay on the local device. The application sends nothing to an external service.
The audio is never stored and never logged.

## Components

Each component is in its own folder, has its own `README.md`, and does one task.

| Folder | Task |
|--------|------|
| `ai/` | Models: quality gate, object detectors, region vectors, matching, scene description, speech to text, the voice of the wearable, a small language model |
| `storage/` | Image files on disk, without metadata |
| `db/` | Stored data: SQLite metadata and the FAISS vector index |
| `backend/` | The control flow, the HTTP routes (also the routes of the wearable), the worker and the search. The only interface of the UI |
| `ui/` | The pages (NiceGUI). They call the backend only |
| `benchmarks/` | Measurements of the pipeline against a small dataset |
| `docs/` | Design documents and guides |
| `tests/` | Tests (`pytest`) |

## Pages

| Page | Use |
|------|-----|
| `/upload` | Upload pictures by hand |
| `/enroll` | Register an object: name, kind, and 5 to 10 photos |
| `/find` | Ask where an object is. Shows the picture and the spoken sentence |
| `/memory` | Browse every stored frame and what the pipeline saved for it |

## Local Project Setup

1. Clone the repo
1. Make sure you have [uv](https://docs.astral.sh/uv/) installed.
1. Run: `uv sync` in the project directory.
1. Since you already have `uv` installed, install `pre-commit` with `uv tool install pre-commit`.
1. Then run `pre-commit install` in this project directory.
1. Update the `pre-commit` to use the latest versions of `ruff` and `black` with `pre-commit autoupdate` in this project directory.
1. That's it! You're good to begin development :)

## Quick start (a computer or a Raspberry Pi)

- `./start` starts everything and hosts it on the local network. It installs the packages, makes a local certificate for HTTPS,
  and prints the addresses to open on the phone. The phone must use the same Wi-Fi. `./start --http` starts without HTTPS.
- `./clear` removes all stored data (pictures, database, indexes, registered objects). Stop the application first. `./clear --all` also removes the certificate.
- For a Raspberry Pi 5, see `docs/Raspberry Pi.md`.

## Run the application by hand

1. Run: `uv run python main.py`
1. Open `http://<address of this computer>:8080/upload` in a browser to upload pictures.
1. Open `http://<address of this computer>:8080/find` to ask where an object is. Choose the object, write it in your own words, or press the Talk button.
1. The wearable (or a script) sends one picture at a time: `curl -X POST --data-binary @photo.jpg http://<address>:8080/api/images`

- For a guided test of all components, see `docs/Manual Test Guide.md`.
- The first start downloads the AI models (about 3.9 GB). It needs internet one time.
- To require an access token for the routes under `/api`, set the environment variable `SAARTHI_API_TOKEN`
  before the start. Every request then needs the header `Authorization: Bearer <token>`.
  The talk routes of the wearable work only with a token (see `docs/backend/Wearable.md`).
- Only one copy of the application may run on the same data.
- To add the objects that you want to find, open `http://<address>:8080/enroll`. Add a name, the kind of object (for example "a bottle"),
  and 5 to 10 photos. Check the crop of each photo. The object is ready when 3 photos are accepted.
  The same steps are available through `/api/objects` (see `docs/backend/Enrollment.md`).
- Press Ctrl+C to stop. The application finishes the picture that it works on and saves its data.
- The settings are in the `config.toml` file of each module (`backend/`, `db/`, `storage/`, `ai/`).
  The design documents are in `docs/` and in the `docs` folder of each module.

## Voice and HTTPS

- The Talk button listens after a tap. English only. The server turns the speech into text with Whisper (base) and drops the audio at once.
- The browser says the answer with a local voice of the device. The page uses no voice model on the server.
- The wearable has no voice of its own. For the wearable, the server makes the spoken answer with a local voice (Piper) and sends it as a WAV file.
  The protocol for the firmware is in `docs/backend/Wearable.md`.
- The microphone of a phone or tablet works only on a page with HTTPS. `./start` makes the certificate and serves HTTPS.
  The phone must trust the local certificate authority: open `http://<address>:8081/ca.pem` on the phone while the application runs.
- For a manual HTTPS setup, see `docs/backend/HTTPS.md`. For the voice stages, see `docs/backend/Voice.md`.

## Contributors

- Karthik Saiharsh
- Nihanth Yadalam
- M Pranesh
- Leela Mahalakshmi

## A few ground rules

1. This project is divided into modules, and each module has its own readme stating its purpose.
2. This project will primarily be developed in python with uv as the project management tool.
3. Code quality is of prime importance in this project. Therefore, I urge the contributors to not rely blindly on LLMs.
4. Any change must be submitted as a PR first, and it must pass all code quality checks.

## Code Standards

- All the contributors must first go through [Tiger Style of Coding](https://tigerstyle.dev/).
- Though written for Zig, the Tiger Style has some incredible principles that can be incorporated into any project irrespective of language.
- In short if you are contributing to this project, you must confirm to the below standards:
  - Always mention data types next to variables, function formal arguments, and return types in function definition; do not keep the reader guessing
  - Always assert function parameters
  - Always handle exceptions explicitly
  - Use simple control flow, and avoid recursion unless really needed
  - Ensure to log things often
  - Do not use wildcard imports
  - Comments must have defined purpose and meaning in a given context, do not comment what is already obvious
  - Use a code formatter (preferably [Black](https://pypi.org/project/black/)) and always format your code before committing
  - Use `snake_case` naming convention
  - Write descriptive commit messages. PRs don't survive `git blame` commits are what future contributors are going to read
