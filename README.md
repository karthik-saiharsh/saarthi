# Saarthi

Edge Efficient Episodic Memory Aid

## Local Project Setup

1. Clone the repo
1. Make sure you have [uv](https://docs.astral.sh/uv/) installed.
1. Run: `uv sync` in the project directory.
1. Since you already have `uv` installed, install `pre-commit` with `uv tool install pre-commit`.
1. Then run `pre-commit install` in this project directory.
1. Update the `pre-commit` to use the latest versions of `ruff` and `black` with `pre-commit autoupdate` in this project directory.
1. That's it! You're good to begin development :)

## Run the application

1. Run: `uv run python main.py`
1. Open `http://<address of this computer>:8080/upload` in a browser to upload pictures.
1. The wearable (or a script) sends one picture at a time: `curl -X POST --data-binary @photo.jpg http://<address>:8080/api/images`

- The first start downloads the AI models (about 780 MB). It needs internet one time.
- To require an access token for the upload route, set the environment variable `SAARTHI_API_TOKEN`
  before the start. Every request then needs the header `Authorization: Bearer <token>`.
- Only one copy of the application may run on the same data.
- To add the objects that you want to find, open `http://<address>:8080/enroll`. Add a name, the kind of object (for example "a bottle"),
  and 5 to 10 photos. Check the crop of each photo. The object is ready when 3 photos are accepted.
  The same steps are available through `/api/objects` (see `docs/backend/Enrollment.md`).
- Press Ctrl+C to stop. The application finishes the picture that it works on and saves its data.
- The settings are in the `config.toml` file of each module (`backend/`, `db/`, `storage/`, `ai/`).
  The design documents are in `docs/` and in the `docs` folder of each module.

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
