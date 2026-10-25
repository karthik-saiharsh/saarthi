"""The question of the wearable: a recording becomes a sentence to say (stages Q0 to Q2, for a device with no screen).

The wearable sends a WAV recording and gets an id at once. One background thread makes the answer:
the recording becomes text (Q0), the text becomes an object (the resolver), the search finds the place (Q1),
and the place becomes a sentence (Q2). The wearable asks for the answer with the id until it is done.

The answer is always a sentence that the wearable can say, also for an error. If the voice is on (`[tts]` of
`ai/config.toml`), the same thread also makes the WAV of the sentence (stage Q3) before the job is done. The sentence never asks the user to
look at a picture. If the words fit several objects, the sentence names them and asks the user to press the button
again and say the name. The next press is a new question: the server keeps no dialogue.

Privacy: the recording stays in memory and is dropped after the text is made. The text and the answer are never
logged. A job is deleted `talk.keep_seconds` after the question came.
"""

import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum

from backend.describe import DescribeService
from backend.errors import (
    InvalidAudioError,
    InvalidQueryError,
    QueueFullError,
    VoiceUnavailableError,
)
from backend.open_search import OpenAnswer, OpenSearchService
from backend.query import QueryResolver, QueryResult, QueryStatus
from backend.search import Answer, SearchService
from backend.settings import BackendConfig, TalkConfig
from backend.voice import VoiceService

logger: logging.Logger = logging.getLogger(__name__)

# The fixed sentences. They contain no word of the user.
NOT_HEARD_TEXT: str = (
    "I did not hear a question. Please press the button and ask again."
)
UNKNOWN_TEXT: str = (
    "I do not know which object you mean. Please press the button and say its name."
)
FAILED_TEXT: str = "Something went wrong. Please try again later."
OFF_TEXT: str = "Listening is switched off."
CHOOSE_END_TEXT: str = "Please press the button and say the name."
MESSAGE_BUSY: str = "Too many questions wait for an answer."

# A job id has 16 random bytes, written in URL-safe base 64 (22 characters).
ID_BYTES: int = 16


class TalkStatus(StrEnum):
    """The state of a job."""

    PENDING = "pending"
    DONE = "done"


class TalkOutcome(StrEnum):
    """What the answer says. The wearable can use it, for example for a light."""

    FOUND = "found"  # A sure sighting
    MAYBE = "maybe"  # A weak sighting, or an object that nobody registered
    NOT_SEEN = "not_seen"  # The object is known, but it was not seen lately
    CHOOSE = "choose"  # The words fit several objects. The sentence names them
    UNKNOWN = "unknown"  # No object fits the words
    NOT_HEARD = "not_heard"  # Silence, or a recording that is not usable
    FAILED = "failed"  # An error of the server


@dataclass(frozen=True)
class TalkJob:
    """One question. `text` and `outcome` are None while the job is pending."""

    id: str
    created: float  # time.monotonic() when the question came
    status: TalkStatus
    outcome: TalkOutcome | None = None
    text: str | None = None
    detail: str | None = None  # A fixed message for a recording that is not usable
    speech: bytes | None = field(
        default=None, repr=False
    )  # The WAV of `text`, or None if the voice is off or failed


def choose_text(names: list[str]) -> str:
    """Return the question for several objects, for example "Do you mean your watch, or your headset?"."""
    assert len(names) >= 2, "a choice needs two or more names"
    mine: list[str] = [f"your {n.strip()}" for n in names]
    return f"Do you mean {', '.join(mine[:-1])}, or {mine[-1]}? {CHOOSE_END_TEXT}"


class TalkService:
    """Keeps the jobs of the wearable and makes their answers on one background thread. It is safe to call from several threads."""

    def __init__(
        self,
        config: BackendConfig,
        voice: VoiceService,
        resolver: QueryResolver,
        search: SearchService,
        describer: DescribeService,
        open_search: OpenSearchService | None = None,
    ) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(voice, VoiceService), "voice must be a VoiceService"
        assert isinstance(resolver, QueryResolver), "resolver must be a QueryResolver"
        assert isinstance(search, SearchService), "search must be a SearchService"
        assert isinstance(
            describer, DescribeService
        ), "describer must be a DescribeService"
        self._config: TalkConfig = config.talk
        self._voice: VoiceService = voice
        self._resolver: QueryResolver = resolver
        self._search: SearchService = search
        self._describer: DescribeService = describer
        self._open_search: OpenSearchService | None = open_search
        self._jobs: dict[str, TalkJob] = {}
        self._lock: threading.Lock = threading.Lock()
        # One thread: the models are shared with the worker, so two answers at one time only make both slower.
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="talk"
        )

    @property
    def enabled(self) -> bool:
        """True if the speech model is on. If False, `submit` must not be called."""
        return self._voice.enabled

    @property
    def max_bytes(self) -> int:
        """The largest recording that `submit` accepts."""
        return self._voice.max_bytes

    def submit(self, wav: bytes) -> TalkJob:
        """Keep a new question and start its answer. Raise `QueueFullError` if `talk.max_jobs` questions wait."""
        assert isinstance(wav, bytes) and wav, "wav must not be empty"
        assert self.enabled, "listening is switched off"
        job: TalkJob = TalkJob(
            secrets.token_urlsafe(ID_BYTES), time.monotonic(), TalkStatus.PENDING
        )
        with self._lock:
            self._prune()
            waiting: int = sum(
                j.status == TalkStatus.PENDING for j in self._jobs.values()
            )
            if waiting >= self._config.max_jobs:
                logger.warning("Talk: a question was refused, %d wait", waiting)
                raise QueueFullError(MESSAGE_BUSY)
            self._jobs[job.id] = job
        try:
            self._executor.submit(self._run, job.id, wav)
        except (
            RuntimeError
        ):  # The thread was stopped (shutdown): keep no job that never runs
            with self._lock:
                self._jobs.pop(job.id, None)
            raise
        logger.info("Talk: question %s… was received", job.id[:6])
        return job

    def get(self, job_id: str) -> TalkJob | None:
        """Return the job, or None if it does not exist or was deleted."""
        assert isinstance(job_id, str), "job_id must be a string"
        with self._lock:
            self._prune()
            return self._jobs.get(job_id)

    def close(self) -> None:
        """Stop the background thread. A job that has not started is dropped."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            self._jobs.clear()

    def answer(self, text: str) -> tuple[TalkOutcome, str]:
        """Return the outcome and the sentence for the words of the user."""
        assert isinstance(text, str), "text must be a string"
        if not text.strip():
            return TalkOutcome.NOT_HEARD, NOT_HEARD_TEXT
        try:
            result: QueryResult = self._resolver.resolve(text)
        except InvalidQueryError:
            return TalkOutcome.UNKNOWN, UNKNOWN_TEXT
        if result.status == QueryStatus.MATCHED:
            best = result.candidates[0]
            return self._found(best.object_id, best.name)
        if result.status == QueryStatus.AMBIGUOUS:
            return TalkOutcome.CHOOSE, choose_text([c.name for c in result.candidates])
        return self._open(text)

    # ---- The steps ----

    def _found(self, object_id: int, name: str) -> tuple[TalkOutcome, str]:
        answer: Answer = self._search.find(object_id)
        text: str = self._describer.describe(name, answer, listening=True).text
        if not answer.found:
            return TalkOutcome.NOT_SEEN, text
        return (TalkOutcome.FOUND if answer.strong else TalkOutcome.MAYBE), text

    def _open(self, text: str) -> tuple[TalkOutcome, str]:
        """No registered object fits: look for what the broad detection saw. The result is always weak."""
        if self._open_search is None:
            return TalkOutcome.UNKNOWN, UNKNOWN_TEXT
        found: OpenAnswer | None = self._open_search.find(text)
        if found is None or found.answer.age_seconds is None:
            return TalkOutcome.UNKNOWN, UNKNOWN_TEXT
        sentence: str = self._describer.describe(
            found.name, found.answer, owned=False, listening=True
        ).text
        return TalkOutcome.MAYBE, sentence

    def _run(self, job_id: str, wav: bytes) -> None:
        """Make the answer of one job. Every error gives a sentence, so the wearable always has something to say."""
        started: float = time.monotonic()
        detail: str | None = None
        try:
            words: str = self._voice.transcribe(wav)
            del wav  # The recording is not needed after the text is made
            outcome, text = self.answer(words)
        except InvalidAudioError as error:
            outcome, text, detail = TalkOutcome.NOT_HEARD, NOT_HEARD_TEXT, str(error)
        except VoiceUnavailableError:
            outcome, text = TalkOutcome.FAILED, OFF_TEXT
        # The boundary of the thread: every error becomes a sentence.
        except Exception as error:  # noqa: BLE001
            logger.error("Talk: the answer failed: %s", type(error).__name__)
            outcome, text = TalkOutcome.FAILED, FAILED_TEXT
        speech: bytes | None = None
        try:
            speech = self._voice.speak(text)
        # The boundary of the thread: the text is still an answer without the audio.
        except Exception as error:  # noqa: BLE001
            logger.error("Talk: the speech failed: %s", type(error).__name__)
        with self._lock:
            job: TalkJob | None = self._jobs.get(job_id)
            if job is None:
                return  # Deleted in the meantime
            self._jobs[job_id] = replace(
                job,
                status=TalkStatus.DONE,
                outcome=outcome,
                text=text,
                detail=detail,
                speech=speech,
            )
        logger.info(
            "Talk: question %s… answered (%s) in %.1f s",
            job_id[:6],
            outcome.value,
            time.monotonic() - started,
        )

    def _prune(self) -> None:
        """Delete the jobs that are older than `talk.keep_seconds`, and all but the newest `talk.max_jobs` answers.

        Call it with the lock held.
        """
        oldest: float = time.monotonic() - self._config.keep_seconds
        for job_id in [k for k, j in self._jobs.items() if j.created < oldest]:
            del self._jobs[job_id]
        done: list[TalkJob] = sorted(
            (j for j in self._jobs.values() if j.status == TalkStatus.DONE),
            key=lambda j: j.created,
        )
        for job in done[: max(0, len(done) - self._config.max_jobs)]:
            del self._jobs[job.id]
