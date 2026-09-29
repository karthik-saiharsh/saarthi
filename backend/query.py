"""From the words of a user to an enrolled object (the start of stage Q1).

The user can say anything: "my blue bottle", or "the blue thing I use to drink
stuff". The resolver compares the words with a description of each object that
can be searched (`ready` and not a look-alike). The description is the name, the
category phrase and the synonyms. The comparison uses the query encoder of the
ai component: a score is the cosine between two unit vectors.

There are three results:

- `matched`: one object clearly fits. The best score is high, and well above the second best.
- `ambiguous`: no object is sure. The caller shows the best objects, and the user chooses.
- `none`: no object fits the words.

The words of the user are private. They are never written to the log.
"""

import logging
import threading
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

import db.api as db
from ai.api import AiEngine
from backend.errors import InvalidQueryError
from backend.settings import BackendConfig, QueryConfig
from db.schema import EnrolledObject, ObjectStatus

logger: logging.Logger = logging.getLogger(__name__)

MESSAGE_QUERY: str = "Write what you are looking for, in a few words."


class QueryStatus(StrEnum):
    """How sure the resolver is."""

    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    NONE = "none"


@dataclass(frozen=True)
class QueryCandidate:
    """An object that fits the words, with its score (the cosine, from -1 to 1)."""

    object_id: int
    name: str
    score: float


@dataclass(frozen=True)
class QueryResult:
    """The candidates, best first. `matched` has one, `none` has none."""

    status: QueryStatus
    candidates: tuple[QueryCandidate, ...]


def clean_query(text: str, max_chars: int) -> str:
    """Return the words without extra spaces. Raise `InvalidQueryError` if they are not usable."""
    assert isinstance(text, str), "text must be a string"
    assert max_chars > 0, "max_chars must be positive"
    words: str = " ".join(text.split())
    if not words or len(words) > max_chars or not words.isprintable():
        raise InvalidQueryError(MESSAGE_QUERY)
    return words


def describe(obj: EnrolledObject) -> str:
    """Return the text that stands for an object: its name, its category and its synonyms."""
    return f"{obj.name}: {', '.join([obj.category_phrase, *obj.synonyms])}"


class QueryResolver:
    """Finds the object that the words of a user mean. It is safe to call from several threads."""

    def __init__(self, config: BackendConfig, engine: AiEngine) -> None:
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        self._config: QueryConfig = config.query
        self._engine: AiEngine = engine
        # The vector of each description. The gallery is small and changes rarely.
        self._cache: dict[str, np.ndarray] = {}
        self._lock: threading.Lock = threading.Lock()

    def resolve(self, text: str) -> QueryResult:
        """Find the object for the words. Raise `InvalidQueryError` for unusable words."""
        words: str = clean_query(text, self._config.max_chars)
        objects: list[EnrolledObject] = [
            o
            for o in db.gallery.list_objects()
            if o.status == ObjectStatus.READY.value and o.tracked
        ]
        if not objects:
            logger.info("Query: no object can be searched")
            return QueryResult(QueryStatus.NONE, ())

        descriptions: np.ndarray = self._description_vectors(objects)
        question: np.ndarray = self._engine.encode_texts([words])[0]
        scores: np.ndarray = descriptions @ question
        order: list[int] = sorted(range(len(objects)), key=lambda i: -float(scores[i]))
        ranked: list[QueryCandidate] = [
            QueryCandidate(objects[i].id, objects[i].name, float(scores[i]))
            for i in order
        ]

        best: float = ranked[0].score
        second: float = ranked[1].score if len(ranked) > 1 else -1.0
        cfg: QueryConfig = self._config
        if best >= cfg.matched_min_score and best - second >= cfg.matched_min_gap:
            result: QueryResult = QueryResult(QueryStatus.MATCHED, (ranked[0],))
        elif best >= cfg.suggest_min_score:
            fitting: list[QueryCandidate] = [
                c for c in ranked if c.score >= cfg.suggest_min_score
            ]
            result = QueryResult(
                QueryStatus.AMBIGUOUS, tuple(fitting[: cfg.max_candidates])
            )
        else:
            result = QueryResult(QueryStatus.NONE, ())
        logger.info(
            "Query: %s with %d candidate(s) from %d object(s)",
            result.status.value,
            len(result.candidates),
            len(objects),
        )
        return result

    def _description_vectors(self, objects: list[EnrolledObject]) -> np.ndarray:
        """Return one vector for each object. A description is encoded once."""
        texts: list[str] = [describe(o) for o in objects]
        with self._lock:
            missing: list[str] = sorted({t for t in texts if t not in self._cache})
            if missing:
                for text, vector in zip(missing, self._engine.encode_texts(missing)):
                    self._cache[text] = vector
            # Forget the descriptions of objects that are gone or changed.
            self._cache = {t: self._cache[t] for t in set(texts)}
            return np.stack([self._cache[t] for t in texts])
