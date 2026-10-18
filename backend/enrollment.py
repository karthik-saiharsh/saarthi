"""Enrollment: the gallery of the objects that the owner wants to find (stages E1 to E4).

A carer creates an object with a name and a category phrase, adds 5 to 10
photos, and confirms the crop of each photo. In between, the worker finds the
object in each photo and makes its views (see `backend/worker.py`).

`EnrollmentService` holds the steps that need more than one component:
storage (the photo files), db (the rows), and ai (the calibration). All methods
block. Call them from a thread, not from the event loop.

What a carer types goes into the prompt of the detector. The words are
checked: only lowercase letters, digits, spaces, apostrophes and hyphens.
Names are never written to the log. The log shows ids and counts.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import db.api as db
import storage.api as storage
from ai.api import AiEngine, AiError
from backend.errors import InvalidInputError, LimitReachedError, NotFoundError
from backend.references import ReferenceVectors
from backend.settings import BackendConfig, EnrollmentConfig
from db.schema import EnrolledObject, PhotoStatus, ReferencePhoto

logger: logging.Logger = logging.getLogger(__name__)

MAX_NAME_LENGTH: int = 80
MAX_PHRASE_LENGTH: int = 40
MAX_PHRASE_WORDS: int = 6
MAX_SYNONYMS: int = 5

# A phrase for the detector. No period: a period ends a phrase in the prompt.
_WORD: str = r"[a-z0-9]+(?:['-][a-z0-9]+)*"
_PHRASE: re.Pattern[str] = re.compile(rf"{_WORD}(?: {_WORD})*")

MESSAGE_NAME: str = f"The name must have 1 to {MAX_NAME_LENGTH} printable characters."
MESSAGE_PHRASE: str = (
    f"A phrase can have up to {MAX_PHRASE_WORDS} words and {MAX_PHRASE_LENGTH} characters. "
    "Use lowercase letters, digits, spaces, apostrophes and hyphens."
)
MESSAGE_SYNONYMS: str = f"At most {MAX_SYNONYMS} synonyms are allowed."


@dataclass(frozen=True)
class PhotoInfo:
    """A reference photo, as the carer sees it."""

    id: int
    status: str  # pending, processing, located, confirmed or failed
    detail: str | None  # The reason of a failure
    box: tuple[int, int, int, int] | None  # Where the object is, when it is located
    det_score: float | None


@dataclass(frozen=True)
class ObjectSummary:
    """An enrolled object, as the carer sees it."""

    id: int
    name: str
    category_phrase: str
    synonyms: list[str]
    tracked: bool
    status: str  # draft or ready
    tau: float | None
    tau_strong: float | None
    note: str | None  # What is missing, or a warning
    small: bool  # The detector also looks in 4 tiles of each frame
    photo_counts: dict[str, int]  # Photos for each status


@dataclass(frozen=True)
class ObjectDetail:
    summary: ObjectSummary
    photos: list[PhotoInfo]


def clean_name(text: str) -> str:
    """Return the name without extra spaces. Raise `InvalidInputError` if it is not valid."""
    assert isinstance(text, str), "name must be a string"
    name: str = " ".join(text.split())
    if not name or len(name) > MAX_NAME_LENGTH or not name.isprintable():
        raise InvalidInputError(MESSAGE_NAME)
    return name


def clean_phrase(text: str) -> str:
    """Return a phrase for the detector. Raise `InvalidInputError` if it is not valid."""
    assert isinstance(text, str), "phrase must be a string"
    phrase: str = " ".join(text.lower().split())
    if (
        not phrase
        or len(phrase) > MAX_PHRASE_LENGTH
        or len(phrase.split()) > MAX_PHRASE_WORDS
        or not _PHRASE.fullmatch(phrase)
    ):
        raise InvalidInputError(MESSAGE_PHRASE)
    return phrase


class EnrollmentService:
    """The steps of the carer, from a new object to a calibrated object."""

    def __init__(
        self,
        config: BackendConfig,
        engine: AiEngine,
        on_photo_added: Callable[[], None] | None = None,
        references: ReferenceVectors | None = None,
    ) -> None:
        """`on_photo_added` runs after a photo is saved. The worker uses it to wake up.

        `references` gives the backgrounds of the photos for the self-test (plan part D1). Without it, there is no test.
        """
        assert isinstance(config, BackendConfig), "config must be a BackendConfig"
        assert isinstance(engine, AiEngine), "engine must be an AiEngine"
        assert on_photo_added is None or callable(
            on_photo_added
        ), "on_photo_added must be callable"
        self._config: EnrollmentConfig = config.enrollment
        self._engine: AiEngine = engine
        self._on_photo_added: Callable[[], None] | None = on_photo_added
        self._references: ReferenceVectors | None = references

    # ---- Objects ----

    def create_object(
        self,
        name: str,
        category_phrase: str,
        synonyms: list[str] | None = None,
        tracked: bool = True,
    ) -> ObjectSummary:
        """Add an object. Raise `InvalidInputError`, `LimitReachedError` or `db.DuplicateObjectError`."""
        assert synonyms is None or isinstance(synonyms, list), "synonyms must be a list"
        assert isinstance(tracked, bool), "tracked must be a bool"
        clean: str = clean_name(name)
        category: str = clean_phrase(category_phrase)
        extra: list[str] = []
        for text in synonyms or []:
            phrase: str = clean_phrase(text)
            if phrase != category and phrase not in extra:
                extra.append(phrase)
        if len(extra) > MAX_SYNONYMS:
            raise InvalidInputError(MESSAGE_SYNONYMS)
        if db.gallery.count_objects() >= self._config.max_objects:
            raise LimitReachedError(
                f"The gallery has {self._config.max_objects} objects"
            )
        obj: EnrolledObject = db.gallery.create_object(clean, category, extra, tracked)
        self._recalibrate(obj.id)  # Sets the note "needs N photos"
        return self._summary(db.gallery.get_object(obj.id))  # type: ignore[arg-type]

    def list_objects(self) -> list[ObjectSummary]:
        return [self._summary(obj) for obj in db.gallery.list_objects()]

    def get_object(self, object_id: int) -> ObjectDetail:
        """Return an object with its photos. Raise `NotFoundError` if it does not exist."""
        obj: EnrolledObject = self._require_object(object_id)
        photos: list[PhotoInfo] = [
            self._photo_info(p) for p in db.gallery.list_photos(object_id)
        ]
        return ObjectDetail(self._summary(obj), photos)

    def set_small(self, object_id: int, small: bool) -> ObjectSummary:
        """Mark an object as small (keys, a ring): the detector then also looks in 4 tiles of each frame. It costs time."""
        assert isinstance(small, bool), "small must be a bool"
        self._require_object(object_id)
        return self._summary(db.gallery.set_small(object_id, small))

    def delete_object(self, object_id: int) -> None:
        """Delete an object, its photos and its files. The other objects are calibrated again."""
        self._require_object(object_id)
        paths: list[str] = db.gallery.delete_object(object_id)  # Rows first, files last
        storage.delete_images(paths)
        self.recalibrate_all()

    # ---- Photos ----

    def add_photo(self, object_id: int, data: bytes) -> PhotoInfo:
        """Check, store and register a reference photo. The worker analyses it later.

        Raise `NotFoundError`, `LimitReachedError`, the storage errors for a bad
        image, or `db.DuplicateImageError` if the object has this picture already.
        If the row cannot be added, the file is deleted again.
        """
        assert isinstance(data, bytes), "data must be bytes"
        self._require_object(object_id)
        if len(db.gallery.list_photos(object_id)) >= self._config.max_photos_per_object:
            raise LimitReachedError(
                f"An object can have {self._config.max_photos_per_object} photos"
            )
        prepared: storage.PreparedImage = storage.prepare_image(data)
        path: str = storage.save_image(prepared)
        try:
            photo: ReferencePhoto = db.gallery.add_reference_photo(
                object_id, path, prepared.sha256, prepared.width, prepared.height
            )
        except db.DbError:
            storage.delete_images([path])
            raise
        logger.info("Added reference photo %d to object %d", photo.id, object_id)
        if self._on_photo_added is not None:
            try:
                self._on_photo_added()
            except Exception:  # A failing hook must not fail a photo that is stored.
                logger.exception("The on_photo_added hook failed")
        return self._photo_info(photo)

    def confirm_photo(self, object_id: int, photo_id: int) -> PhotoInfo:
        """The carer accepts the crop. The object is calibrated again."""
        self._require_photo(object_id, photo_id)
        photo: ReferencePhoto = db.gallery.confirm_photo(photo_id)
        self.recalibrate_all()
        return self._photo_info(photo)

    def remove_photo(self, object_id: int, photo_id: int) -> None:
        """The carer removes a photo (a wrong crop). The object is calibrated again."""
        self._require_photo(object_id, photo_id)
        path: str | None = db.gallery.remove_photo(photo_id)
        if path is not None:
            storage.delete_images([path])
        self.recalibrate_all()

    def retry_photo(self, object_id: int, photo_id: int) -> PhotoInfo:
        """Try a failed photo again."""
        self._require_photo(object_id, photo_id)
        photo: ReferencePhoto = db.gallery.retry_photo(photo_id)
        if self._on_photo_added is not None:
            self._on_photo_added()
        return self._photo_info(photo)

    def get_crop(self, object_id: int, photo_id: int, max_side: int = 512) -> bytes:
        """Return a JPEG of the crop where the object was found, for the confirmation screen."""
        photo: ReferencePhoto = self._require_photo(object_id, photo_id)
        located: bool = photo.status in (
            PhotoStatus.LOCATED.value,
            PhotoStatus.CONFIRMED.value,
        )
        if not located or photo.x0 is None:
            raise NotFoundError("The object is not located in this photo yet")
        return storage.read_crop(
            photo.file_path, (photo.x0, photo.y0, photo.x1, photo.y1), max_side
        )

    # ---- Calibration (E4) ----

    def recalibrate_all(self) -> None:
        """Calibrate every object again. The negatives of an object change when another object changes."""
        for obj in db.gallery.list_objects():
            self._recalibrate(obj.id)

    def _recalibrate(self, object_id: int) -> None:
        """An object with enough confirmed photos is `ready`. Another object is `draft`."""
        model: str = self._engine.embedding_model
        views, photo_ids = db.gallery.load_views(object_id, model)
        needed: int = self._engine.min_photos - len(set(photo_ids))
        if needed > 0:
            db.gallery.clear_calibration(
                object_id, f"Needs {needed} more confirmed photo(s)."
            )
            return
        try:
            result = self._engine.calibrate_object(
                views, photo_ids, db.gallery.load_other_views(object_id, model)
            )
        except AiError:
            logger.exception("The calibration of object %d failed", object_id)
            db.gallery.clear_calibration(object_id, "The calibration failed.")
            return
        notes: list[str] = [n for n in (result.note, self._place_note(object_id)) if n]
        db.gallery.set_calibration(
            object_id,
            result.tau,
            result.tau_strong,
            result.margin_min,
            model,
            " ".join(notes) or None,
        )
        logger.info(
            "Object %d is ready: %d views of %d photos, tau %.3f",
            object_id,
            result.n_views,
            result.n_photos,
            result.tau,
        )

    def _place_note(self, object_id: int) -> str | None:
        """The self-test (plan part D1): a warning if all confirmed photos show one place. An error gives no warning."""
        if self._references is None:
            return None
        try:
            found = self._references.get(object_id)
        except (AiError, db.DbError):
            logger.warning("The self-test of object %d could not run", object_id)
            return None
        return None if found is None else self._engine.place_note(found.place_vectors)

    # ---- Helpers ----

    def _require_object(self, object_id: int) -> EnrolledObject:
        assert (
            isinstance(object_id, int) and object_id > 0
        ), "object_id must be positive"
        obj: EnrolledObject | None = db.gallery.get_object(object_id)
        if obj is None:
            raise NotFoundError("The object does not exist")
        return obj

    def _require_photo(self, object_id: int, photo_id: int) -> ReferencePhoto:
        """Return a photo of this object. A photo of another object is not found."""
        self._require_object(object_id)
        assert isinstance(photo_id, int) and photo_id > 0, "photo_id must be positive"
        photo: ReferencePhoto | None = db.gallery.get_photo(photo_id)
        if photo is None or photo.enrolled_object_id != object_id:
            raise NotFoundError("The photo does not exist")
        return photo

    @staticmethod
    def _photo_info(photo: ReferencePhoto) -> PhotoInfo:
        box: tuple[int, int, int, int] | None = (
            None if photo.x0 is None else (photo.x0, photo.y0, photo.x1, photo.y1)
        )
        return PhotoInfo(
            photo.id, photo.status, photo.status_detail, box, photo.det_score
        )

    @staticmethod
    def _summary(obj: EnrolledObject) -> ObjectSummary:
        counts: dict[PhotoStatus, int] = db.gallery.count_photos_by_status(obj.id)
        return ObjectSummary(
            id=obj.id,
            name=obj.name,
            category_phrase=obj.category_phrase,
            synonyms=list(obj.synonyms),
            tracked=obj.tracked,
            status=obj.status,
            tau=obj.tau,
            tau_strong=obj.tau_strong,
            note=obj.calibration_note,
            small=bool(obj.small),
            photo_counts={status.value: count for status, count in counts.items()},
        )
