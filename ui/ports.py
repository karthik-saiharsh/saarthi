"""Interface between the UI and the rest of the application.

The UI imports only this module to reach the backend. The backend gives
the real implementations of `UploadService` and `EnrollmentService`.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class OutcomeStatus(StrEnum):
    """Result of one upload."""

    STORED = "stored"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class UploadOutcome:
    """Result of one upload, with a short text for the user."""

    status: OutcomeStatus
    # Must not contain personal data or internal error text.
    message: str


class UploadService(Protocol):
    """Accepts one image at a time."""

    async def submit_image(
        self, filename: str, content_type: str, data: bytes
    ) -> UploadOutcome:
        """Store one image. Return the outcome. Do not raise for expected cases."""
        ...


class PhotoState(StrEnum):
    """State of a reference photo of an object."""

    PENDING = "pending"  # Saved, the server has not looked at it yet
    PROCESSING = "processing"  # The server looks for the object
    LOCATED = "located"  # The object was found. The carer must check the crop
    CONFIRMED = "confirmed"  # The carer accepted the crop
    FAILED = "failed"  # The object was not found, or an error occurred


class EnrollmentError(Exception):
    """An expected problem. `str(error)` is a short text that the UI shows.

    The text must not contain personal data or internal error text.
    """


@dataclass(frozen=True)
class PhotoInfo:
    """One reference photo."""

    id: int
    state: PhotoState
    # The reason of a failure, when the state is `failed`.
    detail: str | None


@dataclass(frozen=True)
class ObjectInfo:
    """One enrolled object."""

    id: int
    name: str
    category_phrase: str
    synonyms: tuple[str, ...]
    # False: the object is a look-alike and is never reported.
    tracked: bool
    ready: bool
    # What is missing, or a warning. None when there is nothing to tell.
    note: str | None
    photo_counts: dict[PhotoState, int]


@dataclass(frozen=True)
class ObjectDetail:
    """An object with its photos, oldest first."""

    info: ObjectInfo
    photos: tuple[PhotoInfo, ...]


@dataclass(frozen=True)
class ReviewItem:
    """A box of a recent frame that may show the open object: "Is this your object?"."""

    region_id: int
    age_seconds: int  # The time since the frame was received


class EnrollmentService(Protocol):
    """The gallery: objects and their reference photos.

    Every method raises `EnrollmentError` for an expected problem, for example
    a name that exists. Other errors are bugs.
    """

    async def create_object(
        self,
        name: str,
        category_phrase: str,
        synonyms: tuple[str, ...],
        tracked: bool,
    ) -> ObjectInfo: ...

    async def list_objects(self) -> list[ObjectInfo]: ...

    async def get_object(self, object_id: int) -> ObjectDetail: ...

    async def delete_object(self, object_id: int) -> None: ...

    async def add_photo(self, object_id: int, data: bytes) -> PhotoInfo: ...

    async def confirm_photo(self, object_id: int, photo_id: int) -> None: ...

    async def remove_photo(self, object_id: int, photo_id: int) -> None: ...

    async def retry_photo(self, object_id: int, photo_id: int) -> None: ...

    async def get_crop(self, object_id: int, photo_id: int, max_side: int) -> bytes:
        """Return a JPEG of the part of the photo where the object was found."""
        ...

    async def list_review(self, object_id: int) -> list[ReviewItem]:
        """Return the boxes of recent frames that may show the object. Empty if the object is not ready."""
        ...

    async def get_review_picture(self, region_id: int, max_side: int) -> bytes:
        """Return a JPEG of a box to check, with some space around it."""
        ...

    async def answer_review(self, object_id: int, region_id: int, yes: bool) -> None:
        """Keep the answer of the carer. "Yes" adds a photo of the object."""
        ...


class BrowseError(Exception):
    """An expected problem of the memory browser. `str(error)` is a short text that the UI shows.

    The text must not contain personal data or internal error text.
    """


@dataclass(frozen=True)
class FrameInfo:
    """A frame that the worker took, ready to show in the list."""

    id: int
    status: str  # pending, processing, done, rejected or failed
    detail: str | None  # For example the reason of a rejection
    age_seconds: int  # The time since the server received the frame
    width: int
    height: int
    regions: int  # The boxes of the registered objects
    sightings: int  # The matches of enrolled objects
    open_boxes: int  # The boxes of the broad detector


@dataclass(frozen=True)
class FramesPage:
    """One page of frames."""

    frames: list[FrameInfo]
    total: int  # The number of frames with the filter
    offset: int
    limit: int
    counts: dict[str, int]  # The number of frames for each status


@dataclass(frozen=True)
class RegionInfo:
    """A box of the registered-object detector."""

    id: int
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels of the frame
    phrases: tuple[str, ...]
    confidence: float
    source: str
    in_hand: bool
    quality_ok: bool  # False: the crop is not used for matching


@dataclass(frozen=True)
class SightingInfo:
    """A match of an enrolled object."""

    object_name: str
    region_id: int
    similarity: float
    margin: float
    strong: bool
    in_hand: bool


@dataclass(frozen=True)
class OpenBoxInfo:
    """A box of the broad detector."""

    id: int
    box: tuple[int, int, int, int]
    label: str  # The name from the detector. It can be wrong
    confidence: float
    linked_region_id: int | None


@dataclass(frozen=True)
class FrameDetail:
    """What the pipeline stored for one frame."""

    frame: FrameInfo
    blur_score: float | None
    regions: tuple[RegionInfo, ...]
    sightings: tuple[SightingInfo, ...]
    open_boxes: tuple[OpenBoxInfo, ...]
    rooms: tuple[tuple[str, float], ...]  # The best rooms with their probability
    last_seen_of: tuple[str, ...]
    descriptions: tuple[
        tuple[str, str], ...
    ] = ()  # (object name, sentence about its place)


class BrowseService(Protocol):
    """The memory browser. Every method raises `BrowseError` for an expected problem."""

    async def list_frames(
        self, status: str | None, offset: int, limit: int
    ) -> FramesPage:
        """Return one page of frames, the newest first. `status` None lists all."""
        ...

    async def get_frame(self, image_id: int) -> FrameDetail:
        """Return what the pipeline stored for a frame."""
        ...

    async def get_picture(self, image_id: int, max_side: int) -> bytes:
        """Return a JPEG of the whole frame, at most `max_side` pixels on the longest side."""
        ...


class VoiceError(Exception):
    """An expected problem of the voice input. `str(error)` is a short text that the UI shows.

    The text must not contain personal data or internal error text.
    """


class VoiceService(Protocol):
    """The voice input: a recording becomes the words of the user."""

    max_bytes: int  # The largest recording that the service accepts
    max_seconds: int  # The longest recording, in seconds

    async def transcribe(self, wav: bytes) -> str:
        """Return the words in a WAV recording (16-bit, mono, 16 kHz). Raise `VoiceError` on a problem.

        The text is empty if the service hears no speech.
        """
        ...


class FindStatus(StrEnum):
    """What the page shows after a search."""

    FOUND = "found"  # The system saw the object. The page shows where and when
    NOT_SEEN = "not_seen"  # The object is known, but the system did not see it
    CHOOSE = "choose"  # The words fit several objects. The user chooses
    NO_MATCH = "no_match"  # The words fit no object


class FindError(Exception):
    """An expected problem of a search. `str(error)` is a short text that the UI shows.

    The text must not contain personal data or internal error text.
    """


@dataclass(frozen=True)
class FindChoice:
    """An object that the user can look for."""

    object_id: int
    name: str


@dataclass(frozen=True)
class Sighting:
    """Where and when the system saw an object, ready to show."""

    age_seconds: int  # The time since the frame was received
    in_hand: bool  # The person held the object
    strong: bool  # False: the system is not sure. The page says "maybe"
    stale: bool  # An old sighting, and nothing newer was found
    picture: bytes | None  # A JPEG of the object with some space around it


@dataclass(frozen=True)
class FindResult:
    """The result of one search."""

    status: FindStatus
    # The name of the object, for FOUND and NOT_SEEN.
    object_name: str | None = None
    sighting: Sighting | None = None
    # The objects to choose from, for CHOOSE. Best first.
    choices: tuple[FindChoice, ...] = ()
    # False: the search stopped early, so "not seen" may be wrong.
    complete: bool = True
    # The sentence that describes the answer, for FOUND. The place can be left out of it.
    description: str | None = None
    # A newer weak sighting in another place, for FOUND: "maybe it is here now". None if there is none.
    maybe: Sighting | None = None


class FindService(Protocol):
    """The search: choose an object, or write what you look for.

    Every method raises `FindError` for an expected problem.
    """

    async def list_choices(self) -> list[FindChoice]:
        """Return the objects that a search can find, in a stable order."""
        ...

    async def ask(self, text: str) -> FindResult:
        """Find the object that the words mean, and where it was seen."""
        ...

    async def find(self, object_id: int) -> FindResult:
        """Find where an object was seen last."""
        ...

    async def ask_open(self, text: str) -> FindResult:
        """Look for an object that nobody registered, with the words of the user.

        The page calls it when the user taps "Something else" under the choices.
        The result is `found` (a weak answer, with its picture) or `no_match`.
        """
        ...
