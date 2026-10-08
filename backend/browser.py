"""The memory browser: what the pipeline stored for each frame.

`MemoryBrowser` lists the frames that the worker took, and gives the details of one frame:
the registered-object regions, the sightings, the boxes of the broad detector and the room scores.
It also cuts a picture of a frame (a small one for the list, a large one for the details).
It only reads. The frames are private pictures: the page that shows them must not be open to other people.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import db.api as db
import storage.api as storage
from backend.errors import InvalidInputError, NotFoundError
from db.browse import MAX_PAGE, FrameDetail, FrameRow
from db.schema import ImageStatus, utc_now

logger: logging.Logger = logging.getLogger(__name__)

STATUSES: tuple[str, ...] = tuple(s.value for s in ImageStatus)
MAX_PICTURE_SIDE: int = 2048
# The room scores that the details show: the best ones only.
TOP_ROOMS: int = 3

MESSAGE_STATUS: str = "The status is not known."
MESSAGE_PAGE: str = "The page is not valid."
MESSAGE_FRAME: str = "The frame does not exist."


@dataclass(frozen=True)
class FrameSummary:
    """A frame in the list."""

    id: int
    status: str
    status_detail: str | None  # For example the reason of a rejection
    captured_at: datetime  # Naive UTC time at which the server received the frame
    age_seconds: int
    width: int
    height: int
    blur_score: float | None
    regions: int
    sightings: int
    open_boxes: int


@dataclass(frozen=True)
class FramePage:
    """One page of the list."""

    frames: list[FrameSummary]
    total: int  # The number of frames with this filter
    offset: int
    limit: int
    counts: dict[str, int]  # The number of frames for each status


@dataclass(frozen=True)
class RegionView:
    id: int
    box: tuple[int, int, int, int]
    phrases: list[str]  # The words that the detector linked to the box
    confidence: float
    source: str  # full, tile or generic
    in_hand: bool
    quality_ok: bool  # False: the crop is small, blurred or cut by the edge, so it is never matched


@dataclass(frozen=True)
class SightingView:
    object_name: str
    region_id: int
    similarity: float
    margin: float
    strong: bool
    in_hand: bool


@dataclass(frozen=True)
class OpenBoxView:
    id: int
    box: tuple[int, int, int, int]
    label: str  # The name that the broad detector gave the box. It can be wrong
    confidence: float
    linked_region_id: int | None  # The registered-object region that this box overlaps


@dataclass(frozen=True)
class FrameDetailView:
    summary: FrameSummary
    processed_at: datetime | None
    regions: list[RegionView]
    sightings: list[SightingView]
    open_boxes: list[OpenBoxView]
    rooms: list[
        tuple[str, float]
    ]  # The best rooms with their probability. Empty if not scored
    last_seen_of: list[str]  # The objects whose last-seen record is this frame
    descriptions: list[
        tuple[str, str]
    ]  # The checked sentences about the place of an object


def _summary(row: FrameRow, now: datetime) -> FrameSummary:
    image = row.image
    return FrameSummary(
        id=image.id,
        status=image.status,
        status_detail=image.status_detail,
        captured_at=image.captured_at,
        age_seconds=max(0, int((now - image.captured_at).total_seconds())),
        width=image.width,
        height=image.height,
        blur_score=image.blur_score,
        regions=row.regions,
        sightings=row.sightings,
        open_boxes=row.open_boxes,
    )


class MemoryBrowser:
    """Reads the stored frames. It is safe to call from several threads."""

    def list_frames(
        self, status: str | None = None, offset: int = 0, limit: int = 24
    ) -> FramePage:
        """List the frames, the newest first. `status` None lists all of them."""
        if status is not None and status not in STATUSES:
            raise InvalidInputError(MESSAGE_STATUS)
        if not 0 < limit <= MAX_PAGE or offset < 0:
            raise InvalidInputError(MESSAGE_PAGE)
        counts: dict[str, int] = db.browse.count_by_status()
        total: int = sum(counts.values()) if status is None else counts[status]
        now: datetime = utc_now()
        rows: list[FrameRow] = db.browse.list_frames(status, limit, offset)
        return FramePage([_summary(r, now) for r in rows], total, offset, limit, counts)

    def get_frame(self, image_id: int) -> FrameDetailView:
        """Return what the pipeline stored for a frame. Raise `NotFoundError` if it does not exist."""
        assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
        detail: FrameDetail | None = db.browse.get_frame_detail(image_id)
        if detail is None:
            raise NotFoundError(MESSAGE_FRAME)
        image = detail.image
        counts_row = FrameRow(
            image, len(detail.regions), len(detail.sightings), len(detail.open_regions)
        )
        rooms: list[tuple[str, float]] = sorted(
            (detail.rooms or {}).items(), key=lambda item: -item[1]
        )[:TOP_ROOMS]
        return FrameDetailView(
            summary=_summary(counts_row, utc_now()),
            processed_at=image.processed_at,
            regions=[
                RegionView(
                    r.id,
                    (r.x0, r.y0, r.x1, r.y1),
                    list(r.phrases),
                    r.det_score,
                    r.source,
                    bool(r.in_hand),
                    bool(r.quality_ok),
                )
                for r in detail.regions
            ],
            sightings=[
                SightingView(
                    s.object_name,
                    s.sighting.region_id,
                    s.sighting.similarity,
                    s.sighting.margin,
                    bool(s.sighting.strong),
                    bool(s.sighting.in_hand),
                )
                for s in detail.sightings
            ],
            open_boxes=[
                OpenBoxView(
                    o.id,
                    (o.x0, o.y0, o.x1, o.y1),
                    o.label,
                    o.confidence,
                    o.region_id,
                )
                for o in detail.open_regions
            ],
            rooms=rooms,
            last_seen_of=detail.last_seen_of,
            descriptions=detail.descriptions,
        )

    def get_picture(self, image_id: int, max_side: int = 512) -> bytes:
        """Return a JPEG of the whole frame. Raise `NotFoundError` if the frame or its file is gone."""
        assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
        assert (
            isinstance(max_side, int) and 0 < max_side <= MAX_PICTURE_SIDE
        ), "max_side out of range"
        image = db.get_image(image_id)
        if image is None:
            raise NotFoundError(MESSAGE_FRAME)
        try:
            return storage.read_crop(
                image.file_path, (0, 0, image.width, image.height), max_side
            )
        except storage.StorageError as error:
            logger.warning("The file of frame %d cannot be read", image_id)
            raise NotFoundError(MESSAGE_FRAME) from error
