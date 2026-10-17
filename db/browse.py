"""Read-only views of the stored frames, for the memory browser.

The browser shows every frame that the worker took, with what the pipeline stored for it: the candidate regions
of the registered objects, the sightings, the boxes of the broad detector (regions with `source = broad`) and the
cached room scores.
These helpers read rows only. They change nothing and they read no file.
"""

import logging
from dataclasses import dataclass

from peewee import fn

from db.schema import (
    EnrolledObject,
    FrameDescription,
    FrameScene,
    Image,
    ImageStatus,
    LastSeen,
    Region,
    RegionSource,
    Sighting,
)
from db.support import translate_errors

logger: logging.Logger = logging.getLogger(__name__)

# The most frames that one call lists.
MAX_PAGE: int = 100


@dataclass(frozen=True)
class FrameRow:
    """A frame with the number of rows that the pipeline stored for it."""

    image: Image
    regions: int
    sightings: int
    open_boxes: int


@dataclass(frozen=True)
class SightingRow:
    """A match of an enrolled object in a region of the frame."""

    sighting: Sighting
    object_name: str


@dataclass(frozen=True)
class FrameDetail:
    """Everything that the pipeline stored for one frame."""

    image: Image
    regions: list[Region]  # The candidates of the detector of the registered objects
    sightings: list[SightingRow]
    open_regions: list[
        Region
    ]  # The boxes of the broad detector, the most confident first
    # The cached room scores, or None if no description needed this frame yet.
    rooms: dict[str, float] | None
    # The names of the objects whose last-seen record is this frame.
    last_seen_of: list[str]
    # The sentences about the place of an object (object name, text). A refused sentence is not listed.
    descriptions: list[tuple[str, str]]


def _check_status(status: str | None) -> None:
    assert status is None or status in {s.value for s in ImageStatus}, "unknown status"


def count_by_status() -> dict[str, int]:
    """Return the number of frames for each status. Every status is in the result."""
    with translate_errors("count_by_status"):
        counts: dict[str, int] = {s.value: 0 for s in ImageStatus}
        for status, number in (
            Image.select(Image.status, fn.COUNT(Image.id))
            .group_by(Image.status)
            .tuples()
        ):
            counts[status] = number
        return counts


def list_frames(status: str | None, limit: int, offset: int) -> list[FrameRow]:
    """Return the frames, the newest first. `status` None lists all frames."""
    _check_status(status)
    assert isinstance(limit, int) and 0 < limit <= MAX_PAGE, "limit out of range"
    assert isinstance(offset, int) and offset >= 0, "offset must not be negative"
    with translate_errors("list_frames"):
        query = Image.select().order_by(Image.captured_at.desc(), Image.id.desc())
        if status is not None:
            query = query.where(Image.status == status)
        images: list[Image] = list(query.limit(limit).offset(offset))
        ids: list[int] = [i.id for i in images]
        if not ids:
            return []

        def counts(model: type, column: object, *where: object) -> dict[int, int]:
            rows = (
                model.select(column, fn.COUNT(model.id))
                .where(column.in_(ids), *where)
                .group_by(column)
                .tuples()
            )
            return {image_id: number for image_id, number in rows}

        broad: str = RegionSource.BROAD.value
        regions = counts(Region, Region.image, Region.source != broad)
        sightings = counts(Sighting, Sighting.image)
        open_boxes = counts(Region, Region.image, Region.source == broad)
        return [
            FrameRow(
                i,
                regions.get(i.id, 0),
                sightings.get(i.id, 0),
                open_boxes.get(i.id, 0),
            )
            for i in images
        ]


def get_frame_detail(image_id: int) -> FrameDetail | None:
    """Return everything stored for a frame, or None if the frame does not exist."""
    assert isinstance(image_id, int) and image_id > 0, "image_id must be positive"
    with translate_errors("get_frame_detail"):
        image: Image | None = Image.get_or_none(Image.id == image_id)
        if image is None:
            return None
        broad: str = RegionSource.BROAD.value
        regions: list[Region] = list(
            Region.select()
            .where(Region.image == image_id, Region.source != broad)
            .order_by(Region.id)
        )
        sightings: list[SightingRow] = [
            SightingRow(row, row.enrolled_object.name)
            for row in Sighting.select(Sighting, EnrolledObject)
            .join(EnrolledObject)
            .where(Sighting.image == image_id)
            .order_by(Sighting.id)
        ]
        open_regions: list[Region] = list(
            Region.select()
            .where(Region.image == image_id, Region.source == broad)
            .order_by(Region.det_score.desc(), Region.id)
        )
        scene: FrameScene | None = FrameScene.get_or_none(FrameScene.image == image_id)
        last_seen_of: list[str] = [
            row.enrolled_object.name
            for row in LastSeen.select(LastSeen, EnrolledObject)
            .join(EnrolledObject)
            .where(LastSeen.image == image_id)
            .order_by(EnrolledObject.name)
        ]
        descriptions: list[tuple[str, str]] = [
            (row.enrolled_object.name, row.text)
            for row in FrameDescription.select(FrameDescription, EnrolledObject)
            .join(EnrolledObject)
            .where(FrameDescription.image == image_id, FrameDescription.text != "")
            .order_by(EnrolledObject.name)
        ]
        return FrameDetail(
            image,
            regions,
            sightings,
            open_regions,
            dict(scene.rooms) if scene is not None else None,
            last_seen_of,
            descriptions,
        )
