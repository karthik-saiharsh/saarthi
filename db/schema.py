"""Database tables.

Ingestion (the region archive):
- `Image`: one row for each uploaded frame.
- `Region`: one row for each detected box in a frame.

`Region.id` is also the vector id in the FAISS index. These vectors are not
stored in SQLite.

Enrollment (the gallery of the objects of the owner):
- `EnrolledObject`: one row for each object that the owner wants to find.
- `ReferencePhoto`: one row for each photo that a carer gave for an object.
- `ObjectView`: one vector for each view of an object. The gallery is small,
  so these vectors are stored in SQLite and not in FAISS.

Memory (what the system saw of the objects of the owner):
- `Sighting`: the log. One row for each accepted match of an object in a frame.
- `LastSeen`: one row for each object. The newest sighting that the rules accept.
  It keeps its frame: retention never deletes an image that a `LastSeen` row uses.

All timestamps are naive UTC values.
"""

from datetime import UTC, datetime
from enum import StrEnum

from peewee import (
    BlobField,
    BooleanField,
    CharField,
    Check,
    DatabaseProxy,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    TextField,
)
from playhouse.sqlite_ext import JSONField

# The real database is bound at start-up by `db.api.init_db`.
database_proxy: DatabaseProxy = DatabaseProxy()


class ImageStatus(StrEnum):
    """Processing state of an image."""

    PENDING = "pending"  # Saved on disk, not yet analysed
    PROCESSING = "processing"  # A worker is analysing the image
    DONE = "done"  # Regions and vectors are written
    REJECTED = "rejected"  # Dropped by a quality gate (blur, duplicate)
    FAILED = "failed"  # Processing raised an error


class ObjectStatus(StrEnum):
    """State of an enrolled object."""

    DRAFT = "draft"  # Photos are missing, in progress, or not confirmed
    READY = "ready"  # Calibrated. Matching can use the object


class PhotoStatus(StrEnum):
    """State of a reference photo."""

    PENDING = "pending"  # Saved on disk, not yet analysed
    PROCESSING = "processing"  # The worker looks for the object in the photo
    LOCATED = "located"  # The object was found. The carer must confirm the crop
    CONFIRMED = "confirmed"  # The carer accepted the crop. The views are used
    FAILED = "failed"  # The object was not found, or an error occurred


class RegionSource(StrEnum):
    """Detector pass that produced a region."""

    FULL = "full"  # Full-frame pass
    TILE = "tile"  # Tiled pass for small objects
    GENERIC = "generic"  # Generic "an object" pass


def _check_in(column: str, values: type[StrEnum]) -> Check:
    """Build a CHECK constraint that limits `column` to the enum values."""
    assert column.isidentifier(), f"Invalid column name: {column!r}"
    quoted: str = ", ".join(f"'{member.value}'" for member in values)
    return Check(f"{column} IN ({quoted})")


def utc_now() -> datetime:
    """Return the current UTC time as a naive datetime."""
    return datetime.now(UTC).replace(tzinfo=None)


class BaseModel(Model):
    """Base class that binds all tables to the shared database proxy."""

    class Meta:
        database = database_proxy


class Image(BaseModel):
    """One uploaded frame. The file on disk has no metadata (EXIF is stripped)."""

    # Relative to the storage root. Never absolute.
    file_path: CharField = CharField(unique=True)
    # Hash of the stripped file. Rejects exact duplicate uploads.
    sha256: CharField = CharField(max_length=64, unique=True)
    # Perceptual hash for near-duplicate checks. Null until computed.
    phash: CharField = CharField(max_length=16, null=True, index=True)
    width: IntegerField = IntegerField(constraints=[Check("width > 0")])
    height: IntegerField = IntegerField(constraints=[Check("height > 0")])
    # Time at which the server received the image.
    captured_at: DateTimeField = DateTimeField(default=utc_now, index=True)
    room: CharField = CharField(null=True)
    # Laplacian variance. Null until computed.
    blur_score: FloatField = FloatField(null=True)
    status: CharField = CharField(
        default=ImageStatus.PENDING.value,
        index=True,
        constraints=[_check_in("status", ImageStatus)],
    )
    # Reject reason or error text. Must not contain image content or personal data.
    status_detail: TextField = TextField(null=True)
    processed_at: DateTimeField = DateTimeField(null=True)

    class Meta:
        table_name = "image"


class Region(BaseModel):
    """One detected box in an image."""

    image: ForeignKeyField = ForeignKeyField(
        Image, backref="regions", on_delete="CASCADE", index=True
    )
    # Pixel coordinates in the frame of the parent image.
    x0: IntegerField = IntegerField(constraints=[Check("x0 >= 0")])
    y0: IntegerField = IntegerField(constraints=[Check("y0 >= 0")])
    x1: IntegerField = IntegerField(constraints=[Check("x1 > x0")])
    y1: IntegerField = IntegerField(constraints=[Check("y1 > y0")])
    # Detector score. For analysis only. Not used to decide identity.
    det_score: FloatField = FloatField()
    source: CharField = CharField(constraints=[_check_in("source", RegionSource)])
    # List of detector labels that remain after merging duplicate boxes.
    phrases: JSONField = JSONField(default=list)
    in_hand: BooleanField = BooleanField(default=False)
    # False when the crop failed the size, blur or edge gate.
    quality_ok: BooleanField = BooleanField(default=True)
    # Model that made the vector. Null means the vector is not in FAISS yet.
    embedding_model: CharField = CharField(null=True, index=True)

    class Meta:
        table_name = "region"


class EnrolledObject(BaseModel):
    """An object that the owner wants to find, for example "my blue bottle"."""

    name: CharField = CharField(max_length=80, unique=True)
    # The words that the detector looks for, for example "a bottle".
    category_phrase: CharField = CharField(max_length=40)
    # More words for the same kind of object, for example ["a flask", "a water bottle"].
    synonyms: JSONField = JSONField(default=list)
    # False: the object is only a look-alike. It is a negative and is never reported.
    tracked: BooleanField = BooleanField(default=True)
    status: CharField = CharField(
        default=ObjectStatus.DRAFT.value,
        index=True,
        constraints=[_check_in("status", ObjectStatus)],
    )
    # The result of the calibration. Null until the object is `ready`.
    tau: FloatField = FloatField(null=True)
    tau_strong: FloatField = FloatField(null=True)
    margin_min: FloatField = FloatField(null=True)
    # The model whose vectors the calibration used.
    calibration_model: CharField = CharField(null=True)
    # A short text for the carer, for example a warning about a look-alike.
    calibration_note: TextField = TextField(null=True)
    created_at: DateTimeField = DateTimeField(default=utc_now)
    updated_at: DateTimeField = DateTimeField(default=utc_now)

    class Meta:
        table_name = "enrolled_object"


class ReferencePhoto(BaseModel):
    """A photo of an enrolled object. The file is in the storage component."""

    enrolled_object: ForeignKeyField = ForeignKeyField(
        EnrolledObject, backref="photos", on_delete="CASCADE", index=True
    )
    # Relative to the storage root, like `Image.file_path`.
    file_path: CharField = CharField(unique=True)
    sha256: CharField = CharField(max_length=64)
    width: IntegerField = IntegerField(constraints=[Check("width > 0")])
    height: IntegerField = IntegerField(constraints=[Check("height > 0")])
    status: CharField = CharField(
        default=PhotoStatus.PENDING.value,
        index=True,
        constraints=[_check_in("status", PhotoStatus)],
    )
    # The reason of a failure. Must not contain personal data.
    status_detail: TextField = TextField(null=True)
    # The box of the object in the photo. Null until the object is located.
    x0: IntegerField = IntegerField(
        null=True, constraints=[Check("x0 IS NULL OR x0 >= 0")]
    )
    y0: IntegerField = IntegerField(
        null=True, constraints=[Check("y0 IS NULL OR y0 >= 0")]
    )
    x1: IntegerField = IntegerField(
        null=True, constraints=[Check("x1 IS NULL OR x1 > x0")]
    )
    y1: IntegerField = IntegerField(
        null=True, constraints=[Check("y1 IS NULL OR y1 > y0")]
    )
    det_score: FloatField = FloatField(null=True)
    created_at: DateTimeField = DateTimeField(default=utc_now)

    class Meta:
        table_name = "reference_photo"
        # The same picture can belong to two objects, but only once to one object.
        indexes = ((("enrolled_object", "sha256"), True),)


class ObjectView(BaseModel):
    """One vector of an enrolled object. A photo gives several (see the augmentations)."""

    enrolled_object: ForeignKeyField = ForeignKeyField(
        EnrolledObject, backref="views", on_delete="CASCADE", index=True
    )
    photo: ForeignKeyField = ForeignKeyField(
        ReferencePhoto, backref="views", on_delete="CASCADE", index=True
    )
    # float32 numbers of unit length, as bytes.
    vector: BlobField = BlobField()
    # Vectors of different models cannot be compared.
    embedding_model: CharField = CharField()
    # How the view was made, for example "original" or "box_smaller".
    augmentation: CharField = CharField()

    class Meta:
        table_name = "object_view"


class Sighting(BaseModel):
    """An accepted match of an enrolled object in a region of a frame."""

    enrolled_object: ForeignKeyField = ForeignKeyField(
        EnrolledObject, backref="sightings", on_delete="CASCADE", index=True
    )
    # The rows go with the frame (retention) and with the region (recovery).
    image: ForeignKeyField = ForeignKeyField(
        Image, backref="sightings", on_delete="CASCADE", index=True
    )
    region: ForeignKeyField = ForeignKeyField(
        Region, backref="sightings", on_delete="CASCADE", index=True
    )
    # The time of the frame (`Image.captured_at`), copied for the time window queries.
    seen_at: DateTimeField = DateTimeField(index=True)
    similarity: FloatField = FloatField()
    margin: FloatField = FloatField()
    # True: above `tau_strong` with the strong margin, or a weak match that passed the check
    # of stage I5. It does not need a second hit.
    strong: BooleanField = BooleanField()
    in_hand: BooleanField = BooleanField(default=False)

    class Meta:
        table_name = "sighting"
        indexes = ((("enrolled_object", "seen_at"), False),)


class LastSeen(BaseModel):
    """Where and when an object was seen last. One row for each object."""

    enrolled_object: ForeignKeyField = ForeignKeyField(
        EnrolledObject, backref="last_seen", on_delete="CASCADE", primary_key=True
    )
    # RESTRICT: the answer shows the crop of this frame, so the frame must stay.
    image: ForeignKeyField = ForeignKeyField(
        Image, backref="last_seen", on_delete="RESTRICT", index=True
    )
    # The box in the frame. It is a copy: the region row can go away.
    x0: IntegerField = IntegerField(constraints=[Check("x0 >= 0")])
    y0: IntegerField = IntegerField(constraints=[Check("y0 >= 0")])
    x1: IntegerField = IntegerField(constraints=[Check("x1 > x0")])
    y1: IntegerField = IntegerField(constraints=[Check("y1 > y0")])
    seen_at: DateTimeField = DateTimeField()
    similarity: FloatField = FloatField()
    margin: FloatField = FloatField()
    strong: BooleanField = BooleanField()
    in_hand: BooleanField = BooleanField(default=False)
    updated_at: DateTimeField = DateTimeField(default=utc_now)

    class Meta:
        table_name = "last_seen"
