"""Strict server-side job validation. No filesystem paths or remote URLs from clients."""
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ..export import ExportSettings
from ..masks import Mask


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MaskPayload(StrictModel):
    kind: Literal["rectangle", "polygon", "brush"]
    points: list[tuple[float, float]] = Field(min_length=1, max_length=10000)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    padding: int = Field(ge=0, le=100)
    feather: int = Field(ge=0, le=50)
    brush_size: int = Field(ge=1, le=500)

    @model_validator(mode="after")
    def valid_mask(self):
        Mask(**self.model_dump()).validate()
        return self


class MetadataPayload(StrictModel):
    width: int = Field(gt=0, le=4096)
    height: int = Field(gt=0, le=4096)
    duration: float = Field(gt=0, le=30)
    video_index: int = Field(ge=0)
    fps: str = Field(max_length=32)
    time_base: str = Field(max_length=32)
    start_time: float
    rotation: int
    sar: str = Field(max_length=32)
    pix_fmt: str = Field(max_length=64)
    color_space: str = Field(max_length=64)
    color_transfer: str = Field(max_length=64)
    color_primaries: str = Field(max_length=64)
    color_range: str = Field(max_length=64)
    audio_codec: str | None = Field(max_length=64)
    hdr: bool


class ExportPayload(StrictModel):
    codec: Literal["h264", "h265"]
    crf: int = Field(ge=0, le=51)
    preset: Literal["fast", "medium", "slow"]
    method: Literal["TELEA", "NS"]
    radius: int = Field(ge=1, le=20)


class ConsentPayload(StrictModel):
    granted: Literal[True]
    provider_name: str = Field(min_length=1, max_length=100)
    expires_after_hours: int = Field(ge=1, le=24)
    delete_after_download: Literal[True]

    @field_validator("granted", "delete_after_download", mode="before")
    @classmethod
    def explicit_boolean_consent(cls, value):
        if value is not True:
            raise ValueError("Explicit boolean consent is required")
        return value


class JobDocument(StrictModel):
    schema_version: Literal["1.0"]
    metadata: MetadataPayload
    masks: list[MaskPayload] = Field(min_length=1, max_length=32)
    model: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    export: ExportPayload
    consent: ConsentPayload

    @model_validator(mode="after")
    def mask_bounds(self):
        for mask in self.masks:
            if mask.end > self.metadata.duration + .1:
                raise ValueError("Mask interval exceeds video duration")
            if any(not 0 <= x < self.metadata.width or not 0 <= y < self.metadata.height
                   for x, y in mask.points):
                raise ValueError("Mask coordinates are outside the source frame")
        return self


class UploadSpec(StrictModel):
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extension: Literal[".mp4", ".mov"]


class CreateJob(StrictModel):
    capability_revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    document: JobDocument
    upload: UploadSpec


class MaskUpload(StrictModel):
    masks: list[MaskPayload] = Field(min_length=1, max_length=32)
