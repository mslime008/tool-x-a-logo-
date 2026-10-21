"""Shared cloud contracts. Real HTTP client is in cloud_client; mock never uploads."""
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol
import uuid
from .export import ExportSettings
from .masks import Mask
from .metadata import VideoMetadata


class JobState(str, Enum):
    QUEUED = "queued"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class UploadConsent:
    granted: bool
    provider_name: str
    expires_after_hours: int = 24
    delete_after_download: bool = True


def job_document(meta: VideoMetadata, masks: list[Mask], settings: ExportSettings,
                 model: str, consent: UploadConsent) -> dict:
    if not consent.granted or not consent.provider_name:
        raise PermissionError("Cần xác nhận rõ nhà cung cấp trước khi tải video lên.")
    if not 1 <= consent.expires_after_hours <= 24 or not consent.delete_after_download:
        raise ValueError("Lưu tối đa 24 giờ và bắt buộc xóa sau tải kết quả.")
    settings.validate()
    for mask in masks:
        mask.validate()
    if not model.strip():
        raise ValueError("Cần tên model cụ thể.")
    metadata = asdict(meta)
    metadata.pop("path")  # Do not leak local filesystem layout to a service.
    return {"schema_version": "1.0", "metadata": metadata,
            "masks": [m.to_dict() for m in masks], "model": model,
            "export": asdict(settings), "consent": asdict(consent)}


class CloudProvider(Protocol):
    name: str
    def submit(self, document: dict, video: Path) -> str: ...
    def poll(self, job_id: str) -> dict: ...
    def cancel(self, job_id: str) -> None: ...
    def retrieve(self, job_id: str, destination: Path) -> Path: ...
    def delete(self, job_id: str) -> None: ...


class MockProvider:
    name = "MOCK — không xử lý AI, không tải lên"

    def __init__(self):
        self.jobs: dict[str, dict] = {}

    def submit(self, document: dict, video: Path) -> str:
        consent = document.get("consent", {})
        if not consent.get("granted") or consent.get("provider_name") != self.name:
            raise PermissionError("Chưa xác nhận nhà cung cấp mock.")
        if document.get("schema_version") != "1.0":
            raise ValueError("Phiên bản schema không hỗ trợ.")
        job = str(uuid.uuid4())
        self.jobs[job] = {"state": JobState.COMPLETED.value, "mock": True,
                          "message": "Chỉ mô phỏng vòng đời; không có video AI đầu ra."}
        return job

    def poll(self, job_id: str) -> dict:
        return self.jobs[job_id].copy()

    def cancel(self, job_id: str) -> None:
        self.jobs[job_id]["state"] = JobState.CANCELLED.value

    def retrieve(self, job_id: str, destination: Path) -> Path:
        raise NotImplementedError("Mock không tạo video hoặc giả mạo kết quả AI.")

    def delete(self, job_id: str) -> None:
        del self.jobs[job_id]
