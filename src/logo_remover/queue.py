from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
import uuid
from PySide6.QtCore import QObject, QRunnable, Signal, Slot
from .export import Cancelled, ExportSettings, export_video
from .masks import Mask
from .metadata import VideoMetadata
from .cloud_client import CloudOptions, HttpGPUProvider
from .providers import UploadConsent, job_document


@dataclass
class Job:
    meta: VideoMetadata
    masks: list[Mask]
    output: Path
    settings: ExportSettings
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "Đang chờ"
    progress: int = 0
    error: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    cloud: CloudOptions | None = None
    remote_id: str | None = None

    @classmethod
    def snapshot(cls, meta, masks, output, settings):
        return cls(meta, deepcopy(masks), Path(output), deepcopy(settings))


class WorkerSignals(QObject):
    progress = Signal(str, int, str)
    finished = Signal(str, str, str)


class ExportWorker(QRunnable):
    def __init__(self, job: Job):
        super().__init__()
        self.job = job
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        job = self.job
        try:
            if job.cloud is None:
                export_video(job.meta, job.masks, job.output, job.settings, job.cancel,
                             lambda n, text: self.signals.progress.emit(job.id, n, text))
            else:
                self.run_cloud(job)
            self.signals.finished.emit(job.id, "Hoàn tất", "")
        except Cancelled as exc:
            self.signals.finished.emit(job.id, "Đã hủy", str(exc))
        except Exception as exc:
            self.signals.finished.emit(job.id, "Lỗi", str(exc))

    def run_cloud(self, job):
        options = job.cloud
        if not 0 <= time.time() - options.approved_at <= 86400:
            raise PermissionError("Xác nhận tải lên đã hết hạn; vui lòng xác nhận lại.")
        provider = HttpGPUProvider.from_environment(name=options.provider_name,
            cancel=job.cancel, progress=lambda n, text: self.signals.progress.emit(job.id, n, text),
            idempotency_key=options.idempotency_key, job_id=job.remote_id, capability_revision=options.capability_revision)
        try:
            if provider.url != options.url:
                raise PermissionError("Địa chỉ cloud đã thay đổi; cần xác nhận lại.")
            document = job_document(job.meta, job.masks, job.settings, options.model,
                UploadConsent(True, options.provider_name, options.retention_hours))
            remote_id = provider.submit(document, Path(job.meta.path))
            job.remote_id = remote_id
            provider.wait_and_retrieve(remote_id, job.output)
        finally:
            job.remote_id = provider.job_id
            if job.cancel.is_set() and provider.job_id:
                try:
                    provider.cancel(provider.job_id)
                    provider.delete(provider.job_id)
                except Exception:
                    pass  # Server TTL is still mandatory when cancellation cannot reach the server.
            provider.close()
