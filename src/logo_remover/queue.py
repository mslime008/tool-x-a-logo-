from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import threading
import uuid
from PySide6.QtCore import QObject, QRunnable, Signal, Slot
from .export import Cancelled, ExportSettings, export_video
from .masks import Mask
from .metadata import VideoMetadata


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
            export_video(job.meta, job.masks, job.output, job.settings, job.cancel,
                         lambda n, text: self.signals.progress.emit(job.id, n, text))
            self.signals.finished.emit(job.id, "Hoàn tất", "")
        except Cancelled as exc:
            self.signals.finished.emit(job.id, "Đã hủy", str(exc))
        except Exception as exc:
            self.signals.finished.emit(job.id, "Lỗi", str(exc))
