"""Headless Qt checks; these are not visual Windows acceptance tests."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import threading
from pathlib import Path
from PySide6.QtWidgets import QApplication
from logo_remover.app import Window
from logo_remover.export import ExportSettings
from logo_remover.masks import Mask
from logo_remover.metadata import parse_metadata
from logo_remover.queue import Job


def test_window_and_mask_configuration():
    app = QApplication.instance() or QApplication([])
    window = Window()
    assert window.windowTitle().startswith("Xóa logo")
    window.add_mask("rectangle", [(1, 1), (5, 5)])  # No imported media: ignored.
    assert not window.masks
    window.change_tool(2)
    assert window.canvas.tool == "brush"
    window.close()
    app.processEvents()


def test_batch_snapshot_is_independent():
    meta = parse_metadata({"streams": [{"codec_type": "video", "width": 100, "height": 80,
                           "duration": "8", "avg_frame_rate": "25/1"}]}, "input.mp4")
    masks = [Mask(points=[(1, 1), (5, 5)])]
    settings = ExportSettings()
    job = Job.snapshot(meta, masks, Path("result.mp4"), settings)
    masks[0].points.append((99, 99))
    settings.crf = 30
    assert len(job.masks[0].points) == 2 and job.settings.crf == 18


def test_canvas_letterbox_draw_and_async_import(tmp_path):
    import shutil
    import subprocess
    import time
    import numpy as np
    import pytest
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtTest import QTest
    from logo_remover.app import Canvas
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required for asynchronous import")
    source = tmp_path / "gui.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=size=320x180:duration=1", "-c:v", "libx264", "-threads", "2", str(source)], check=True)
    app = QApplication.instance() or QApplication([])
    window = Window()
    window.open_video(str(source))
    deadline = time.monotonic() + 10
    while window.loading and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    assert window.meta is not None and window.original.shape == (180, 320, 3)
    window.canvas.resize(640, 480)
    rect = window.canvas.image_rect()
    assert rect.y() == 60
    emitted = []
    window.canvas.drawn.connect(lambda kind, points: emitted.append((kind, points)))
    QTest.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    QTest.mouseMove(window.canvas, QPoint(200, 200))
    QTest.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(200, 200))
    assert emitted[0][0] == "rectangle"
    assert emitted[0][1][0] == (50, 20)
    assert len(window.masks) == 1
    window.compute_preview()
    deadline = time.monotonic() + 10
    while window.tasks and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    assert window.repaired is not None
    assert window.view_mode.currentIndex() == 1
    window.output_dir = str(tmp_path)
    window.enqueue()
    deadline = time.monotonic() + 15
    while window.active_job is not None and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    assert len(window.jobs) == 1
    assert window.jobs[0].status == "Hoàn tất", window.jobs[0].error
    assert window.jobs[0].output.exists()
    window.close()
    app.processEvents()


def test_cloud_decline_never_enqueues_or_uploads(monkeypatch, tmp_path):
    import time
    from PySide6.QtTest import QTest
    from logo_remover.app import HttpGPUProvider
    app = QApplication.instance() or QApplication([])
    calls = []
    class Provider:
        url = "https://private-gpu.example"
        def capabilities(self):
            calls.append("capabilities")
            return {"provider_name": "Test GPU", "model": "licensed-model", "test_model": False,
                    "retention_hours_max": 24, "capability_revision": "a" * 64,
                    "data_region": "Development", "privacy_policy_url": "https://example.org/policy"}
        def close(self):
            pass
    monkeypatch.setattr(HttpGPUProvider, "from_environment", lambda **kwargs: Provider())
    window = Window()
    monkeypatch.setattr(window, "confirm_cloud", lambda *args: False)
    meta = parse_metadata({"streams": [{"codec_type": "video", "width": 100, "height": 80,
                           "duration": "8", "avg_frame_rate": "25/1"}]}, "input.mp4")
    job = Job.snapshot(meta, [Mask(points=[(1, 1), (5, 5)], end=8)], tmp_path / "output.mp4", ExportSettings())
    window.prepare_cloud(job)
    deadline = time.monotonic() + 5
    while window.tasks and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    assert calls == ["capabilities"] and window.jobs == []
    window.close()
    app.processEvents()
