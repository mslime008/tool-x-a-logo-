from dataclasses import replace
import threading
import cv2
import numpy as np
import pytest
from logo_remover.gpu_service.config import ModelConfig
from logo_remover.gpu_service.models import load_adapter
from logo_remover.gpu_service.processing import TemporalProcessor, temporal_windows
from logo_remover.masks import Mask, active_mask
from logo_remover.metadata import parse_metadata


class WindowPredictor:
    """Deterministic test oracle, not an AI model."""
    def __init__(self):
        self.calls = []
    def infer(self, frames, masks):
        self.calls.append((frames.shape, masks.copy()))
        return np.full_like(frames, 40 * len(self.calls))


def setup_processor(tmp_path, adapter=None):
    meta = parse_metadata({"streams": [{"codec_type": "video", "width": 64, "height": 48,
                           "duration": ".6", "avg_frame_rate": "10/1"}]}, "source.mp4")
    frame = np.full((48, 64, 3), 120, np.uint8)
    for i in range(6):
        cv2.imwrite(str(tmp_path / f"source-{i:06d}.png"), frame)
    masks = [Mask(points=[(20, 15), (30, 25)], end=.5, padding=0, feather=0)]
    config = ModelConfig(window=4, overlap=2, context_pixels=4, max_pixels=4096)
    processor = TemporalProcessor(tmp_path, meta, masks, [i / 10 for i in range(6)], config,
                                  adapter or WindowPredictor(), threading.Event(), lambda *_: None)
    return processor, frame, masks


def test_temporal_context_overlap_blend_and_exact_outside_mask(tmp_path):
    model = WindowPredictor()
    processor, frame, masks = setup_processor(tmp_path, model)
    processor.run()
    assert len(model.calls) == 2 and all(shape[0] == 4 for shape, _ in model.calls)
    for i in range(6):
        output = processor(frame, masks, i / 10)
        binary, _ = active_mask(masks, 64, 48, i / 10)
        assert np.array_equal(output[binary == 0], frame[binary == 0])
        if i == 2:
            assert output[20, 25, 0] == 53  # (40*2 + 80*1)/3
        if i == 5:
            assert output is frame  # no active mask at interval end


def test_windows_scene_boundaries_and_spatial_budget(tmp_path):
    assert list(temporal_windows(0, 6, 4, 2)) == [(0, 4), (2, 6)]
    processor, frame, masks = setup_processor(tmp_path)
    for i in range(3, 6):
        frame = np.zeros_like(frame)
        frame[..., 2] = 255
        cv2.imwrite(str(tmp_path / f"source-{i:06d}.png"), frame)
    assert processor.scenes() == [(0, 3), (3, 6)]
    processor.run()
    assert all(shape[0] == 4 for shape, _ in processor.adapter.calls)
    with pytest.raises(ValueError, match="pixel budget"):
        TemporalProcessor(tmp_path, processor.meta, masks, processor.times,
            replace(processor.config, max_pixels=64), WindowPredictor(), threading.Event(), lambda *_: None)


def test_license_gate_and_checkpoint_integrity(tmp_path):
    with pytest.raises(ValueError, match="license/commercial"):
        load_adapter(ModelConfig())
    with pytest.raises(ValueError):
        load_adapter(ModelConfig(adapter="test-identity"))
    assert load_adapter(ModelConfig(adapter="test-identity"), allow_test=True)
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"not a model")
    config = ModelConfig(artifact=str(artifact), sha256="0" * 64, license_name="Operator agreement",
                         license_url="https://example.org/license", license_reviewed=True,
                         commercial_use_authorized=True)
    with pytest.raises(ValueError, match="SHA-256"):
        load_adapter(config)
