"""Operator-owned model contract; no pretrained artifacts are downloaded implicitly."""
import importlib
from typing import Protocol
import numpy as np
from .config import ModelConfig
from .store import file_hash
from pathlib import Path


class TemporalAdapter(Protocol):
    def infer(self, frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """BGR uint8 T,H,W,3 and binary uint8 T,H,W -> same BGR uint8 shape."""
        ...


class IdentityTestAdapter:
    """Transport/temporal-window diagnostics ONLY. Performs no logo removal."""
    def infer(self, frames, masks):
        return frames.copy()


class TorchScriptAdapter:
    def __init__(self, config: ModelConfig):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; this worker requires a GPU, use desktop local mode for CPU")
        self.torch, self.device = torch, config.device
        torch.manual_seed(0)
        if config.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        # A trusted, locally provisioned artifact only; TorchScript is executable code.
        self.model = torch.jit.load(config.artifact, map_location=self.device).eval()

    def infer(self, frames, masks):
        torch = self.torch
        rgb = np.ascontiguousarray(frames[..., ::-1].transpose(0, 3, 1, 2))
        video = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).div_(255)
        holes = torch.from_numpy(np.ascontiguousarray(masks[:, None] > 0)).to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            output = self.model(video, holes)
        if (not isinstance(output, torch.Tensor) or not output.is_floating_point()
                or tuple(output.shape) != tuple(video.shape)):
            raise ValueError("Model must return RGB float T,C,H,W with unchanged dimensions")
        if not bool(torch.isfinite(output).all()) or bool((output < 0).any()) or bool((output > 1).any()):
            raise ValueError("Model returned non-finite/out-of-range pixels")
        return np.ascontiguousarray(output.mul(255).round().to(torch.uint8).cpu().numpy()
                                    .transpose(0, 2, 3, 1)[..., ::-1])


def load_adapter(config: ModelConfig, allow_test=False) -> TemporalAdapter:
    config.validate(allow_test)
    if config.adapter == "test-identity":
        return IdentityTestAdapter()
    if file_hash(Path(config.artifact)) != config.sha256:
        raise ValueError("Model artifact SHA-256 differs from the reviewed configuration")
    if config.adapter == "torchscript":
        return TorchScriptAdapter(config)
    module, factory = config.adapter.split(":")
    adapter = getattr(importlib.import_module(module), factory)(config)
    if not callable(getattr(adapter, "infer", None)):
        raise ValueError("Configured adapter does not implement infer")
    return adapter
