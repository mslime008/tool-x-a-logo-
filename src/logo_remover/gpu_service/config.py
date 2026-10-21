from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class ModelConfig:
    name: str = "unconfigured"
    adapter: str = "torchscript"
    artifact: str = ""
    sha256: str = ""
    license_name: str = ""
    license_url: str = ""
    license_reviewed: bool = False
    commercial_use_authorized: bool = False
    device: str = "cuda"
    window: int = 12
    overlap: int = 6
    context_pixels: int = 96
    max_pixels: int = 1280 * 720
    alignment: int = 8
    scene_cut_threshold: float = .65
    deterministic: bool = True

    def validate(self, allow_test: bool = False) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.name):
            raise ValueError("Invalid configured model name")
        if not 2 <= self.window <= 64 or not 1 <= self.overlap < self.window:
            raise ValueError("Temporal inference requires overlapping windows")
        if not 0 <= self.context_pixels <= 512 or not 64 <= self.max_pixels <= 4096 * 4096:
            raise ValueError("Invalid model spatial budget")
        if self.alignment not in {1, 2, 4, 8, 16, 32, 64} or not 0 < self.scene_cut_threshold <= 1:
            raise ValueError("Invalid alignment/scene threshold")
        if self.adapter == "test-identity" and allow_test:
            return
        if self.license_reviewed is not True or self.commercial_use_authorized is not True:
            raise ValueError("Model license/commercial authorization has not been reviewed")
        if not self.license_name or not self.license_url.startswith("https://"):
            raise ValueError("A model license reference is required")
        if not re.fullmatch(r"[a-f0-9]{64}", self.sha256) or not Path(self.artifact).is_file():
            raise ValueError("Configure an existing model artifact and its SHA-256")
        if not re.fullmatch(r"cuda(?::[0-9]+)?", self.device):
            raise ValueError("Production GPU adapter requires CUDA")
        if self.adapter != "torchscript" and not re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", self.adapter):
            raise ValueError("Invalid operator-configured adapter")


@dataclass(frozen=True)
class ServiceConfig:
    data_dir: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    provider_name: str = "Private GPU worker"
    data_region: str = ""
    privacy_policy_url: str = ""
    storage_encryption_confirmed: bool = False
    max_work_bytes: int = 40 * 1024**3
    max_upload_bytes: int = 512 * 1024 * 1024
    max_duration: float = 30.0
    max_frames: int = 1800
    max_dimension: int = 4096
    max_jobs_per_owner: int = 8
    max_storage_bytes: int = 80 * 1024**3
    min_free_bytes: int = 512 * 1024**2
    lease_seconds: int = 60
    job_timeout: int = 1800
    max_attempts: int = 2
    allow_test_model: bool = False

    def __post_init__(self):
        if not self.provider_name or len(self.provider_name) > 100:
            raise ValueError("Invalid provider name")
        if not 3 <= self.lease_seconds <= 3600 or not 1 <= self.max_attempts <= 5:
            raise ValueError("Invalid worker recovery settings")
        if self.max_upload_bytes <= 0 or self.max_storage_bytes <= 0 or self.job_timeout <= 0:
            raise ValueError("Invalid resource limits")

    @property
    def revision(self) -> str:
        snapshot = {"model": asdict(self.model), "provider": self.provider_name,
                    "region": self.data_region, "policy": self.privacy_policy_url}
        return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()

    def validate_runtime(self):
        self.model.validate(self.allow_test_model)
        if not (self.allow_test_model and self.model.adapter == "test-identity"):
            if self.storage_encryption_confirmed is not True:
                raise ValueError("Operator must provision encrypted media storage")
            if not self.data_region or not self.privacy_policy_url.startswith("https://"):
                raise ValueError("Provider data region and privacy policy are required")


def load_config() -> ServiceConfig:
    path = os.environ.get("LOGO_SERVICE_CONFIG")
    if not path:
        raise RuntimeError("Set LOGO_SERVICE_CONFIG to an operator-owned JSON configuration")
    data = json.loads(Path(path).read_text())
    data["data_dir"] = Path(data["data_dir"]).resolve()
    data["model"] = ModelConfig(**data.get("model", {}))
    # The test adapter cannot be enabled accidentally through a submitted job.
    data["allow_test_model"] = os.environ.get("LOGO_ALLOW_TEST_MODEL") == "1"
    return ServiceConfig(**data)


def api_tokens() -> dict[str, str]:
    tokens = json.loads(os.environ.get("LOGO_API_TOKENS", "{}"))
    if not isinstance(tokens, dict) or not tokens:
        raise RuntimeError("LOGO_API_TOKENS must map principal names to independent bearer tokens")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", k) or not isinstance(v, str)
           or len(v) < 32 or len(v) > 256 or not v.isascii() for k, v in tokens.items()):
        raise RuntimeError("Invalid token mapping; use high-entropy tokens of at least 32 characters")
    if len(set(tokens.values())) != len(tokens):
        raise RuntimeError("Each principal needs a distinct token")
    return tokens
