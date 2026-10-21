"""One job per child: isolate model/FFmpeg failures; OS-level OOM still needs host quotas."""
import argparse
import json
from pathlib import Path
import threading
from .config import load_config
from .contracts import JobDocument
from .models import load_adapter
from .processing import process_video
from .store import Store, file_hash
from ..export import ExportSettings
from ..masks import Mask
from ..metadata import probe


def execute(job_id: str, claim: str):
    config = load_config()
    store = Store(config)
    row = store.get(job_id)
    if row["state"] != "running" or row["claim"] != claim or row["cancel"]:
        raise RuntimeError("Stale worker claim")
    document = JobDocument.model_validate_json(row["document"])
    spec = json.loads(row["upload"])
    source = store.directory(job_id) / ("source" + spec["extension"])
    if file_hash(source) != spec["sha256"] or source.stat().st_size != spec["size_bytes"]:
        raise ValueError("Source integrity check failed")
    meta = probe(source, safe=True)
    if (meta.width, meta.height) != (document.metadata.width, document.metadata.height):
        raise ValueError("Source metadata mismatch")
    if meta.duration > config.max_duration or max(meta.width, meta.height) > config.max_dimension or meta.hdr:
        raise ValueError("Unsupported source")
    if document.model != config.model.name or row["model_revision"] != config.revision:
        raise ValueError("Job model no longer matches worker configuration")
    work = store.directory(job_id) / ("work-" + claim)
    work.mkdir(mode=0o700, exist_ok=False)
    adapter = load_adapter(config.model, config.allow_test_model)
    process_video(meta, [Mask(**mask.model_dump()) for mask in document.masks],
                  ExportSettings(**document.export.model_dump()), work / "result.mp4", config.model, adapter,
                  work, threading.Event(), lambda n, _: store.progress(job_id, claim, n), config.max_frames, config.max_work_bytes)
    # Keep only the final encoded file, not raw frames/prediction accumulators/logs.
    for path in work.iterdir():
        if path.name != "result.mp4":
            path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("claim")
    args = parser.parse_args()
    execute(args.job_id, args.claim)


if __name__ == "__main__":
    main()
