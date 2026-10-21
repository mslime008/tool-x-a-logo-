import json
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator
from logo_remover.export import ExportSettings
from logo_remover.metadata import parse_metadata
from logo_remover.masks import Mask
from logo_remover.providers import MockProvider, UploadConsent, job_document


def test_mock_lifecycle_and_schema(tmp_path):
    meta = parse_metadata({"streams": [{"codec_type": "video", "width": 1920, "height": 1080,
                           "duration": "8", "avg_frame_rate": "30/1"}]}, "private/source.mp4")
    provider = MockProvider()
    masks = [Mask(points=[(10, 10), (30, 30)], end=8)]
    with pytest.raises(PermissionError):
        job_document(meta, masks, ExportSettings(), "mock-no-ai", UploadConsent(False, provider.name))
    document = job_document(meta, masks, ExportSettings(), "mock-no-ai", UploadConsent(True, provider.name))
    # JSON wire roundtrip also converts Python tuples into schema arrays.
    document = json.loads(json.dumps(document))
    schema = json.loads((Path(__file__).parents[1] / "schemas/cloud-job-v1.json").read_text())
    Draft202012Validator(schema).validate(document)
    assert "path" not in document["metadata"]
    job_id = provider.submit(document, Path(meta.path))
    assert provider.poll(job_id)["mock"] is True
    with pytest.raises(NotImplementedError):
        provider.retrieve(job_id, tmp_path / "result.mp4")
    assert not list(tmp_path.iterdir())
    provider.cancel(job_id)
    assert provider.poll(job_id)["state"] == "cancelled"
    provider.delete(job_id)
    with pytest.raises(KeyError):
        provider.poll(job_id)
