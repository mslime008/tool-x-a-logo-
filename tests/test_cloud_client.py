import hashlib
from pathlib import Path
import threading
import httpx
import pytest
from logo_remover.cloud_client import CloudError, HttpGPUProvider

TOKEN = "test-credential-" + "x" * 40
JOB = "a" * 32


def test_https_and_no_embedded_credentials_or_unconsented_upload(tmp_path):
    for url in ("http://example.com", "https://user:pass@example.com", "https://example.com?token=secret"):
        with pytest.raises(ValueError):
            HttpGPUProvider(url, TOKEN)
    requests = []
    client = httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request)))
    provider = HttpGPUProvider("https://gpu.example", TOKEN, client=client, name="Authorized provider")
    with pytest.raises(PermissionError):
        provider.submit({"consent": {"granted": False}}, tmp_path / "does-not-exist.mp4")
    assert not requests


def test_redirects_do_not_forward_bearer():
    hosts = []
    def handler(request):
        hosts.append(request.url.host)
        return httpx.Response(307, headers={"Location": "https://attacker.example/steal"})
    provider = HttpGPUProvider("https://gpu.example", TOKEN,
                               client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(CloudError, match="307"):
        provider.capabilities()
    assert hosts == ["gpu.example"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_checksum_download_no_overwrite_and_delete_only_after_verification(tmp_path, corrupt):
    data = b"encoded video result"
    digest = hashlib.sha256(data).hexdigest()
    methods = []
    def handler(request):
        methods.append((request.method, request.url.path))
        if request.url.path.endswith("/result"):
            return httpx.Response(200, content=b"bad" if corrupt else data,
                                  headers={"X-Content-SHA256": digest})
        if request.method == "DELETE":
            return httpx.Response(200, json={"state": "deleted"})
        return httpx.Response(200, json={"state": "completed", "result_sha256": digest, "result_size": len(data)})
    provider = HttpGPUProvider("https://gpu.example", TOKEN,
                               client=httpx.Client(transport=httpx.MockTransport(handler)))
    output = tmp_path / "result.mp4"
    if corrupt:
        with pytest.raises(CloudError, match="SHA-256"):
            provider.retrieve(JOB, output)
        assert not output.exists()
        assert not any(method == "DELETE" for method, _ in methods)
    else:
        assert provider.retrieve(JOB, output).read_bytes() == data
        assert any(method == "DELETE" for method, _ in methods)
        with pytest.raises(FileExistsError):
            provider.retrieve(JOB, output)
    assert not list(tmp_path.glob(".gpu-download-*"))


def test_cancel_requests_remote_cleanup_and_preserves_original(tmp_path):
    from logo_remover.export import Cancelled
    requests = []
    def handler(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(200, json={"state": "running", "progress": 1})
    event = threading.Event()
    event.set()
    provider = HttpGPUProvider("https://gpu.example", TOKEN, cancel=event,
                               client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(Cancelled):
        provider.wait_and_retrieve(JOB, tmp_path / "result.mp4")
    assert ("POST", f"/v1/jobs/{JOB}/cancel") in requests
    assert ("DELETE", f"/v1/jobs/{JOB}") in requests
    assert not list(tmp_path.iterdir())


def test_lost_commit_response_reconnects_without_duplicate_paid_submission(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"test-video")
    state = {"value": "uploading", "creates": 0, "commits": 0}
    def handler(request):
        path = request.url.path
        if path.endswith("/capabilities"):
            return httpx.Response(200, json={"api_version": "1.0", "provider_name": "GPU worker",
                                            "capability_revision": "b" * 64})
        if request.method == "POST" and path == "/v1/jobs":
            state["creates"] += 1
            return httpx.Response(201, json={"id": JOB})
        if path.endswith("/video"):
            assert request.read() == video.read_bytes()
            state["value"] = "uploaded"
        elif path.endswith("/commit"):
            state["commits"] += 1
            state["value"] = "queued"
            raise httpx.ReadError("lost response after commit", request=request)
        return httpx.Response(200, json={"state": state["value"]})
    provider = HttpGPUProvider("https://gpu.example", TOKEN,
                               client=httpx.Client(transport=httpx.MockTransport(handler)))
    document = {"consent": {"granted": True, "provider_name": "GPU worker"}, "masks": []}
    with pytest.raises(CloudError):
        provider.submit(document, video)
    assert provider.job_id == JOB
    assert provider.submit(document, video) == JOB
    assert state["creates"] == state["commits"] == 1
