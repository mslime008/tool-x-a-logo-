import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import pytest
from fastapi.testclient import TestClient
from logo_remover.gpu_service.api import create_app
from logo_remover.gpu_service.config import ModelConfig, ServiceConfig
from logo_remover.gpu_service.store import Store, Conflict
from logo_remover.providers import UploadConsent, job_document
from logo_remover.export import ExportSettings
from logo_remover.masks import Mask
from logo_remover.metadata import parse_metadata, probe, frame_times

TOKEN_A = "a" * 48
TOKEN_B = "b" * 48
AUTH = {"Authorization": "Bearer " + TOKEN_A}


@pytest.fixture
def service(tmp_path):
    config = ServiceConfig(data_dir=tmp_path / "data", model=ModelConfig(name="test-identity", adapter="test-identity",
        window=4, overlap=2, context_pixels=8, max_pixels=65536), allow_test_model=True, lease_seconds=3)
    store = Store(config)
    app = create_app(config, {"alice": TOKEN_A, "bob": TOKEN_B}, store)
    return TestClient(app), store, config


def payload(content=b"video", meta=None):
    meta = meta or parse_metadata({"streams": [{"codec_type": "video", "width": 160, "height": 96,
                                   "duration": "1", "avg_frame_rate": "10/1"}]}, "input.mp4")
    document = job_document(meta, [Mask(points=[(20, 20), (40, 40)], end=meta.duration)],
                            ExportSettings(preset="fast"), "test-identity", UploadConsent(True, "Private GPU worker", 1))
    return {"document": document, "upload": {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                                               "extension": ".mp4"}}


def create(client, data, key="test-job-123"):
    data = {**data, "capability_revision": client.get("/v1/capabilities", headers=AUTH).json()["capability_revision"]}
    response = client.post("/v1/jobs", json=data, headers={**AUTH, "Idempotency-Key": key})
    assert response.status_code in {200, 201}, response.text
    return response.json()["id"]


def upload(client, job_id, content):
    return client.put(f"/v1/jobs/{job_id}/video", content=content,
                      headers={**AUTH, "Content-Type": "application/octet-stream"})


def test_auth_owner_isolation_idempotency_and_no_payload_echo(service):
    client, store, config = service
    assert client.get("/v1/capabilities").status_code == 401
    assert client.get("/healthz").json() == {"status": "ok"}
    job_id = create(client, payload())
    assert create(client, payload()) == job_id
    assert client.post("/v1/jobs", json={**payload(b"different"), "capability_revision": config.revision},
                       headers={**AUTH, "Idempotency-Key": "test-job-123"}).status_code == 409
    for method, path in [("get", ""), ("get", "/result"), ("delete", ""), ("post", "/cancel")]:
        response = getattr(client, method)(f"/v1/jobs/{job_id}" + path,
                        headers={"Authorization": "Bearer " + TOKEN_B})
        assert response.status_code == 404
    data = payload()
    data["document"]["secret_path"] = "/never-echo-me"
    response = client.post("/v1/jobs", json=data, headers={**AUTH, "Idempotency-Key": "invalid-key-123"})
    assert response.status_code == 422 and "never-echo" not in response.text
    assert client.get("/v1/jobs/not-an-id", headers=AUTH).status_code == 404


def test_upload_integrity_limits_masks_and_commit_order(service):
    client, store, config = service
    job_id = create(client, payload())
    assert client.post(f"/v1/jobs/{job_id}/commit", headers=AUTH).status_code == 409
    assert upload(client, job_id, b"wrong").status_code == 422
    assert store.get(job_id)["state"] == "uploading"
    assert not list(store.directory(job_id).glob("*.part"))
    assert upload(client, job_id, b"video").status_code == 200
    assert upload(client, job_id, b"video").status_code == 200
    assert store.get(job_id)["state"] == "uploaded"
    bad_masks = payload()["document"]["masks"]
    bad_masks[0]["end"] = .0
    assert client.put(f"/v1/jobs/{job_id}/masks", json={"masks": bad_masks}, headers=AUTH).status_code == 422
    assert client.post(f"/v1/jobs/{job_id}/commit", headers=AUTH).status_code == 422  # invalid media
    assert store.get(job_id)["state"] == "uploaded"
    assert client.post("/v1/jobs", content=b" " * (512 * 1024 + 1),
                       headers={**AUTH, "Content-Type": "application/json"}).status_code == 413


def test_expiry_cancellation_cleanup_and_fenced_recovery(service):
    client, store, config = service
    now = [1000.0]
    store.clock = lambda: now[0]
    job_id = create(client, payload())
    assert upload(client, job_id, b"video").status_code == 200
    assert store.mutate(job_id, states=("uploaded",), state="queued")
    old = store.claim_next()
    assert old and old["attempt"] == 1
    (store.directory(job_id) / ("work-" + old["claim"])).mkdir()
    now[0] += 4
    store.maintenance()
    assert not store.heartbeat(job_id, old["claim"])
    replacement = store.claim_next()
    assert replacement["attempt"] == 2 and replacement["claim"] != old["claim"]
    assert not store.finish(job_id, old["claim"], None, "stale")
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=AUTH).status_code == 200
    assert not store.heartbeat(job_id, replacement["claim"])
    store.maintenance()
    assert not store.directory(job_id).exists()
    assert client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()["cleanup"] == "complete"
    now[0] += 3601
    store.maintenance()
    assert client.get(f"/v1/jobs/{job_id}/result", headers=AUTH).status_code == 410
    assert store.get(job_id)["document"] == "{}"


def test_recover_interrupted_upload_and_retry_limit(service):
    client, store, config = service
    now = [1000.0]
    store.clock = lambda: now[0]
    job_id = create(client, payload())
    store.mutate(job_id, states=("uploading",), state="receiving")
    (store.directory(job_id) / "video.part").write_bytes(b"partial")
    now[0] += 311
    store.maintenance()
    assert store.get(job_id)["state"] == "uploading"
    assert not (store.directory(job_id) / "video.part").exists()
    assert upload(client, job_id, b"video").status_code == 200
    store.mutate(job_id, states=("uploaded",), state="queued")
    job = store.claim_next()
    assert store.finish(job_id, job["claim"], None, "test_failure")
    assert client.post(f"/v1/jobs/{job_id}/retry", headers=AUTH).status_code == 200
    job = store.claim_next()
    assert store.finish(job_id, job["claim"], None, "test_failure")
    assert client.post(f"/v1/jobs/{job_id}/retry", headers=AUTH).status_code == 409


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_full_api_child_worker_result_audio_and_deletion(service, tmp_path, monkeypatch):
    from logo_remover.gpu_service.worker import Worker
    from logo_remover.cloud_client import HttpGPUProvider
    from dataclasses import asdict
    client, store, config = service
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=10:duration=1",
        "-f", "lavfi", "-i", "sine=duration=1", "-c:v", "libx264", "-threads", "2", "-c:a", "aac", str(source)], check=True)
    meta = probe(source)
    document = payload(source.read_bytes(), meta)["document"]
    provider = HttpGPUProvider("http://localhost", TOKEN_A, client=client, name=config.provider_name,
                               allow_http_loopback=True)
    remote = provider.submit(document, source)
    assert provider.poll(remote)["state"] == "queued"
    # Changes cannot alter a queued job's masks or source.
    assert client.put(f"/v1/jobs/{remote}/masks", json={"masks": document["masks"]}, headers=AUTH).status_code == 409
    cfg = asdict(config)
    cfg["data_dir"] = str(config.data_dir)
    cfg_path = tmp_path / "operator.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setenv("LOGO_SERVICE_CONFIG", str(cfg_path))
    monkeypatch.setenv("LOGO_ALLOW_TEST_MODEL", "1")
    worker = Worker(config, store)
    assert worker.run_once()
    status = provider.poll(remote)
    assert status["state"] == "completed", status
    result = tmp_path / "download.mp4"
    provider.retrieve(remote, result)
    assert probe(result).audio_codec == "aac"
    assert frame_times(probe(result))[0] == frame_times(meta)[0]
    def audio_hash(path):
        return subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                                        "-c", "copy", "-f", "hash", "-hash", "sha256", "-"])
    assert audio_hash(result) == audio_hash(source)
    assert provider.poll(remote)["state"] == "deleted"
    store.maintenance()
    assert not store.directory(remote).exists()
    assert not list(tmp_path.glob(".gpu-download-*"))


def test_capability_drift_license_gate_and_tenant_listing(service, tmp_path):
    from dataclasses import replace
    client, store, config = service
    body = {**payload(), "capability_revision": "0" * 64}
    assert client.post("/v1/jobs", json=body, headers={**AUTH, "Idempotency-Key": "drift-check"}).status_code == 409
    job_id = create(client, payload())
    assert client.get("/v1/jobs", headers=AUTH).json()["jobs"][0]["id"] == job_id
    assert client.get("/v1/jobs", headers={"Authorization": "Bearer " + TOKEN_B}).json()["jobs"] == []
    unconfigured = replace(config, model=ModelConfig(), allow_test_model=False)
    guarded = TestClient(create_app(unconfigured, {"alice": TOKEN_A}, Store(unconfigured)))
    assert guarded.get("/v1/capabilities", headers=AUTH).status_code == 503
    with pytest.raises(ValueError):
        create_app(config, {"alice": TOKEN_A, "bob": TOKEN_A}, store)


@pytest.mark.skipif(__import__('os').name == 'nt', reason="GPU service supervisor targets Linux")
def test_active_worker_cancellation_kills_child_process_group(service, monkeypatch):
    import sys
    import threading
    import logo_remover.gpu_service.worker as worker_module
    client, store, config = service
    job_id = create(client, payload())
    store.mutate(job_id, states=("uploading",), state="queued")
    real_popen = subprocess.Popen
    children = []
    def sleeper(*args, **kwargs):
        process = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(worker_module.subprocess, "Popen", sleeper)
    worker = worker_module.Worker(config, store)
    thread = threading.Thread(target=worker.run_once)
    thread.start()
    deadline = time.monotonic() + 5
    while not children and time.monotonic() < deadline:
        time.sleep(.02)
    assert children
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=AUTH).status_code == 200
    thread.join(timeout=5)
    assert not thread.is_alive() and children[0].poll() is not None
    store.maintenance()
    assert not store.directory(job_id).exists()


def test_completed_media_expires_and_configuration_schema_is_published(service):
    from jsonschema import Draft202012Validator
    client, store, config = service
    now = [1000.0]
    store.clock = lambda: now[0]
    data = {**payload(), "capability_revision": config.revision}
    schema = json.loads((Path(__file__).parents[1] / "schemas/gpu-create-job-v1.json").read_text())
    Draft202012Validator(schema).validate(json.loads(json.dumps(data)))
    job_id = create(client, data)
    upload(client, job_id, b"video")
    store.mutate(job_id, states=("uploaded",), state="queued")
    job = store.claim_next()
    work = store.directory(job_id) / ("work-" + job["claim"])
    work.mkdir()
    result = work / "result.mp4"
    result.write_bytes(b"test encoded bytes")
    assert store.finish(job_id, job["claim"], result)
    assert client.get(f"/v1/jobs/{job_id}/result", headers=AUTH).content == b"test encoded bytes"
    now[0] += 3601
    assert client.get(f"/v1/jobs/{job_id}/result", headers=AUTH).status_code == 410
    assert not store.directory(job_id).exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_real_http_eight_second_gpu_protocol_with_diagnostic_adapter(tmp_path):
    """Real TCP + API/worker processes; diagnostic adapter, no neural inference claim."""
    from dataclasses import asdict
    import os
    import socket
    import sys
    from logo_remover.cloud_client import HttpGPUProvider
    source = tmp_path / "eight-seconds.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=8:duration=8",
        "-f", "lavfi", "-i", "sine=duration=8", "-vf", "drawbox=x=110:y=10:w=30:h=16:color=white:t=fill",
        "-c:v", "libx264", "-threads", "2", "-c:a", "aac", str(source)], check=True)
    config = ServiceConfig(data_dir=tmp_path / "tcp-data", model=ModelConfig(name="test-identity", adapter="test-identity",
                           window=4, overlap=2, context_pixels=8, max_pixels=65536), allow_test_model=True)
    cfg = asdict(config)
    cfg["data_dir"] = str(config.data_dir)
    config_path = tmp_path / "tcp-operator.json"
    config_path.write_text(json.dumps(cfg))
    env = {**os.environ, "LOGO_SERVICE_CONFIG": str(config_path), "LOGO_ALLOW_TEST_MODEL": "1",
           "LOGO_API_TOKENS": json.dumps({"alice": TOKEN_A})}
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    processes = []
    with (tmp_path / "tcp-test.log").open("wb") as log:
        try:
            processes.append(subprocess.Popen([sys.executable, "-m", "uvicorn",
                "logo_remover.gpu_service.api:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port),
                "--no-access-log"], env=env, stdout=log, stderr=log))
            processes.append(subprocess.Popen([sys.executable, "-m", "logo_remover.gpu_service.worker"],
                                               env=env, stdout=log, stderr=log))
            provider = HttpGPUProvider(f"http://127.0.0.1:{port}", TOKEN_A,
                                        name=config.provider_name, allow_http_loopback=True)
            deadline = time.monotonic() + 15
            while True:
                try:
                    provider.capabilities()
                    break
                except Exception:
                    if time.monotonic() >= deadline or any(p.poll() is not None for p in processes):
                        raise AssertionError("API/worker did not start; see test log")
                    time.sleep(.05)
            document = payload(source.read_bytes(), probe(source))["document"]
            remote = provider.submit(document, source)
            result = tmp_path / "eight-second-result.mp4"
            provider.wait_and_retrieve(remote, result, timeout=30)
            assert len(frame_times(probe(result))[0]) == 64
            assert probe(result).duration == pytest.approx(8, abs=.001)
            def audio_hash(path):
                return subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                                                "-c", "copy", "-f", "hash", "-hash", "sha256", "-"])
            assert audio_hash(source) == audio_hash(result)
            assert provider.poll(remote)["state"] == "deleted"
            provider.close()
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def test_numeric_true_is_not_upload_consent(service):
    client, store, config = service
    data = {**payload(), "capability_revision": config.revision}
    data["document"]["consent"]["granted"] = 1
    response = client.post("/v1/jobs", json=data, headers={**AUTH, "Idempotency-Key": "not-consent-123"})
    assert response.status_code == 422
    assert client.get("/v1/jobs", headers=AUTH).json()["jobs"] == []


def test_disguised_playlist_cannot_fetch_network_resources(service):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    client, store, config = service
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_error(404)
        def log_message(self, *args):
            pass
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        playlist = ("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n"
                    f"http://127.0.0.1:{httpd.server_port}/must-not-fetch.ts\n#EXT-X-ENDLIST\n").encode()
        job_id = create(client, payload(playlist))
        assert upload(client, job_id, playlist).status_code == 200
        assert client.post(f"/v1/jobs/{job_id}/commit", headers=AUTH).status_code == 422
        assert requests == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_cleanup_without_model_or_credentials_and_refuse_unrelated_directory(service, tmp_path):
    from dataclasses import replace
    from logo_remover.gpu_service.cleanup import cleanup_once
    client, store, config = service
    job_id = create(client, payload())
    upload(client, job_id, b"video")
    with store.connect() as con:
        con.execute("UPDATE jobs SET expires=0 WHERE id=?", (job_id,))
    cleanup_once(replace(config, model=ModelConfig(), allow_test_model=False))
    assert not store.directory(job_id).exists()
    unrelated = tmp_path / "existing-user-data"
    unrelated.mkdir()
    keep = unrelated / "important.txt"
    keep.write_text("preserve this")
    with pytest.raises(RuntimeError, match="dedicated"):
        Store(replace(config, data_dir=unrelated))
    assert keep.read_text() == "preserve this"
