"""HTTPS client for the private GPU API. Never knows a RunPod control-plane key."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import urlsplit
import uuid
import httpx
from .export import Cancelled


class CloudError(RuntimeError):
    pass


@dataclass(frozen=True)
class CloudOptions:
    url: str
    provider_name: str
    model: str
    approved_at: float
    retention_hours: int = 1
    test_model: bool = False
    idempotency_key: str = ""
    capability_revision: str = ""
    data_region: str = ""
    privacy_policy_url: str = ""


def sha256_file(path: Path, cancel: threading.Event | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if cancel is not None and cancel.is_set():
                raise Cancelled("Đã hủy tác vụ.")
            digest.update(chunk)
    return digest.hexdigest()


class HttpGPUProvider:
    def __init__(self, url: str, token: str, *, name: str = "GPU worker", client=None,
                 cancel: threading.Event | None = None, progress=lambda *_: None,
                 allow_http_loopback=False, idempotency_key=None, job_id=None, capability_revision=None):
        parsed = urlsplit(url)
        development = (allow_http_loopback and parsed.scheme == "http"
                       and parsed.hostname in {"localhost", "127.0.0.1", "::1"})
        if (parsed.scheme != "https" and not development) or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Cần URL HTTPS không chứa thông tin đăng nhập, query hoặc fragment.")
        if not token or len(token) < 32 or not token.isascii():
            raise ValueError("Cần LOGO_CLOUD_TOKEN hợp lệ trong biến môi trường.")
        self.url, self.name = url.rstrip("/"), name
        self.client = client or httpx.Client(timeout=httpx.Timeout(30, read=60), verify=True, follow_redirects=False)
        self.owns_client = client is None
        self.headers = {"Authorization": "Bearer " + token}
        self.cancel_event, self.progress = cancel or threading.Event(), progress
        self.idempotency_key = idempotency_key or uuid.uuid4().hex
        self.job_id = job_id
        self.capability_revision = capability_revision

    @classmethod
    def from_environment(cls, **kwargs):
        return cls(os.environ.get("LOGO_CLOUD_URL", ""), os.environ.get("LOGO_CLOUD_TOKEN", ""),
                   allow_http_loopback=os.environ.get("LOGO_CLOUD_ALLOW_LOCAL_HTTP") == "1", **kwargs)

    def close(self):
        if self.owns_client:
            self.client.close()

    def request(self, method, path, **kwargs):
        headers = {**self.headers, **kwargs.pop("headers", {})}
        try:
            response = self.client.request(method, self.url + path, headers=headers, follow_redirects=False, **kwargs)
        except httpx.HTTPError:
            raise CloudError("Lỗi kết nối GPU worker; có thể thử lại với cùng mã tác vụ.") from None
        if response.status_code >= 300:
            # Do not echo remote bodies, signed URLs, local paths or credentials.
            raise CloudError(f"GPU worker trả HTTP {response.status_code}.")
        return response.json()

    def capabilities(self):
        result = self.request("GET", "/v1/capabilities")
        if result.get("api_version") != "1.0":
            raise CloudError("Phiên bản GPU API không hỗ trợ.")
        if self.capability_revision and self.capability_revision != result["capability_revision"]:
            raise PermissionError("Model hoặc chính sách cloud đã thay đổi; cần xác nhận lại.")
        self.capability_revision = result["capability_revision"]
        self.name = result["provider_name"]
        return result

    def submit(self, document: dict, video: Path) -> str:
        consent = document.get("consent", {})
        if consent.get("granted") is not True or consent.get("provider_name") != self.name:
            raise PermissionError("Chưa xác nhận tải lên nhà cung cấp này.")
        if self.cancel_event.is_set():
            raise Cancelled("Đã hủy tác vụ.")
        if self.job_id is None:
            self.capabilities()
            spec = {"size_bytes": video.stat().st_size, "sha256": sha256_file(video, self.cancel_event),
                    "extension": video.suffix.lower()}
            row = self.request("POST", "/v1/jobs", json={"document": document, "upload": spec, "capability_revision": self.capability_revision},
                               headers={"Idempotency-Key": self.idempotency_key})
            self.job_id = row["id"]
        row = self.poll(self.job_id)
        path = f"/v1/jobs/{self.job_id}"
        if row["state"] == "uploading":
            size = video.stat().st_size
            def chunks():
                sent = 0
                with video.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        if self.cancel_event.is_set():
                            raise Cancelled("Đã hủy tải lên.")
                        sent += len(chunk)
                        self.progress(int(10 * sent / max(1, size)), "Đang tải video lên GPU worker")
                        yield chunk
            self.request("PUT", path + "/video", content=chunks(), headers={
                "Content-Type": "application/octet-stream", "Content-Length": str(size)})
            row = self.poll(self.job_id)
        if row["state"] == "uploaded":
            self.request("PUT", path + "/masks", json={"masks": document["masks"]})
            self.request("POST", path + "/commit")
        elif row["state"] == "failed":
            self.request("POST", path + "/retry")  # Only invoked by an explicit user retry.
        elif row["state"] not in {"queued", "running", "completed"}:
            raise CloudError("Tác vụ không sẵn sàng; thử lại sau hoặc tạo tác vụ mới nếu đã hết hạn/hủy.")
        return self.job_id

    def poll(self, job_id):
        if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("Mã tác vụ cloud không hợp lệ.")
        return self.request("GET", f"/v1/jobs/{job_id}")

    def cancel(self, job_id):
        self.poll(job_id)
        self.request("POST", f"/v1/jobs/{job_id}/cancel")

    def delete(self, job_id):
        self.poll(job_id)
        return self.request("DELETE", f"/v1/jobs/{job_id}")

    def retrieve(self, job_id: str, destination: Path) -> Path:
        row = self.poll(job_id)
        if row["state"] != "completed":
            raise CloudError("Kết quả chưa sẵn sàng.")
        destination = Path(destination).resolve()
        if destination.exists():
            raise FileExistsError("Không ghi đè tệp đã có.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".gpu-download-", dir=destination.parent, delete=False) as handle:
                temporary = Path(handle.name)
                digest, received = hashlib.sha256(), 0
                try:
                    with self.client.stream("GET", self.url + f"/v1/jobs/{job_id}/result",
                                            headers=self.headers, follow_redirects=False) as response:
                        if response.status_code != 200:
                            raise CloudError(f"Tải kết quả thất bại (HTTP {response.status_code}).")
                        if response.headers.get("x-content-sha256") != row["result_sha256"]:
                            raise CloudError("Checksum của kết quả không khớp trạng thái tác vụ.")
                        for chunk in response.iter_bytes(1024 * 1024):
                            if self.cancel_event.is_set():
                                raise Cancelled("Đã hủy tải kết quả.")
                            received += len(chunk)
                            if received > row["result_size"]:
                                raise CloudError("Kết quả vượt kích thước đã công bố.")
                            digest.update(chunk)
                            handle.write(chunk)
                except httpx.HTTPError:
                    raise CloudError("Tải kết quả gián đoạn; dữ liệu tạm đã được dọn, có thể thử lại.") from None
                handle.flush()
                os.fsync(handle.fileno())
            if received != row["result_size"] or digest.hexdigest() != row["result_sha256"]:
                raise CloudError("Kết quả bị thiếu hoặc sai SHA-256; không công bố tệp.")
            os.link(temporary, destination)  # Atomic no-overwrite publication, as in local export.
            # Server deletion only after a verified, atomically published result. Failure leaves TTL as backstop.
            try:
                self.delete(job_id)
            except CloudError:
                self.progress(100, "Đã lưu kết quả; chưa xác nhận yêu cầu xóa từ xa, TTL vẫn áp dụng")
            return destination
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)

    def wait_and_retrieve(self, job_id: str, output: Path, timeout=3600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.cancel_event.is_set():
                try:
                    self.cancel(job_id)
                    self.delete(job_id)
                except CloudError:
                    pass
                raise Cancelled("Đã yêu cầu hủy cloud; nếu mất kết nối, máy chủ dọn theo TTL.")
            row = self.poll(job_id)
            label = {"queued": "đang chờ", "running": "đang xử lý", "completed": "hoàn tất",
                     "failed": "lỗi", "cancelled": "đã hủy", "expired": "hết hạn", "deleted": "đã yêu cầu xóa"}.get(row["state"], "đang kết nối")
            self.progress(10 + int(.8 * row["progress"]), f"Máy chủ GPU: {label}")
            if row["state"] == "completed":
                return self.retrieve(job_id, output)
            if row["state"] in {"failed", "cancelled", "expired", "deleted"}:
                raise CloudError(f"Máy chủ GPU: {label}; hãy kiểm tra trạng thái tác vụ trên máy chủ.")
            self.cancel_event.wait(1)
        raise CloudError("Hết thời gian chờ; tác vụ còn trên máy chủ đến khi hủy/hết hạn, thử lại để kết nối lại.")


def main():
    """Recovery without re-upload: credentials remain in environment, never CLI arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Recover/list private GPU jobs; no uploads are made by this CLI")
    parser.add_argument("action", choices=["list", "status", "download", "cancel", "delete"])
    parser.add_argument("--job")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action != "list" and not args.job:
        parser.error("--job is required")
    if args.action == "download" and not args.output:
        parser.error("--output is required")
    provider = HttpGPUProvider.from_environment()
    try:
        if args.action == "list":
            result = provider.request("GET", "/v1/jobs")
        elif args.action == "status":
            result = provider.poll(args.job)
        elif args.action == "download":
            result = {"output": str(provider.retrieve(args.job, args.output))}
        elif args.action == "cancel":
            result = provider.cancel(args.job)
        else:
            result = provider.delete(args.job)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        provider.close()


if __name__ == "__main__":
    main()
