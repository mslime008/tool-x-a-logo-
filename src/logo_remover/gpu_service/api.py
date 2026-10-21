"""Authenticated API; deploy behind an HTTPS terminator, no public unencrypted ingress."""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import time
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from .config import ServiceConfig, api_tokens, load_config
from .contracts import CreateJob, JobDocument, MaskUpload
from .store import Conflict, Store
from ..metadata import probe


class BodyLimit:
    """Reject oversized streaming JSON before Pydantic/JSON parsing allocates it."""
    def __init__(self, app, limit=512 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].endswith("/video"):
            return await self.app(scope, receive, send)
        total = 0
        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.limit:
                raise HTTPException(413, "JSON body too large")
            return message
        await self.app(scope, bounded_receive, send)


def public_job(row: dict, store: Store) -> dict:
    position = 0
    if row["state"] == "queued":
        with store.connect() as con:
            position = con.execute("SELECT count(*) FROM jobs WHERE state='queued' AND created<=?",
                                   (row["created"],)).fetchone()[0]
    return {"id": row["id"], "state": row["state"], "progress": row["progress"],
            "queue_position": position, "attempt": row["attempt"], "expires_at": row["expires"],
            "error": row["error"], "result_sha256": row["result_sha"], "result_size": row["result_size"],
            "cleanup": ("pending" if store.directory(row["id"]).exists() else "complete")
                       if row["state"] in {"cancelled", "expired", "deleted"} else "not_requested"}


def create_app(config: ServiceConfig | None = None, tokens: dict[str, str] | None = None,
               store: Store | None = None) -> FastAPI:
    config = config or load_config()
    tokens = tokens if tokens is not None else api_tokens()
    if (not tokens or len(set(tokens.values())) != len(tokens)
        or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", k) or not isinstance(v, str)
               or not 32 <= len(v) <= 256 or not v.isascii() for k, v in tokens.items())):
        raise ValueError("Independent high-entropy bearer tokens are required")
    store = store or Store(config)

    @asynccontextmanager
    async def lifespan(app):
        async def reap():
            while True:
                try:
                    await run_in_threadpool(store.maintenance)
                    app.state.cleanup_healthy = True
                except OSError:
                    app.state.cleanup_healthy = False
                    logging.getLogger(__name__).error("Media cleanup failed; new jobs disabled until cleanup recovers")
                await asyncio.sleep(5)
        task = asyncio.create_task(reap())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="Private Video GPU API", version="1.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimit)
    app.state.store = store
    app.state.cleanup_healthy = True
    validation_limit = asyncio.Semaphore(1)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Never echo input payloads, consent, tokens, masks or local operator paths.
        return JSONResponse({"detail": "Invalid request schema"}, status_code=422)

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    def principal(authorization: str = Header(default="")) -> str:
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Bearer authentication required", headers={"WWW-Authenticate": "Bearer"})
        supplied = authorization[7:]
        if not supplied.isascii() or len(supplied) > 256:
            raise HTTPException(401, "Invalid credentials")
        matched = None
        for name, token in tokens.items():
            if hmac.compare_digest(supplied, token):
                matched = name
        if matched is None:
            raise HTTPException(401, "Invalid credentials")
        return matched

    def owned(job_id, owner, allow_gone=False):
        try:
            row = store.get(job_id, owner)
        except KeyError:
            raise HTTPException(404, "Job not found") from None
        if row["expires"] <= store.clock() and row["state"] not in {"expired", "deleted"}:
            store.maintenance()
            try:
                row = store.get(job_id, owner)
            except KeyError:
                raise HTTPException(410, "Job expired") from None
        if not allow_gone and row["state"] in {"expired", "deleted"}:
            raise HTTPException(410, "Job expired or deleted")
        return row

    def configured():
        if not app.state.cleanup_healthy:
            raise HTTPException(503, "Media cleanup is unhealthy; new jobs disabled")
        try:
            config.validate_runtime()
        except ValueError:
            raise HTTPException(503, "Model is not configured or commercially authorized") from None

    @app.get("/healthz")
    def health():
        return {"status": "ok"}  # Liveness only; no filesystem/model configuration leakage.

    @app.get("/v1/capabilities")
    def capabilities(owner=Depends(principal)):
        configured()
        return {"api_version": "1.0", "capability_revision": config.revision,
                "data_region": config.data_region, "privacy_policy_url": config.privacy_policy_url,
                "provider_name": config.provider_name, "model": config.model.name,
                "test_model": config.model.adapter == "test-identity", "max_upload_bytes": config.max_upload_bytes,
                "max_duration": config.max_duration, "retention_hours_max": 24,
                "license": {"name": config.model.license_name, "url": config.model.license_url},
                "temporal_window": config.model.window, "overlap": config.model.overlap}

    @app.get("/v1/jobs")
    def list_jobs(owner=Depends(principal)):
        with store.connect() as con:
            rows = con.execute("SELECT * FROM jobs WHERE owner=? ORDER BY created DESC LIMIT 100", (owner,)).fetchall()
        return JSONResponse({"jobs": [public_job(dict(row), store) for row in rows]},
                            headers={"Cache-Control": "no-store"})

    @app.post("/v1/jobs")
    def create(payload: CreateJob, owner=Depends(principal), idempotency_key: str = Header(default="")):
        configured()
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", idempotency_key):
            raise HTTPException(422, "An 8–128 character Idempotency-Key is required")
        if payload.capability_revision != config.revision:
            raise HTTPException(409, "Provider/model policy changed; obtain fresh consent")
        if payload.upload.size_bytes > config.max_upload_bytes:
            raise HTTPException(413, "Video too large")
        if payload.document.model != config.model.name:
            raise HTTPException(422, "Unsupported model")
        if payload.document.consent.provider_name != config.provider_name:
            raise HTTPException(422, "Consent does not name this provider")
        row, created = store.create(owner, idempotency_key, payload.model_dump(mode="json"))
        return JSONResponse(public_job(row, store), status_code=201 if created else 200,
                            headers={"Cache-Control": "no-store"})

    @app.put("/v1/jobs/{job_id}/video")
    async def upload(job_id: str, request: Request, owner=Depends(principal)):
        row = owned(job_id, owner)
        spec = json.loads(row["upload"])
        if row["state"] == "uploaded":
            return public_job(row, store)  # Exact bytes already verified; lost response is safe to retry.
        if request.headers.get("content-type", "").split(";")[0] != "application/octet-stream":
            raise HTTPException(415, "Use application/octet-stream")
        length = request.headers.get("content-length")
        if length is not None and length != str(spec["size_bytes"]):
            raise HTTPException(413, "Upload size differs from reservation")
        if shutil.disk_usage(store.root).free < spec["size_bytes"] + config.min_free_bytes:
            raise HTTPException(507, "Insufficient upload space")
        if not store.mutate(job_id, states=("uploading",), state="receiving"):
            raise HTTPException(409, "Job is not accepting an upload")
        path = store.directory(job_id) / "video.part"
        total, digest = 0, hashlib.sha256()
        started = time.monotonic()
        try:
            with path.open("xb") as handle:
                os.chmod(path, 0o600)
                async with asyncio.timeout(300):
                    async for chunk in request.stream():
                        if time.monotonic() - started > 300:
                            raise HTTPException(408, "Upload deadline exceeded")
                        total += len(chunk)
                        if total > spec["size_bytes"] or total > config.max_upload_bytes:
                            raise HTTPException(413, "Upload exceeds reserved size")
                        if store.get(job_id)["state"] != "receiving":
                            raise HTTPException(409, "Job stopped accepting data")
                        digest.update(chunk)
                        await run_in_threadpool(handle.write, chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if total != spec["size_bytes"] or digest.hexdigest() != spec["sha256"]:
                raise HTTPException(422, "Upload checksum/size mismatch")
            # Serialize publication against cancellation/expiry; paths are server-generated.
            with store.connect() as con:
                con.execute("BEGIN IMMEDIATE")
                current = con.execute("SELECT state,expires FROM jobs WHERE id=?", (job_id,)).fetchone()
                if current["state"] != "receiving" or current["expires"] <= store.clock():
                    raise HTTPException(409, "Job no longer accepts data")
                os.replace(path, store.directory(job_id) / ("source" + spec["extension"]))
                con.execute("UPDATE jobs SET state='uploaded',updated=? WHERE id=?", (store.clock(), job_id))
        except TimeoutError:
            raise HTTPException(408, "Upload deadline exceeded") from None
        except OSError:
            raise HTTPException(507, "Temporary upload storage unavailable") from None
        finally:
            path.unlink(missing_ok=True)
            store.mutate(job_id, states=("receiving",), state="uploading")
        return public_job(store.get(job_id), store)

    @app.put("/v1/jobs/{job_id}/masks")
    def upload_masks(job_id: str, payload: MaskUpload, owner=Depends(principal)):
        row = owned(job_id, owner)
        document = json.loads(row["document"])
        document["masks"] = payload.model_dump(mode="json")["masks"]
        try:
            document = JobDocument.model_validate(document).model_dump(mode="json")
        except ValueError:
            raise HTTPException(422, "Invalid mask bounds/timestamps") from None
        if not store.mutate(job_id, states=("uploaded", "uploading"), document=json.dumps(document)):
            raise HTTPException(409, "Masks are immutable after commit")
        return public_job(store.get(job_id), store)

    @app.post("/v1/jobs/{job_id}/commit")
    async def commit(job_id: str, owner=Depends(principal)):
        row = owned(job_id, owner)
        if row["state"] in {"queued", "running", "completed"}:
            return public_job(row, store)
        if not store.mutate(job_id, states=("uploaded",), state="validating"):
            raise HTTPException(409, "Upload a complete video first")
        try:
            async with validation_limit:
                source = store.directory(job_id) / ("source" + json.loads(row["upload"])["extension"])
                actual = await run_in_threadpool(probe, source, safe=True)
            expected = json.loads(row["document"])["metadata"]
            if (actual.width, actual.height) != (expected["width"], expected["height"]):
                raise HTTPException(422, "Uploaded dimensions differ from job metadata")
            if not math.isfinite(actual.duration) or actual.duration > config.max_duration or max(actual.width, actual.height) > config.max_dimension:
                raise HTTPException(422, "Media exceeds configured processing limits")
            if abs(actual.duration - expected["duration"]) > .1 or actual.hdr or actual.width % 2 or actual.height % 2:
                raise HTTPException(422, "Unsupported media duration/HDR/dimensions")
            if not store.mutate(job_id, states=("validating",), state="queued"):
                raise HTTPException(409, "Job cancelled during validation")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(422, "Uploaded media could not be validated") from None
        finally:
            store.mutate(job_id, states=("validating",), state="uploaded")
        return public_job(store.get(job_id), store)

    @app.get("/v1/jobs/{job_id}")
    def status(job_id: str, owner=Depends(principal)):
        return JSONResponse(public_job(owned(job_id, owner, allow_gone=True), store),
                            headers={"Cache-Control": "no-store"})

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str, owner=Depends(principal)):
        owned(job_id, owner, allow_gone=True)
        store.cancel(job_id)
        return public_job(store.get(job_id), store)

    @app.post("/v1/jobs/{job_id}/retry")
    def retry(job_id: str, owner=Depends(principal)):
        owned(job_id, owner)
        store.retry(job_id)
        return public_job(store.get(job_id), store)

    @app.get("/v1/jobs/{job_id}/result")
    def result(job_id: str, owner=Depends(principal)):
        row = owned(job_id, owner)
        if row["state"] != "completed":
            raise HTTPException(409, "Result is not ready")
        path = (store.root / row["result"]).resolve()
        if not path.is_relative_to(store.directory(job_id)):
            raise HTTPException(500, "Invalid internal result path")
        try:
            handle = path.open("rb")
        except OSError:
            raise HTTPException(410, "Result no longer available") from None
        def chunks():
            try:
                yield from iter(lambda: handle.read(1024 * 1024), b"")
            finally:
                handle.close()
        return StreamingResponse(chunks(), media_type="video/mp4", headers={
            "Content-Length": str(row["result_size"]), "X-Content-SHA256": row["result_sha"],
            "Content-Disposition": 'attachment; filename="result.mp4"', "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff"})

    @app.delete("/v1/jobs/{job_id}")
    def delete(job_id: str, owner=Depends(principal)):
        owned(job_id, owner, allow_gone=True)
        store.cancel(job_id, delete=True)
        # Reaper/worker confirms cleanup asynchronously, including open/in-flight files.
        return {"id": job_id, "state": "deleted", "cleanup": "scheduled"}

    return app
