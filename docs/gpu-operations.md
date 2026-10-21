# Private GPU worker operation (RunPod GPU Pods)

This implementation is an operator-controlled HTTP service and GPU worker intended for a **RunPod GPU Pod** or another single Linux NVIDIA GPU machine. It is **not** a RunPod Serverless `/run` handler, autoscaler, or provisioner. No RunPod control-plane API key is needed by the desktop or this worker. No Pod has been rented, image deployed, model downloaded, or paid infrastructure created by the implementation.

The complete transport/queue/FFmpeg lifecycle is tested on CPU with an explicitly labeled identity test adapter. Real CUDA inference, the GPU Docker image, a real pretrained model's output quality, and a live RunPod HTTPS endpoint still require operator acceptance testing. Do not call a diagnostic result AI removal.

## Components and trust boundaries

```text
Windows desktop -- verified HTTPS + per-principal bearer --> API :8000 (HTTPS proxy/private ingress)
                                                             |
                                                 local private SQLite WAL + media directory
                                                             |
                                                  one worker supervisor / GPU
                                                             |
                                        fenced, time-limited inference child + FFmpeg subprocesses
                                                             |
                                                verified result download -> delete acknowledgement
```

Use one Pod, one worker, one POSIX local filesystem for SQLite/media. Do not put the WAL DB on NFS, an object-store mount or a multi-host shared RunPod network filesystem. Do not run multiple GPU workers or API replicas against this SQLite service. For multi-Pod scheduling, first replace the queue with PostgreSQL/Redis and media with private object storage; that is not supplied here. A Pod restart preserves jobs only when the underlying data mount survives. GPU process memory never survives restart.

The API supports trusted authenticated editors, not an anonymous public media hosting platform. It isolates jobs by principal, bounds payloads, rejects input URLs, uses opaque server-generated paths, verifies checksums, and never runs shell commands supplied by clients. Uploaded media is decoded only through the MOV/MP4 demuxer with file/pipe protocols and external data references disabled; disguised HLS/playlist uploads cannot initiate network fetches. GPU timestamp probing bounds demuxed packets to 10,000 and rejects oversized frame counts; pathological/many-track files can fail closed. Local input handling keeps its existing defaults. FFmpeg and a configured model parse potentially hostile media/code; a child process is fault containment, **not a security sandbox**. Use a hardened container, least-privilege networking, supported FFmpeg/PyTorch, GPU/CPU/RAM/disk quotas and a trusted media policy. Do not load untrusted TorchScript: it is executable code. Custom factories are operator-installed trusted code, never chosen via a submitted module name.

## Provisioning prerequisites (requires your infrastructure approval)

1. Review the actual [model/weights licenses](model-licenses.md). Select and validate a temporal model, its PyTorch/CUDA version, peak VRAM, tensor contract, and artifact checksum. There is no pretrained default.
2. Choose a RunPod **GPU Pod** and region yourself; do not expose a public unencrypted TCP API. Obtain the HTTPS URL from the RunPod console or your own verified TLS ingress. The app does not invent an endpoint. Check the proxy's upload size/read/idle limits; some gateways impose limits below the API's configured 512 MiB/300-second upload budget. Select a compatible proxy or reduce limits accordingly.
3. Provide an **encrypted local/POSIX data mount**, sufficient free scratch disk, and a separate read-only model/config mount. Only UID 10001 should own/access the media root. `storage_encryption_confirmed` is an operator attestation after provisioning encryption, not application encryption. If the selected RunPod storage cannot provide your required at-rest guarantees, do not upload production media. RunPod persistence and encryption semantics must be verified for your selected product/volume.
4. At the HTTPS ingress enforce request/concurrency/rate limits, a 512 KiB JSON limit, configured video size, upload deadline, restricted origin/network access and token-safe logs. Do not log Authorization, job bodies, signed queries, masks or video contents. The app disables Uvicorn access logging. No CORS credentials or browser session cookies are used.
5. Keep `LOGO_API_TOKENS` outside Git, Docker build args/layers and command histories. It is a JSON map of independent principal names to high-entropy tokens (at least 32 ASCII characters). Generate tokens using `secrets.token_urlsafe(48)` in your own secure environment; distribute only each editor's token. Multiple distinct principals cannot share a token. Rotate via your secret manager and restart the service; revocation is immediate after restart. TLS verification stays enabled.

## Build and launch — operator commands, not automatically executed

Review `deploy/Dockerfile.gpu` and build locally/on your build host:

```bash
docker build -f deploy/Dockerfile.gpu -t logo-remover-gpu:reviewed .
```

The default `gpu` image target uses Python 3.11, Debian FFmpeg, and CUDA-enabled PyTorch wheels. `--target service` builds a CPU-only transport/queue diagnostic image; it is not a real inpainting model or production CPU fallback. The latest official CUDA wheel may require `pypi.nvidia.com` in addition to the PyTorch/PyPI download hosts; preserve TLS/checksum verification when configuring restricted build networking. `TORCH_SPEC` and `TORCH_INDEX` are build arguments, **not credentials**. For production pin the exact supported torch release, base image digest, all dependencies and resulting image digest after validating your model. The default CUDA 12.6 wheel index requires a compatible NVIDIA host driver; use PyTorch/NVIDIA's current compatibility documentation for your chosen wheel/driver. `/app/build-packages.txt` records resolved packages. Do not bake checkpoints or secrets into the image. CUDA execution is not validated in this CPU environment. If a corporate build proxy needs a custom CA, supply its existing trusted CA bundle using BuildKit `--secret id=build_ca,src=/path/to/trusted-ca-bundle.pem`; the build uses it only for verified package downloads and does not persist the bundle in the image. Never disable TLS checks.

Copy `deploy/service-config.example.json` **outside the checkout**, edit all placeholders, and mount it read-only. The example intentionally refuses to start: license approval, checksum and encrypted storage are unset. Configure actual `data_region` and an HTTPS `privacy_policy_url` covering retention, provider access, cost and deletion. Set `LOGO_SERVICE_CONFIG` to that mounted file. Provide `LOGO_API_TOKENS` through RunPod's secure secret/environment settings or an operator-owned restricted env file, not an argument with literal credentials.

For a local GPU host, after setting up encrypted mounts and a TLS reverse proxy:

```bash
docker run --rm --gpus all --init --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  -p 127.0.0.1:8000:8000 \
  --mount type=bind,src=/secure/logo-data,dst=/data \
  --mount type=bind,src=/secure/logo-models,dst=/models,readonly \
  --mount type=bind,src=/secure/logo-config,dst=/config,readonly \
  --env-file /secure/logo-secrets.env \
  -e LOGO_SERVICE_CONFIG=/config/operator.json logo-remover-gpu:reviewed
```

Use equivalent reviewed mount/env/HTTP-port settings in the RunPod console. GPU Pod rental/push/deployment are explicit operator actions. The desktop only contacts your **data-plane HTTPS API**; it must never receive the RunPod account key. Use a dedicated, initially empty media root: a service marker guards destructive maintenance and unrelated non-service directories are rejected. The root media directory is mode 0700; API token values are not saved in SQLite or results. Keep diagnostic logs private. Caches requiring writes must go to explicitly allocated private scratch, not the read-only model mount.

`python -m logo_remover.gpu_service.launch` supervises the API and worker. Either process dying stops the service so the container supervisor can restart it. SIGTERM cancels the active child; cancelled workspaces are removed, and an interrupted running lease is recovered by the next worker. Use a container/PID supervisor that kills **all descendants** on Pod restart, including orphan CUDA children after an uncatchable parent failure.

## Model adapter and temporal processing

`ModelConfig.adapter` is `torchscript` or a trusted operator-installed `package.module:factory`. The factory receives `ModelConfig` and returns an object with:

```python
def infer(frames: numpy.ndarray, masks: numpy.ndarray) -> numpy.ndarray:
    # input BGR uint8: (T,H,W,3); holes uint8: (T,H,W), nonzero = reconstruct
    # return BGR uint8 with exactly the same shape and frame order
    ...
```

The TorchScript artifact's `forward(video, holes)` instead receives **RGB float32 [0,1] (T,3,H,W)** and binary float32 **(T,1,H,W)** on CUDA, and must return RGB float in [0,1] with the same video shape. A model using batch dimensions, optical-flow preprocessing, different ranges, reference frames or stateful memory needs a custom adapter; do not rename an arbitrary checkpoint `.pt` and assume compatibility. The artifact checksum is verified before loading. The adapter never fetches code, weights, dependencies or endpoints automatically.

Inference uses a fixed source-space ROI containing every mask plus context. It pads ROI edges to model alignment without resizing the source. `max_pixels` bounds padded ROI size; oversized 4K/multiple distant masks fail with an actionable error rather than silently downsizing or falling back to CPU. Tune `window`, `overlap`, context and pixel budget for measured VRAM. Model load/OOM/unsupported deterministic operators fail the job; the worker does not retry identical OOM configurations indefinitely.

Frames are processed in overlapping temporal windows, with triangular overlap averaging in **pixel space** and bounded disk-backed accumulators. Short tails/scenes repeat their last frame/mask to the configured window size for fixed-T models; padded predictions are discarded and never change output frame count/timestamps. Scene histogram cuts split windows; this is a heuristic and can miss cuts or over-split. Frames outside masks remain context only; inactive frames and every pixel outside padded active masks are copied from original decoded BGR before encoding. Feathering stays inside the mask. No per-frame model initialization or independent per-frame AI calls are used. Actual temporal consistency depends on the model; averaging alone cannot guarantee flicker-free reconstruction. Validate cuts, flow alignment and moving background scenes with your actual model.

Output reuses the original FFmpeg export pipeline: source dimensions/SAR/rotation, PTS/VFR, last-frame duration, known SDR color flags, CRF/preset H.264/H.265 CPU encoding, and source audio stream copy when MP4-compatible. Incompatible audio is explicitly converted to AAC as in local mode. No extra intermediate lossy encode is introduced. HDR remains rejected; BGR8/YUV420 re-encoding is not lossless. Refer to README limitations for color, exotic matrices, mixed audio codecs and containers. Encoding stays on CPU for predictable quality; the GPU runs the model, not an untested NVENC quality path.

RAM/VRAM are window/ROI bounded; PNG frames and float overlap sums live on disk. The original source is decoded a second time through the common export path to retain exactly the same source timeline/pixel compositing semantics. This trades disk/I/O for reuse and verification. `max_work_bytes` (default 40 GiB) and free-space checks reject oversized scratch estimates. `max_storage_bytes` limits **aggregate reserved uploads**, not a filesystem quota for every intermediate; additionally enforce actual volume quotas and monitor free space. Jobs default to at most 30 seconds, 1800 frames, 4096 pixels per dimension, 512 MiB upload, 8 retained jobs per principal and one processing slot. Adjust limits only after measuring resources.

## Desktop setup and per-job consent

Local Fast and Auto remain local and require no service, GPU, token, or consent. Cloud preview still uses the local algorithm and is labeled accordingly; only confirmed cloud exports use the GPU model.

In a new PowerShell terminal, obtain values securely from your operator and set them for that process (avoid pasting real keys into shared terminals/logs):

```powershell
$env:LOGO_CLOUD_URL = "https://YOUR_ACTUAL_HTTPS_DATA_PLANE"
$env:LOGO_CLOUD_TOKEN = "<your-editor-token-from-secure-storage>"
.\run_windows.bat
```

Do not commit these values or place tokens in URLs/QSettings. Select **Đám mây GPU**, choose retention 1–24 hours, and enqueue. The app fetches capabilities (no media upload), then shows video, provider URL/name, region, privacy policy, exact model and retention, warns about provider charges, and requires a positive upload/processing confirmation for **each job and retry**. Decline makes no upload and enqueues nothing. Model/policy configuration revisions are pinned at consent/create/execution. If configuration changes, obtain fresh consent rather than silently switching a model. The explicit diagnostic model is prominently labeled and does not remove logos.

## API protocol (v1)

All `/v1/*` routes require `Authorization: Bearer <editor token>`. `/healthz` is unauthenticated liveness only and says nothing about GPU/model readiness. Capabilities validate licensing/storage policy configuration but are **not a successful CUDA preflight**. A bad/missing model fails closed. The service disables interactive API docs in production; checked-in `schemas/gpu-create-job-v1.json` describes creation. The nested job document uses the existing v1 mask/metadata/export/consent schema.

| Method / route | Purpose |
| --- | --- |
| `GET /v1/capabilities` | Provider, policy/region, immutable revision, model/test flag, limits/license reference |
| `POST /v1/jobs` | Reserve job, document, exact upload size/SHA-256/extension and capability revision; requires 8–128 character `Idempotency-Key` |
| `PUT /v1/jobs/{id}/video` | Stream raw bytes, `application/octet-stream`; enforce size/hash; no client filename/URL accepted |
| `PUT /v1/jobs/{id}/masks` | Upload typed JSON coordinate masks/times; only before commit |
| `POST /v1/jobs/{id}/commit` | Probe actual video, reject mismatched metadata/HDR/unsupported size, enqueue immutably |
| `GET /v1/jobs` | Latest 100 jobs belonging to caller, for recovery after desktop restart |
| `GET /v1/jobs/{id}` | State, attempt, queue position, progress, expiry, safe error code, result hash/size, cleanup status |
| `POST /v1/jobs/{id}/cancel` | Cancel queued/running job and schedule removal of all job files |
| `POST /v1/jobs/{id}/retry` | Explicit retry of failed job, only before expiry and below attempt limit |
| `GET /v1/jobs/{id}/result` | Authenticated streaming result with size/SHA-256, no public/signed persistent download URL |
| `DELETE /v1/jobs/{id}` | Delete request/acknowledgement after verified download; cleanup is asynchronous |

Other principals get 404 for every owned resource. Malformed requests return safe errors without echoing job bodies. Video uploads restart in full after interruption; they are not chunk-resumable. Repeated POST with the same principal/idempotency key and same payload returns the same job; changed input conflicts. Validating/receiving requests are single-owner state transitions and recovered after a 310-second stale deadline. Committed masks/media cannot be replaced. No ZIPs, arbitrary paths, URLs or remote callback endpoints are accepted.

State progression is `uploading -> receiving -> uploaded -> validating -> queued -> running -> completed`; failure/cancel/expiry/delete can interrupt it. The worker atomically claims jobs with a random fencing token, periodically renews its lease, and only that unexpired claim may publish a result. Lost leases are requeued up to `max_attempts` (default 2); failed jobs require explicit retry. Child processes/timeouts/cancel kill the entire process group on Linux. Lost-response uploads/status/downloads can reconnect using the job ID and idempotency key. Network polling failure does not re-submit a new paid job. Desktop retry keeps the known remote job; a cancelled local job gets a new idempotency key only after fresh consent.

The client validates result size and SHA-256, writes to a private temporary file, fsyncs and atomically hard-links to a non-existing output path. A corrupted/interrupted download never appears as a completed file and never triggers delete acknowledgement. NTFS/POSIX hard links are required, as in local export. Successful download requests deletion; failure to reach the delete endpoint leaves mandatory TTL cleanup as a backstop. A running result stream may hold an already-open file descriptor until the transfer ends even after unlink; no new downloads are accepted after expiry.

## Expiry, cleanup and recovery

Retention starts at job creation, **not completion**, and is 1–24 hours. Every operation enforces expiry, and the reaper checks every five seconds while the service runs. Cancellation/delete remove source, temporary frames, model predictions, results and private worker logs. Status reports `cleanup=pending` until filesystem removal completes. Masks/document/results are scrubbed from queue rows; minimal ownership/idempotency/expiry metadata and diagnostic-hash tombstones survive up to 24 hours after expiry to prevent duplicate replay. A lost lease's old workspace is removed before a recovered attempt. Orphan directories from an API crash before a transaction commit are removed after five minutes. Upload scratch and source/result files never use user-supplied paths.

Cleanup failure disables new jobs until maintenance recovers; alert on cleanup errors/free disk. Files are ordinary deletion, not forensic secure erase. SQLite freed pages/WAL, underlying volume snapshots/backups and provider storage may retain historical bytes: configure encrypted storage, retention-compliant snapshots/backups and cryptographic key destruction if required. The app does not claim physical erasure. Do not back up job media to long-lived storage by default.

For maintenance when model/CUDA/license configuration is broken, an operator can mount the existing encrypted store on a CPU host and run `LOGO_SERVICE_CONFIG=/secure/operator.json python -m logo_remover.gpu_service.cleanup`. This one-shot command needs no model, GPU or API credentials; configure its scheduling/lifecycle externally if required. It must see the same dedicated POSIX store and operator configuration.

**A powered-off/unavailable Pod cannot run cleanup.** Provision an external lifecycle/volume deletion policy before production use if you require deletion during Pod downtime. This code does not create an external lifecycle service; TTL is enforced on next startup for surviving data. Never promise a wall-clock deletion SLA through outages without that external control. Transient filesystem errors are retried by the API reaper; permanent failures need operator intervention.

Recover an existing job without uploading media again (token/URL remain in environment):

```powershell
.\.venv\Scripts\python.exe -m logo_remover.cloud_client list
.\.venv\Scripts\python.exe -m logo_remover.cloud_client status --job <job-id>
.\.venv\Scripts\python.exe -m logo_remover.cloud_client download --job <job-id> --output "D:\Videos\recovered.mp4"
.\.venv\Scripts\python.exe -m logo_remover.cloud_client cancel --job <job-id>
.\.venv\Scripts\python.exe -m logo_remover.cloud_client delete --job <job-id>
```

The GUI queue is still session-local. Recovery CLI lists only your principal's recent jobs; it never accepts credentials on the command line and never uploads. For changed model/policy configuration create a freshly consented job. For `inference_failed`, inspect private `worker-<claim>.log` before expiry; correct the adapter/checkpoint/VRAM budget, then obtain fresh consent if any pinned configuration changed. Do not retry identical OOMs or disable license/TLS/checksum gates. For lease loss restart the whole Pod so orphan descendants are terminated, then let fenced recovery proceed.

## Validation and release acceptance

```bash
python -m pip install -e '.[test,service]'
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Tests cover original local behavior, owner/auth isolation, consent decline, JSON/upload limits, checksum corruption, immutable committed masks, idempotency, upload recovery, cancellation/expiry cleanup, fencing/attempt bounds, configuration/license gating, overlap weighting/scene splitting/pixel invariants, and both ASGI and real TCP HTTP client -> API -> separate child worker -> FFmpeg -> verified result/delete round trips with copied AAC (including an 8-second moving-background/overlay input). A disguised playlist regression verifies that media parsing cannot make HTTP requests. CI runs these with the clearly labeled CPU identity adapter; it is not evidence of neural model quality or CUDA performance.

Before serving real users: build/scan the GPU image and pin its digest, verify the actual license/weights record, run the selected model on the actual GPU, measure peak RAM/VRAM/disk and latency, and compare 8–15s real 1080p/4K clips (VFR/rotation/audio offsets/cuts) for ghosting/flicker/seams and encoding quality. Test TLS/auth rotation, reverse-proxy upload/download limits, restart during inference, cancellation while inside CUDA, stale worker fencing, disk full, corrupted results and deletion across outages. Run independent frame-PTS/audio packet checks and visually inspect outputs; architecture tests alone cannot establish high-quality AI removal.
