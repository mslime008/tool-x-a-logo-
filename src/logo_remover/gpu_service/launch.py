"""RunPod GPU Pod entry point: supervise API and worker; deploy is operator-controlled."""
import os
import signal
import subprocess
import sys
import time
from .config import api_tokens, load_config


def main():
    os.umask(0o077)
    config = load_config()
    config.validate_runtime()
    api_tokens()  # Fail before listening if authentication is missing.
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "logo_remover.gpu_service.api:create_app",
                            "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1",
                            "--limit-concurrency", "32", "--timeout-keep-alive", "5", "--no-access-log"])
    worker_env = {k: v for k, v in os.environ.items()
                  if k not in {"LOGO_API_TOKENS", "LOGO_CLOUD_TOKEN", "RUNPOD_API_KEY"}}
    worker = subprocess.Popen([sys.executable, "-m", "logo_remover.gpu_service.worker"], env=worker_env)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping and api.poll() is None and worker.poll() is None:
            time.sleep(.2)
    finally:
        for process in (api, worker):
            if process.poll() is None:
                process.terminate()
        for process in (api, worker):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if not stopping:
        raise SystemExit("API or worker stopped unexpectedly; restart the Pod and inspect operator logs")


if __name__ == "__main__":
    main()
