"""Supervise fenced, killable inference children. One GPU job per process."""
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
from .config import ServiceConfig, load_config
from .store import Store


class Worker:
    def __init__(self, config: ServiceConfig, store: Store | None = None):
        self.config = config
        self.store = store or Store(config)
        self.stop = threading.Event()

    def run_once(self) -> bool:
        self.store.maintenance()
        if self.stop.is_set():
            return False
        job = self.store.claim_next()
        if job is None:
            return False
        job_id, claim = job["id"], job["claim"]
        # Recovered attempts have distinct paths, so a stale child can never publish over a new attempt.
        work = self.store.directory(job_id) / ("work-" + claim)
        log_path = self.store.directory(job_id) / ("worker-" + claim + ".log")
        env = {k: v for k, v in os.environ.items()
               if k in {"PATH", "HOME", "SYSTEMROOT", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                        "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "LOGO_SERVICE_CONFIG",
                        "LOGO_ALLOW_TEST_MODEL", "PYTHONPATH", "VIRTUAL_ENV", "LANG"}}
        env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
        process = None
        try:
            with log_path.open("wb") as log:
                process = subprocess.Popen([sys.executable, "-m", "logo_remover.gpu_service.execute", job_id, claim],
                    stdout=log, stderr=log, env=env, start_new_session=(os.name != "nt"))
                deadline = time.monotonic() + self.config.job_timeout
                while process.poll() is None:
                    if self.stop.wait(.2):
                        self.kill(process)
                        # Leave running state: next worker recovers by lease; never publish incomplete output.
                        return True
                    if not self.store.heartbeat(job_id, claim):
                        self.kill(process)
                        return True
                    if time.monotonic() >= deadline:
                        self.kill(process)
                        self.store.finish(job_id, claim, None, "processing_timeout")
                        return True
                if process.returncode == 0 and (work / "result.mp4").is_file():
                    if self.store.finish(job_id, claim, work / "result.mp4"):
                        return True
                else:
                    self.store.finish(job_id, claim, None, "inference_failed; check private worker logs/configuration")
        except Exception:
            if process and process.poll() is None:
                self.kill(process)
            self.store.finish(job_id, claim, None, "worker_execution_failed")
        finally:
            row = self.store.get(job_id)
            expected = str((work / "result.mp4").relative_to(self.store.root))
            if row["state"] != "completed" or row["result"] != expected:
                shutil.rmtree(work, ignore_errors=True)
            # Logs can contain model paths; retain no logs/media beyond job retention.
            if row["state"] != "failed":
                log_path.unlink(missing_ok=True)
        return True

    @staticmethod
    def kill(process):
        if process.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()

    def run(self):
        while not self.stop.is_set():
            if not self.run_once():
                self.stop.wait(1)


def main():
    config = load_config()
    config.validate_runtime()
    worker = Worker(config)
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, lambda *_: worker.stop.set())
    worker.run()


if __name__ == "__main__":
    main()
