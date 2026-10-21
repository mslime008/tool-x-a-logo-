"""SQLite WAL queue. One local POSIX volume / one Pod, never a shared network DB."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import uuid
from .config import ServiceConfig

TERMINAL = {"completed", "failed", "cancelled", "expired", "deleted"}


class Conflict(RuntimeError):
    pass


class Store:
    def __init__(self, config: ServiceConfig, clock=time.time):
        self.config, self.clock = config, clock
        self.root = config.data_dir.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.db = self.root / "queue.sqlite3"
        with self.connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("BEGIN IMMEDIATE")
            marker = self.root / ".logo-remover-store"
            if not marker.exists():
                allowed = {self.db.name, self.db.name + "-wal", self.db.name + "-shm", self.db.name + "-journal"}
                if any(path.name not in allowed for path in self.root.iterdir()):
                    raise RuntimeError("Data directory must be empty or an existing dedicated Logo Remover store")
                marker.write_text("logo-remover-private-store-v1\n")
                os.chmod(marker, 0o600)
            elif marker.read_text() != "logo-remover-private-store-v1\n":
                raise RuntimeError("Invalid media store marker; refusing destructive maintenance")
            con.commit()
            con.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY, owner TEXT NOT NULL, idem TEXT NOT NULL,
                  fingerprint TEXT NOT NULL, document TEXT NOT NULL, upload TEXT NOT NULL, model_revision TEXT NOT NULL DEFAULT '',
                  state TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL,
                  updated REAL NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
                  claim TEXT, lease REAL, progress INTEGER NOT NULL DEFAULT 0,
                  error TEXT NOT NULL DEFAULT '', result TEXT, result_sha TEXT,
                  result_size INTEGER, cancel INTEGER NOT NULL DEFAULT 0,
                  UNIQUE(owner, idem));
                CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, created);
            ''')
            columns = {r[1] for r in con.execute("PRAGMA table_info(jobs)")}
            if "model_revision" not in columns:
                con.execute("ALTER TABLE jobs ADD COLUMN model_revision TEXT NOT NULL DEFAULT ''")
        os.chmod(self.db, 0o600)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db, timeout=15)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def directory(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise KeyError(job_id)
        return self.root / job_id

    def get(self, job_id: str, owner: str | None = None) -> dict:
        self.directory(job_id)
        with self.connect() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row or (owner is not None and row["owner"] != owner):
            raise KeyError(job_id)
        return dict(row)

    def create(self, owner: str, idem: str, payload: dict) -> tuple[dict, bool]:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        now, job_id = self.clock(), uuid.uuid4().hex
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute("SELECT * FROM jobs WHERE owner=? AND idem=?", (owner, idem)).fetchone()
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise Conflict("Idempotency key already used for different input")
                return dict(existing), False
            count = con.execute("SELECT count(*) FROM jobs WHERE owner=? AND state NOT IN ('expired','deleted','cancelled')",
                                (owner,)).fetchone()[0]
            if count >= self.config.max_jobs_per_owner:
                raise Conflict("Job quota reached; delete completed jobs")
            reserved = con.execute("SELECT upload FROM jobs WHERE state NOT IN ('expired','deleted','cancelled')").fetchall()
            if sum(json.loads(r[0])["size_bytes"] for r in reserved) + payload["upload"]["size_bytes"] > self.config.max_storage_bytes:
                raise Conflict("Service upload reservation quota reached")
            expiry = now + payload["document"]["consent"]["expires_after_hours"] * 3600
            directory = self.directory(job_id)
            directory.mkdir(mode=0o700)
            con.execute("INSERT INTO jobs(id,owner,idem,fingerprint,document,upload,model_revision,state,created,expires,updated) "
                        "VALUES(?,?,?,?,?,?,?,'uploading',?,?,?)",
                        (job_id, owner, idem, fingerprint, json.dumps(payload["document"]),
                         json.dumps(payload["upload"]), self.config.revision, now, expiry, now))
        return self.get(job_id), True

    def mutate(self, job_id: str, *, states: tuple[str, ...], **values) -> bool:
        allowed = {"state", "updated", "document", "claim", "lease", "progress", "error", "result",
                   "result_sha", "result_size", "cancel"}
        if not values or not set(values) <= allowed:
            raise ValueError("Invalid internal mutation")
        values["updated"] = self.clock()
        with self.connect() as con:
            cursor = con.execute("UPDATE jobs SET " + ",".join(f"{k}=?" for k in values)
                                 + " WHERE id=? AND state IN (" + ",".join("?" * len(states)) + ")",
                                 (*values.values(), job_id, *states))
            return cursor.rowcount == 1

    def claim_next(self) -> dict | None:
        now = self.clock()
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM jobs WHERE state='queued' AND expires>? AND cancel=0 "
                              "ORDER BY created LIMIT 1", (now,)).fetchone()
            if not row:
                return None
            token = uuid.uuid4().hex
            con.execute("UPDATE jobs SET state='running',claim=?,lease=?,updated=?,attempt=attempt+1,progress=0 "
                        "WHERE id=?", (token, now + self.config.lease_seconds, now, row["id"]))
        return self.get(row["id"])

    def heartbeat(self, job_id: str, claim: str) -> bool:
        now = self.clock()
        with self.connect() as con:
            cur = con.execute("UPDATE jobs SET lease=?,updated=? WHERE id=? AND claim=? AND state='running' "
                              "AND cancel=0 AND expires>? AND lease>?",
                              (now + self.config.lease_seconds, now, job_id, claim, now, now))
        return cur.rowcount == 1

    def progress(self, job_id: str, claim: str, percent: int):
        with self.connect() as con:
            con.execute("UPDATE jobs SET progress=? WHERE id=? AND claim=? AND state='running' AND cancel=0",
                        (max(0, min(99, percent)), job_id, claim))

    def finish(self, job_id: str, claim: str, result: Path | None, error: str = "") -> bool:
        now = self.clock()
        fields = (str(result.relative_to(self.root)) if result else None,
                  file_hash(result) if result else None, result.stat().st_size if result else None)
        with self.connect() as con:
            cur = con.execute("UPDATE jobs SET state=?,result=?,result_sha=?,result_size=?,progress=?,error=?,"
                              "claim=NULL,lease=NULL,updated=? WHERE id=? AND claim=? AND state='running' "
                              "AND cancel=0 AND expires>? AND lease>?",
                              ("failed" if error else "completed", *fields, 0 if error else 100,
                               error, now, job_id, claim, now, now))
        return cur.rowcount == 1

    def retry(self, job_id: str):
        with self.connect() as con:
            cur = con.execute("UPDATE jobs SET state='queued',error='',progress=0,updated=? "
                              "WHERE id=? AND state='failed' AND expires>? AND attempt<? AND cancel=0",
                              (self.clock(), job_id, self.clock(), self.config.max_attempts))
            if not cur.rowcount:
                raise Conflict("Retry unavailable, expired, or attempt limit reached")

    def cancel(self, job_id: str, delete=False):
        target = "deleted" if delete else "cancelled"
        with self.connect() as con:
            con.execute("UPDATE jobs SET state=?,cancel=1,updated=? WHERE id=? AND state NOT IN ('expired','deleted')",
                        (target, self.clock(), job_id))

    def maintenance(self):
        now = self.clock()
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("UPDATE jobs SET state='expired',cancel=1,document='{}',updated=? "
                        "WHERE expires<=? AND state NOT IN ('expired','deleted')", (now, now))
            con.execute("UPDATE jobs SET state=CASE WHEN attempt<? THEN 'queued' ELSE 'failed' END,"
                        "claim=NULL,lease=NULL,error='worker_lease_lost',updated=? "
                        "WHERE state='running' AND lease<=?", (self.config.max_attempts, now, now))
            stale = con.execute("SELECT id,state FROM jobs WHERE state IN ('receiving','validating') AND updated<?",
                                (now - 310,)).fetchall()
            for row in stale:
                con.execute("UPDATE jobs SET state=?,updated=? WHERE id=?",
                            ("uploading" if row["state"] == "receiving" else "uploaded", now, row["id"]))
                (self.directory(row["id"]) / "video.part").unlink(missing_ok=True)
            obsolete = con.execute("SELECT id FROM jobs WHERE state IN ('cancelled','expired','deleted')").fetchall()
        for row in obsolete:
            self.cleanup_files(row["id"])
        # Tombstones retain idempotency/owner and expiry only; never raw media/masks indefinitely.
        with self.connect() as con:
            con.execute("UPDATE jobs SET document='{}',result=NULL,result_sha=NULL,result_size=NULL "
                        "WHERE state IN ('cancelled','expired','deleted')")
            con.execute("DELETE FROM jobs WHERE expires<? AND state IN ('expired','deleted','cancelled')", (now - 86400,))
        # A killed API can leave a directory before its DB transaction committed.
        with self.connect() as con:
            known = {r[0] for r in con.execute("SELECT id FROM jobs")}
        with self.connect() as con:
            recoverable = [dict(r) for r in con.execute("SELECT id,claim,state FROM jobs WHERE state IN ('queued','failed')")]
        for row in recoverable:
            for old_work in self.directory(row["id"]).glob("work-*"):
                shutil.rmtree(old_work, ignore_errors=True)
        for directory in self.root.iterdir():
            if directory.is_dir() and re.fullmatch(r"[a-f0-9]{32}", directory.name) and directory.name not in known:
                if directory.stat().st_mtime < now - 300:
                    shutil.rmtree(directory)

    def cleanup_files(self, job_id: str):
        directory = self.directory(job_id)
        if directory.exists():
            shutil.rmtree(directory)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
