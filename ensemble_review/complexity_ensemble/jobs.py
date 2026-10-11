"""Persistent single-host job queue. No ML or web-framework dependencies."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import sqlite3
import time
import uuid


TERMINAL = {"succeeded", "failed", "cancelled"}


class QueueConflict(ValueError):
    pass


class JobStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, idempotency_key TEXT UNIQUE,
                    status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '', cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    result TEXT, error TEXT, owner TEXT
                );
                CREATE TABLE IF NOT EXISTS worker (
                    slot INTEGER PRIMARY KEY CHECK(slot=1), owner TEXT, heartbeat REAL
                );
                INSERT OR IGNORE INTO worker(slot,owner,heartbeat) VALUES(1,NULL,0);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            raise KeyError("job not found")
        value = dict(row)
        for key in ("payload", "result", "error"):
            value[key] = json.loads(value[key]) if value[key] else None
        value["cancel_requested"] = bool(value["cancel_requested"])
        return value

    def submit(self, kind, payload, idempotency_key=None):
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        fingerprint = hashlib.sha256((kind + encoded).encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                old = db.execute("SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if old:
                    if old["fingerprint"] != fingerprint:
                        raise QueueConflict("idempotency key was already used with a different request")
                    return self.decode(old)
            job_id = uuid.uuid4().hex
            now = time.time()
            db.execute('''INSERT INTO jobs
                (id,kind,payload,fingerprint,idempotency_key,status,created_at,updated_at)
                VALUES(?,?,?,?,?,'queued',?,?)''',
                       (job_id, kind, encoded, fingerprint, idempotency_key, now, now))
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def get(self, job_id):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def cancel(self, job_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
            if job["status"] not in TERMINAL:
                db.execute('''UPDATE jobs SET cancel_requested=1, updated_at=?,
                    status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,
                    message='Cancellation requested' WHERE id=?''', (time.time(), job_id))
        return self.get(job_id)

    def acquire_worker(self, owner, stale_after=30):
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            lease = db.execute("SELECT * FROM worker WHERE slot=1").fetchone()
            if lease["owner"] and lease["owner"] != owner and now - lease["heartbeat"] <= stale_after:
                raise QueueConflict("another worker already owns this queue")
            # An expired worker must terminate if it can no longer renew its lease.
            db.execute('''UPDATE jobs SET status='failed', updated_at=?,
                error=?, message='Worker interrupted' WHERE status='running' ''',
                       (now, json.dumps({"code": "worker_interrupted", "message": "Worker stopped before completion; resubmit to retry"})))
            db.execute("UPDATE worker SET owner=?,heartbeat=? WHERE slot=1", (owner, now))

    def heartbeat(self, owner):
        with self.connect() as db:
            cursor = db.execute("UPDATE worker SET heartbeat=? WHERE slot=1 AND owner=?", (time.time(), owner))
            return cursor.rowcount == 1

    def release_worker(self, owner):
        with self.connect() as db:
            db.execute("UPDATE worker SET owner=NULL,heartbeat=0 WHERE slot=1 AND owner=?", (owner,))

    def worker_ready(self, stale_after=30):
        with self.connect() as db:
            row = db.execute("SELECT * FROM worker WHERE slot=1").fetchone()
            return bool(row["owner"] and time.time() - row["heartbeat"] <= stale_after)

    def claim(self, owner):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            lease = db.execute("SELECT owner FROM worker WHERE slot=1").fetchone()
            if lease["owner"] != owner:
                raise QueueConflict("worker lease was lost")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',owner=?,updated_at=?,message='Starting' WHERE id=?",
                       (owner, time.time(), row["id"]))
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def progress(self, job_id, owner, fraction, message):
        with self.connect() as db:
            db.execute('''UPDATE jobs SET progress=?,message=?,updated_at=?
                WHERE id=? AND owner=? AND status='running' ''',
                       (max(0., min(1., fraction)), message, time.time(), job_id, owner))

    def finish(self, job_id, owner, result=None, error=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=? AND status='running'", (job_id, owner)).fetchone()
            if row is None:
                raise QueueConflict("job is no longer owned by this worker")
            status = "failed" if error else "cancelled" if row["cancel_requested"] else "succeeded"
            db.execute('''UPDATE jobs SET status=?,result=?,error=?,progress=?,message=?,updated_at=? WHERE id=?''',
                       (status, json.dumps(result, allow_nan=False) if result is not None else None,
                        json.dumps(error) if error else None, 1. if status == "succeeded" else row["progress"],
                        status.capitalize(), time.time(), job_id))
        return self.get(job_id)


class JobCancellation:
    def __init__(self, store, job_id):
        self.store, self.job_id = store, job_id

    @property
    def cancelled(self):
        return self.store.get(self.job_id)["cancel_requested"]

    def is_cancelled(self):
        return self.cancelled
