"""Dedicated, single-host worker for experiments, scenarios, and GIF exports."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import threading
import uuid

from .jobs import JobCancellation, JobStore
from .server_settings import ServerSettings

log = logging.getLogger(__name__)


@contextmanager
def worker_lock(path):
    """The OS releases this lock on a crash; a live worker cannot be displaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another worker already owns this queue") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_job(application, store, job, owner):
    from .paved_scenario import PavedSiteConfig
    from .rainfall_replay import export_gif
    from .serialization import json_safe, public_config
    from .services import execute_experiment, evaluate_scenario, prepare_replay
    token = JobCancellation(store, job["id"])
    payload = job["payload"]
    progress = lambda fraction, message: store.progress(job["id"], owner, fraction, message)
    if token.cancelled:
        return None
    if job["kind"] == "experiment":
        config = application.experiment_config(payload)
        result = execute_experiment(config, progress, token)
        result.resolved_config["dataset_id"] = payload["dataset_id"]
        result.resolved_config["requested_config"] = public_config(config)
        # Persist partial output for inspection; cancelled runs are never promoted.
        if token.cancelled:
            result.cancelled = True
        saved = application.runs.save(result, promote=False)
        if payload.get("save_to_leaderboard") and not token.cancelled and not result.cancelled:
            application.runs.promote(saved.run_id)
        return {"run_id": saved.run_id, "partial": result.cancelled,
                "warnings": result.failures, "has_failures": bool(result.failures),
                "execution_failed": not result.predictions and not result.cancelled}
    run_id = payload["run_id"]
    result = application.runs.result(run_id)
    catalog = application.catalog_for_run(run_id)
    if job["kind"] == "scenario":
        request = payload["request"]
        site = dict(request["site"])
        if site.get("start_date"):
            from datetime import date
            site["start_date"] = date.fromisoformat(site["start_date"])
        output = evaluate_scenario(result, catalog, request["basin_id"], PavedSiteConfig(**site))
        folder = application.settings.storage_dir / "job-results" / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "scenario.json").write_text(json.dumps(json_safe(output), allow_nan=False))
        return {"artifact": "scenario", "run_id": run_id, "summary": json_safe(output.summary)}
    if job["kind"] == "gif":
        request = payload["request"]
        replay = prepare_replay(result, catalog, request["basin_id"], request["episode"],
                                event_days=request["event_days"])
        temporary = export_gif(replay)
        folder = application.settings.storage_dir / "job-results" / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        import shutil
        try:
            shutil.copyfile(temporary, folder / "replay.gif")
        finally:
            temporary.unlink(missing_ok=True)
        return {"artifact": "gif", "run_id": run_id}
    raise ValueError("unknown job kind")


def serve(settings, *, once=False):
    from .application import Application
    store = JobStore(settings.jobs_path)
    application = Application(settings)
    owner = uuid.uuid4().hex
    stopping = threading.Event()
    current = [None]

    def stop(signum=None, frame=None):
        stopping.set()
        if current[0]:
            store.cancel(current[0])

    def heartbeat():
        while not stopping.wait(2):
            try:
                if not store.heartbeat(owner):
                    os._exit(1)  # Never continue computation after losing queue ownership.
            except Exception:
                log.exception("Worker heartbeat failed")
                os._exit(1)

    with worker_lock(settings.storage_dir / "worker.lock"):
        store.acquire_worker(owner, stale_after=0)
        previous_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, stop)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            while not stopping.is_set():
                job = store.claim(owner)
                if job is None:
                    if once:
                        break
                    stopping.wait(0.5)
                    continue
                current[0] = job["id"]
                try:
                    result = run_job(application, store, job, owner)
                    error = ({"code": "training_failed", "message": "Experiment produced no usable predictions"}
                             if result and result.get("execution_failed") else None)
                    store.finish(job["id"], owner, result=result, error=error)
                except Exception:
                    log.exception("Job %s failed", job["id"])
                    store.finish(job["id"], owner, error={
                        "code": "execution_failed", "message": "Execution failed; inspect server logs"})
                finally:
                    current[0] = None
                if once:
                    break
        finally:
            stopping.set()
            thread.join(timeout=3)
            store.release_worker(owner)
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Process at most one queued job")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    serve(ServerSettings.load(args.settings), once=args.once)


if __name__ == "__main__":
    main()
