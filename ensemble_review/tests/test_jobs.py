"""Queue contracts also runnable using the standard-library unittest runner."""
import concurrent.futures
from pathlib import Path
import tempfile
import unittest

from complexity_ensemble.jobs import JobStore, QueueConflict
from complexity_ensemble.serialization import json_safe
from complexity_ensemble.server_settings import ServerSettings


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "jobs.sqlite3"
        self.store = JobStore(self.path)

    def test_idempotency_is_durable_and_conflicting_payloads_are_rejected(self):
        first = self.store.submit("experiment", {"seed": 1}, "request-1")
        second = JobStore(self.path).submit("experiment", {"seed": 1}, "request-1")
        self.assertEqual(first["id"], second["id"])
        with self.assertRaises(QueueConflict):
            self.store.submit("experiment", {"seed": 2}, "request-1")
        with self.assertRaises(QueueConflict):
            self.store.submit("gif", {"seed": 1}, "request-1")

    def test_concurrent_duplicate_submissions_create_one_job(self):
        def submit(_):
            return JobStore(self.path).submit("experiment", {"seed": 1}, "same")["id"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(submit, range(12)))
        self.assertEqual(len(set(ids)), 1)

    def test_queued_cancellation_never_starts(self):
        job = self.store.submit("experiment", {})
        self.assertEqual(self.store.cancel(job["id"])["status"], "cancelled")
        self.store.acquire_worker("worker-1")
        self.assertIsNone(self.store.claim("worker-1"))

    def test_running_cancellation_retains_partial_result(self):
        job = self.store.submit("experiment", {})
        self.store.acquire_worker("worker-1")
        self.store.claim("worker-1")
        self.assertEqual(self.store.cancel(job["id"])["status"], "running")
        output = self.store.finish(job["id"], "worker-1", {"run_id": "partial"})
        self.assertEqual(output["status"], "cancelled")
        self.assertEqual(output["result"]["run_id"], "partial")

    def test_worker_recovery_marks_interrupted_jobs_without_retrying(self):
        old = self.store.submit("experiment", {"n": 1})
        queued = self.store.submit("experiment", {"n": 2})
        self.store.acquire_worker("old")
        self.store.claim("old")
        with self.assertRaises(QueueConflict):
            self.store.acquire_worker("new")
        self.store.release_worker("old")
        self.store.acquire_worker("new")
        interrupted = self.store.get(old["id"])
        self.assertEqual(interrupted["error"]["code"], "worker_interrupted")
        self.assertEqual(self.store.claim("new")["id"], queued["id"])
        self.assertFalse(self.store.heartbeat("old"))
        with self.assertRaises(QueueConflict):
            self.store.finish(old["id"], "old", {"run_id": "late"})

    def test_claims_are_exclusive_and_only_owner_can_finish(self):
        job = self.store.submit("experiment", {})
        self.store.acquire_worker("owner")
        self.assertEqual(self.store.claim("owner")["id"], job["id"])
        self.assertIsNone(self.store.claim("owner"))
        with self.assertRaises(QueueConflict):
            self.store.claim("other")
        with self.assertRaises(QueueConflict):
            self.store.finish(job["id"], "other")
        self.store.progress(job["id"], "owner", .5, "Training")
        output = self.store.finish(job["id"], "owner", {"run_id": "run"})
        self.assertEqual(output["status"], "succeeded")
        self.assertEqual(output["progress"], 1.)
        self.assertEqual(self.store.cancel(job["id"])["status"], "succeeded")

    def test_failure_does_not_hide_output(self):
        job = self.store.submit("experiment", {})
        self.store.acquire_worker("owner")
        self.store.claim("owner")
        output = self.store.finish(job["id"], "owner", {"run_id": "failed-run"},
                                   {"code": "training_failed", "message": "No usable predictions"})
        self.assertEqual(output["status"], "failed")
        self.assertEqual(output["result"]["run_id"], "failed-run")

    def test_missing_job_and_readiness(self):
        self.assertFalse(self.store.worker_ready())
        with self.assertRaises(KeyError):
            self.store.cancel("missing")
        self.store.acquire_worker("owner")
        self.assertTrue(self.store.worker_ready())
        self.store.release_worker("owner")
        self.assertFalse(self.store.worker_ready())


class SettingsAndSerializationTests(unittest.TestCase):
    def test_settings_paths_are_relative_to_settings_file(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "server.json"
            path.write_text(json.dumps({"datasets": {"ch": {"data_root": "data"}},
                                        "storage_dir": "state"}))
            settings = ServerSettings.load(path, token="secret")
            self.assertEqual(settings.datasets["ch"].data_root, (Path(folder) / "data").resolve())
            self.assertEqual(settings.storage_dir, (Path(folder) / "state").resolve())
            self.assertEqual(settings.token, "secret")

    def test_json_safe_converts_nonfinite_numbers_and_dates(self):
        from datetime import date
        self.assertEqual(json_safe({"a": float("nan"), "b": [float("inf"), date(2020, 1, 1)]}),
                         {"a": None, "b": [None, "2020-01-01"]})


if __name__ == "__main__":
    unittest.main()
