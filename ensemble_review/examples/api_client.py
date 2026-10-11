"""Submit an API request, poll its job, and download paginated predictions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument("--idempotency-key", default=None, help="Reuse this value when retrying a submission")
    args = parser.parse_args()
    key = args.idempotency_key or uuid.uuid4().hex
    print(f"Idempotency key: {key}", flush=True)
    base = args.url.rstrip("/") + "/api/v1"
    def call(method, path, payload=None, idempotency_key=None):
        headers = {"Authorization": "Bearer " + os.environ.get("RUNOFF_API_TOKEN", "")}
        data = None
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        request = Request(base + path, data, headers, method=method)
        try:
            with urlopen(request, timeout=120) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(error.read().decode()) from error
    body = json.loads(args.request.read_text())
    call("POST", "/experiments/validate", body)
    job = call("POST", "/experiments", body, key)
    try:
        while job["status"] in {"queued", "running"}:
            print(f"{job['status']}: {job['progress']:.0%} {job['message']}", flush=True)
            time.sleep(1)
            job = call("GET", f"/jobs/{job['job_id']}")
    except KeyboardInterrupt:
        call("POST", f"/jobs/{job['job_id']}/cancel")
        raise
    if job["status"] != "succeeded":
        raise RuntimeError(json.dumps(job))
    run_id = job["result"]["run_id"]
    print(json.dumps(call("GET", f"/runs/{run_id}"), indent=2))
    rows = []
    while True:
        page = call("GET", f"/runs/{run_id}/predictions?offset={len(rows)}&limit=10000")
        rows.extend(page["items"])
        if len(rows) >= page["total"]:
            break
        if not page["items"]:
            raise RuntimeError("Prediction pagination made no progress")
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False))
    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
