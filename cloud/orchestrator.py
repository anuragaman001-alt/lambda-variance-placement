# ---------------------------------------------------------------------------
# cloud/orchestrator.py
# The experiment runner, running INSIDE AWS as a Lambda function
# ("variance-orchestrator" in eu-north-1). It reuses run_experiment.py
# unchanged, so every call is measured exactly as in the earlier windows.
#
# One invocation = one region x 30 configs x N calls. Step Functions starts
# one invocation per region in parallel (see deploy_cloud.py), and each one
# saves its CSV to S3:  s3://<bucket>/results/raw_<window>_<region>_<time>.csv
#
# Event example: {"window": "W4", "region": "eu-west-1", "invocations": 200}
#                window "TEST" -> saved as test_*.csv (never used in analysis)
# ---------------------------------------------------------------------------
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import boto3

import run_experiment as rx
from config import REGION_TZ, all_configs

BUCKET = os.environ["RESULTS_BUCKET"]
s3 = boto3.client("s3")

# Lambda's Python image may not ship the time-zone database. If it is missing,
# fall back to fixed offsets (valid Nov 2026 - Mar 2027, i.e. no DST changes).
try:
    from zoneinfo import ZoneInfo
    for _tz in REGION_TZ.values():
        ZoneInfo(_tz)
except Exception:
    _OFFSET = {"Europe/Stockholm": 1, "Europe/Dublin": 0,
               "America/New_York": -5, "Asia/Singapore": 8}
    rx.ZoneInfo = lambda name: timezone(timedelta(hours=_OFFSET[name]))


def handler(event, context):
    region = event["region"]
    window = event.get("window", "W1")
    n = int(event.get("invocations", 200))
    workers = int(event.get("workers", 8))      # same as the earlier runs

    rx.clients[region] = boto3.client("lambda", region_name=region, config=rx.CLIENT_CFG)
    rx.done = 0
    cfgs = list(all_configs())
    total = len(cfgs) * n

    prefix = "test" if window == "TEST" else "raw"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"/tmp/{prefix}_{window}_{region}_{stamp}.csv"

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rx.FIELDS)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(rx.run_config, region, c, n, window, writer, fh, total)
                    for c in cfgs]
            for f in futs:
                f.result()          # re-raise if a whole config failed

    key = f"results/{os.path.basename(path)}"
    s3.upload_file(path, BUCKET, key)

    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {"region": region, "window": window, "rows": len(rows),
            "cold_starts": sum(int(r["is_cold"]) for r in rows),
            "errors": sum(1 for r in rows if r["error"]),
            "s3_key": key,
            "seconds_left": context.get_remaining_time_in_millis() // 1000}
