"""Invoke every config: 1 forced cold start + N-1 warm calls. Writes a CSV row per call.

Usage:  python3 run_experiment.py --window W1
"""
import argparse
import base64
import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config

from config import INVOCATIONS_PER_CONFIG, REGION, RUNTIMES, all_configs

FIELDS = ["timestamp_utc", "hour_of_day", "window", "region", "function", "runtime",
          "workload", "memory_mb", "invocation_idx", "is_cold", "init_ms",
          "duration_ms", "billed_ms", "max_mem_mb", "client_ms", "error"]
REPORT_RE = {
    "duration_ms": re.compile(r"\tDuration: ([\d.]+) ms"),
    "billed_ms": re.compile(r"Billed Duration: ([\d.]+) ms"),
    "max_mem_mb": re.compile(r"Max Memory Used: (\d+) MB"),
    "init_ms": re.compile(r"Init Duration: ([\d.]+) ms"),
}

lam = boto3.client("lambda", region_name=REGION, config=Config(
    retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=120))
lock = threading.Lock()
done = 0


def parse_report(log_b64):
    log = base64.b64decode(log_b64).decode(errors="replace")
    out = {}
    for k, rx in REPORT_RE.items():
        m = rx.search(log)
        out[k] = float(m.group(1)) if m else ""
    return out


def run_config(cfg, n, window, writer, fh, total):
    global done
    name, rk, w, mem = cfg
    # Force a fresh execution environment -> next call is a cold start
    lam.update_function_configuration(FunctionName=name, Environment={
        "Variables": {"WORKLOAD": w, "COLD_TOKEN": str(time.time())}})
    lam.get_waiter("function_updated_v2").wait(FunctionName=name)

    for i in range(n):
        now = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        err = ""
        try:
            r = lam.invoke(FunctionName=name, LogType="Tail", Payload=b"{}")
            client_ms = (time.perf_counter() - t0) * 1000
            r["Payload"].read()
            err = r.get("FunctionError", "")
            rep = parse_report(r.get("LogResult", ""))
        except Exception as e:  # keep going, record the failure
            client_ms, rep, err = (time.perf_counter() - t0) * 1000, {}, type(e).__name__
        row = {"timestamp_utc": now.isoformat(), "hour_of_day": now.hour,
               "window": window, "region": REGION, "function": name,
               "runtime": RUNTIMES[rk], "workload": w, "memory_mb": mem,
               "invocation_idx": i, "is_cold": int(bool(rep.get("init_ms"))),
               "client_ms": round(client_ms, 2), "error": err,
               **{k: rep.get(k, "") for k in REPORT_RE}}
        with lock:
            writer.writerow(row)
            fh.flush()
            done += 1
            if done % 100 == 0 or done == total:
                print(f"{done}/{total} invocations done")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="W1", help="time-window label, e.g. W1/W2/W3")
    ap.add_argument("--invocations", type=int, default=INVOCATIONS_PER_CONFIG)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel configs (new accounts may have a concurrency limit of 10)")
    a = ap.parse_args()

    cfgs = list(all_configs())
    total = len(cfgs) * a.invocations
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"raw_{a.window}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
    print(f"Running {len(cfgs)} configs x {a.invocations} = {total} invocations -> {out}")

    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futs = [pool.submit(run_config, c, a.invocations, a.window, writer, fh, total)
                    for c in cfgs]
            for f in as_completed(futs):
                print("finished", f.result())
    print("Done. Results in", out)


if __name__ == "__main__":
    main()
