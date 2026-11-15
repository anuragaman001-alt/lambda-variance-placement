# ---------------------------------------------------------------------------
# run_experiment.py
# The main experiment. For each of the 30 functions:
#   1. force a cold start (so the first call starts a brand-new container)
#   2. call the function 200 times in a row (1 cold + 199 warm)
#   3. save the timings of every call as one row in a CSV file
#
# Run with:  python3 run_experiment.py --window W1
# (W1/W2/W3 = the three time-of-day windows in my experimental design)
# ---------------------------------------------------------------------------
import argparse
import base64
import csv
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config

from config import INVOCATIONS_PER_CONFIG, REGION, RUNTIMES, all_configs

# Columns of the output CSV, one row per Lambda call.
FIELDS = ["timestamp_utc", "hour_of_day", "window", "region", "function", "runtime",
          "workload", "memory_mb", "invocation_idx", "is_cold", "init_ms",
          "duration_ms", "billed_ms", "max_mem_mb", "client_ms", "error"]

# After every call, Lambda writes a line like this in its log:
#   REPORT RequestId: ... Duration: 812.40 ms  Billed Duration: 813 ms
#   Memory Size: 128 MB  Max Memory Used: 45 MB  Init Duration: 160.21 ms
# These patterns pull each number out of that line.
# "Init Duration" only appears on a cold start, which is how I detect cold starts.
REPORT_RE = {
    "duration_ms": re.compile(r"\tDuration: ([\d.]+) ms"),
    "billed_ms": re.compile(r"Billed Duration: ([\d.]+) ms"),
    "max_mem_mb": re.compile(r"Max Memory Used: (\d+) MB"),
    "init_ms": re.compile(r"Init Duration: ([\d.]+) ms"),
}

# Lambda client. "adaptive" retries = if AWS says "too many requests"
# (new accounts have low limits), boto3 waits and tries again by itself.
lam = boto3.client("lambda", region_name=REGION, config=Config(
    retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=120))

# Several threads write to the same CSV file, so a lock makes sure only one
# thread writes at a time (otherwise rows could get mixed up).
lock = threading.Lock()
done = 0   # counter for the progress messages


def parse_report(log_b64):
    """Lambda returns the last 4 KB of the log, base64-encoded.
    Decode it and pull out Duration, Billed Duration, Max Memory and Init Duration."""
    log = base64.b64decode(log_b64).decode(errors="replace")
    out = {}
    for k, rx in REPORT_RE.items():
        m = rx.search(log)
        out[k] = float(m.group(1)) if m else ""   # empty if not found
    return out


def run_config(cfg, n, window, writer, fh, total):
    """Run one configuration: force a cold start, then call it n times."""
    global done
    name, rk, w, mem = cfg

    # --- Step 1: force a cold start ---------------------------------------
    # Changing any setting makes Lambda throw away the old containers.
    # I add a dummy COLD_TOKEN with the current time, so the value is always
    # new. The next call then has to start a fresh container = cold start.
    lam.update_function_configuration(FunctionName=name, Environment={
        "Variables": {"WORKLOAD": w, "COLD_TOKEN": str(time.time())}})
    lam.get_waiter("function_updated_v2").wait(FunctionName=name)

    # --- Step 2: call the function n times, one after another --------------
    # Calls go one at a time on purpose: the same warm container is reused,
    # so call 0 is cold and calls 1..n-1 are warm.
    for i in range(n):
        now = datetime.now(timezone.utc)
        t0 = time.perf_counter()        # start my own stopwatch
        err = ""
        try:
            # LogType="Tail" asks Lambda to send back the log, which holds
            # the REPORT line with Lambda's own timings.
            r = lam.invoke(FunctionName=name, LogType="Tail", Payload=b"{}")
            # client_ms = full round trip from my Mac, including network.
            client_ms = (time.perf_counter() - t0) * 1000
            r["Payload"].read()                 # read the answer (not used)
            err = r.get("FunctionError", "")    # set if the code crashed
            rep = parse_report(r.get("LogResult", ""))
        except Exception as e:
            # If a call fails, record the error and carry on with the rest.
            client_ms, rep, err = (time.perf_counter() - t0) * 1000, {}, type(e).__name__

        # --- Step 3: build one CSV row for this call -----------------------
        row = {"timestamp_utc": now.isoformat(),
               "hour_of_day": now.hour,             # GPR input feature
               "window": window, "region": REGION, "function": name,
               "runtime": RUNTIMES[rk], "workload": w, "memory_mb": mem,
               "invocation_idx": i,
               # cold start = Lambda reported an Init Duration
               "is_cold": int(bool(rep.get("init_ms"))),
               "client_ms": round(client_ms, 2), "error": err,
               **{k: rep.get(k, "") for k in REPORT_RE}}

        with lock:                      # one thread writes at a time
            writer.writerow(row)
            fh.flush()                  # save to disk now, so a crash loses nothing
            done += 1
            if done % 100 == 0 or done == total:
                print(f"{done}/{total} invocations done")
    return name


def main():
    # Command-line options, e.g. --window W2 --workers 4
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="W1", help="time-window label, e.g. W1/W2/W3")
    ap.add_argument("--invocations", type=int, default=INVOCATIONS_PER_CONFIG)
    ap.add_argument("--workers", type=int, default=8,
                    help="how many configs run at the same time "
                         "(new accounts may only allow 10 at once)")
    a = ap.parse_args()

    cfgs = list(all_configs())
    total = len(cfgs) * a.invocations                 # 30 x 200 = 6,000

    # New CSV file per run, named with the window and UTC time,
    # e.g. results/raw_W1_20261115T151117Z.csv  -> runs never overwrite each other.
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"raw_{a.window}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
    print(f"Running {len(cfgs)} configs x {a.invocations} = {total} invocations -> {out}")

    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        # Run several configurations in parallel (8 by default) to save time.
        # Inside one configuration the calls are still one after another.
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futs = [pool.submit(run_config, c, a.invocations, a.window, writer, fh, total)
                    for c in cfgs]
            for f in as_completed(futs):
                print("finished", f.result())
    print("Done. Results in", out)


if __name__ == "__main__":
    main()
