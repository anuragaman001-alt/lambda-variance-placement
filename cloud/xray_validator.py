# ---------------------------------------------------------------------------
# cloud/xray_validator.py
# Validation of the cold-start flag with AWS X-Ray, run INSIDE AWS as a Lambda
# function ("variance-xray-validator" in eu-north-1).
#
# The main experiment marks a call as a cold start when Lambda's log report
# contains "Init Duration". This function checks that flag against X-Ray on
# the same calls:
#   - log says cold?    (Init Duration present in the log report)
#   - X-Ray says cold?  (the trace has an "Initialization" subsegment)
#
# It covers all 4 regions and all 6 runtime x workload pairs, using separate
# test functions with X-Ray switched on ("lv-xray-..."), so the 120 main
# functions are never traced and stay comparable with earlier windows.
# Regions run in parallel; inside a region the calls run one at a time.
#
# Event (all keys optional): {"cycles": 5, "warm": 4}
#   cycles = cold starts per function, warm = warm calls after each one
# Output: s3://<bucket>/results/xray_validation_<UTC time>.csv  + a summary
# ---------------------------------------------------------------------------
import base64
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

BUCKET = os.environ["RESULTS_BUCKET"]
REGIONS = ["eu-north-1", "eu-west-1", "us-east-1", "ap-southeast-1"]
RUNTIMES = {"python3.11": "py", "nodejs20.x": "node"}
WORKLOADS = ["cpu", "memory", "io"]
MEMORY = 512
CFG = Config(retries={"max_attempts": 10, "mode": "adaptive"}, read_timeout=120)
FIELDS = ["region", "function", "runtime", "workload", "cycle", "call",
          "log_cold", "log_init_ms", "trace_id", "sampled",
          "xray_found", "xray_cold", "xray_init_ms"]


def fn_name(runtime, workload):
    return f"lv-xray-{RUNTIMES[runtime]}-{workload}-{MEMORY}"


def find_init(doc):
    """Duration (ms) of the Initialization subsegment of an X-Ray segment, or None."""
    for sub in doc.get("subsegments", []):
        if sub.get("name", "").lower().startswith("init"):
            return (sub["end_time"] - sub["start_time"]) * 1000
        found = find_init(sub)
        if found is not None:
            return found
    return None


def invoke_region(region, cycles, warm):
    """Force a cold start, then make warm calls, for each of the 6 functions."""
    lam = boto3.client("lambda", region_name=region, config=CFG)
    calls = []
    for rt in RUNTIMES:
        for w in WORKLOADS:
            name = fn_name(rt, w)
            for c in range(cycles):
                # same cold-start trick as the main experiment
                lam.update_function_configuration(FunctionName=name, Environment={
                    "Variables": {"WORKLOAD": w, "COLD_TOKEN": str(time.time())}})
                lam.get_waiter("function_updated_v2").wait(FunctionName=name)
                for i in range(1 + warm):
                    r = lam.invoke(FunctionName=name, LogType="Tail", Payload=b"{}")
                    log = base64.b64decode(r["LogResult"]).decode(errors="replace")
                    init = re.search(r"Init Duration: ([\d.]+) ms", log)
                    trace = json.loads(r["Payload"].read()).get("trace", "")
                    root = re.search(r"Root=([^;]+)", trace)
                    calls.append({"region": region, "function": name, "runtime": rt,
                                  "workload": w, "cycle": c, "call": i,
                                  "log_cold": int(bool(init)),
                                  "log_init_ms": float(init.group(1)) if init else "",
                                  "trace_id": root.group(1) if root else "",
                                  "sampled": int("Sampled=1" in trace)})
                    time.sleep(1.1)        # X-Ray samples about 1 request per second
    return calls


def fetch_traces(region, calls, deadline):
    """Download each sampled call's trace and look for an Initialization step."""
    xray = boto3.client("xray", region_name=region, config=CFG)
    pending = {c["trace_id"]: c for c in calls if c["trace_id"] and c["sampled"]}
    for c in pending.values():
        c["xray_found"] = 0
    while pending and time.time() < deadline:
        time.sleep(15)
        ids = list(pending)
        for i in range(0, len(ids), 5):            # 5 trace ids per request
            for t in xray.batch_get_traces(TraceIds=ids[i:i + 5])["Traces"]:
                c = pending.get(t["Id"])
                if not c:
                    continue
                docs = [json.loads(s["Document"]) for s in t["Segments"]]
                fn_docs = [d for d in docs if d.get("origin") == "AWS::Lambda::Function"]
                if not fn_docs:
                    continue                       # trace not complete yet
                init = None
                for d in fn_docs:
                    init = find_init(d) if init is None else init
                c.update(xray_found=1, xray_cold=int(init is not None),
                         xray_init_ms=round(init, 2) if init is not None else "")
                pending.pop(t["Id"])


def handler(event, context):
    # Remove the inherited trace header (Sampled=0) so each test function
    # makes its own X-Ray sampling decision, as when called from a laptop.
    os.environ.pop("_X_AMZN_TRACE_ID", None)
    cycles = int(event.get("cycles", 5))
    warm = int(event.get("warm", 4))
    with ThreadPoolExecutor(max_workers=len(REGIONS)) as pool:
        per_region = list(pool.map(lambda g: invoke_region(g, cycles, warm), REGIONS))

    # leave 60 s at the end of the 15-minute limit for saving
    deadline = time.time() + max(60, context.get_remaining_time_in_millis() / 1000 - 60)
    deadline = min(deadline, time.time() + 240)
    with ThreadPoolExecutor(max_workers=len(REGIONS)) as pool:
        list(pool.map(lambda gc: fetch_traces(gc[0], gc[1], deadline), zip(REGIONS, per_region)))

    calls = [c for rc in per_region for c in rc]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = f"/tmp/xray_validation_{stamp}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(calls)
    key = f"results/xray_validation_{stamp}.csv"
    boto3.client("s3").upload_file(path, BUCKET, key)

    summary = {}
    for g in REGIONS + ["ALL"]:
        rows = [c for c in calls if g == "ALL" or c["region"] == g]
        traced = [c for c in rows if c.get("xray_found")]
        both_cold = [c for c in traced if c["log_cold"] and c["xray_cold"]]
        diffs = [abs(c["log_init_ms"] - c["xray_init_ms"]) for c in both_cold]
        summary[g] = {"calls": len(rows), "traced": len(traced),
                      "agree": sum(1 for c in traced if c["log_cold"] == c["xray_cold"]),
                      "cold_starts_forced": len({(c["region"], c["function"], c["cycle"]) for c in rows}),
                      "cold_both": len(both_cold),
                      "init_ms_mean_abs_diff": round(sum(diffs) / len(diffs), 3) if diffs else None}
    return {"results": f"s3://{BUCKET}/{key}", "summary": summary}
