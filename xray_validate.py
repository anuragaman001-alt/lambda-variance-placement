# ---------------------------------------------------------------------------
# xray_validate.py
# Checks that my cold-start detection is correct, using AWS X-Ray.
#
# In the main experiment I detect a cold start when Lambda's REPORT log line
# contains "Init Duration". The proposal planned AWS X-Ray for this. Here I
# compare the two methods on the same calls:
#   - Lambda log says cold?   (Init Duration present)
#   - X-Ray says cold?        (trace has an "Initialization" subsegment)
# If they agree on every call, the Init Duration method is validated.
#
# Why separate functions: X-Ray tracing adds a little overhead to each call.
# If I switched it on for the 30 main functions, later runs would not be
# comparable with earlier ones. So this test uses 4 extra functions
# (names start with "lv-xray-") and the main experiment stays untouched.
#
# Run with:  python3 xray_validate.py            (about 5 minutes)
#            python3 xray_validate.py --cleanup  (delete the 4 test functions)
# Output:    results/xray_validation.csv
# ---------------------------------------------------------------------------
import argparse
import base64
import csv
import io
import json
import re
import time
import zipfile

import boto3
from botocore.config import Config

from config import REGION, ROLE_NAME

CYCLES = 5          # cold starts per function
WARM_PER_CYCLE = 4  # warm calls after each cold start
# (runtime, workload, memory) of the 4 test functions: both runtimes covered
TEST_CONFIGS = [("python3.11", "cpu", 256), ("nodejs20.x", "cpu", 256),
                ("python3.11", "io", 512), ("nodejs20.x", "memory", 512)]

# Small handlers: same workloads as the main functions, but they also return
# the X-Ray trace id (Lambda puts it in the _X_AMZN_TRACE_ID variable), so I
# can find each call's trace afterwards.
PY_CODE = '''
import os
def lambda_handler(event, context):
    w = os.environ.get("WORKLOAD", "cpu")
    if w == "cpu":
        s = 0
        for i in range(1_000_000):
            s += (i * i) % 7
    elif w == "memory":
        b = bytearray(32 * 1024 * 1024)
        for i in range(0, len(b), 4096):
            b[i] = 1
    else:
        with open("/tmp/f.bin", "wb") as f:
            f.write(os.urandom(5 * 1024 * 1024))
        with open("/tmp/f.bin", "rb") as f:
            f.read()
    return {"trace": os.environ.get("_X_AMZN_TRACE_ID", "")}
'''
JS_CODE = '''
import fs from "fs";
import crypto from "crypto";
export const handler = async () => {
  const w = process.env.WORKLOAD || "cpu";
  if (w === "cpu") { let s = 0; for (let i = 0; i < 1000000; i++) s += (i * i) % 7; }
  else if (w === "memory") { const b = Buffer.alloc(32 * 1024 * 1024); for (let i = 0; i < b.length; i += 4096) b[i] = 1; }
  else { fs.writeFileSync("/tmp/f.bin", crypto.randomBytes(5 * 1024 * 1024)); fs.readFileSync("/tmp/f.bin"); }
  return { trace: process.env._X_AMZN_TRACE_ID || "" };
};
'''

lam = boto3.client("lambda", region_name=REGION,
                   config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))
xray = boto3.client("xray", region_name=REGION)


def fn_name(rt, w, m):
    return f"lv-xray-{'py' if rt.startswith('python') else 'node'}-{w}-{m}"


def make_zip(filename, code):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo(filename)
        info.external_attr = 0o644 << 16
        z.writestr(info, code)
    return buf.getvalue()


def setup():
    """Give the role permission to send traces, then create the 4 test functions."""
    iam = boto3.client("iam")
    iam.attach_role_policy(RoleName=ROLE_NAME,
                           PolicyArn="arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess")
    role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
    time.sleep(10)                                   # let IAM catch up

    for rt, w, m in TEST_CONFIGS:
        name = fn_name(rt, w, m)
        py = rt.startswith("python")
        try:
            lam.get_function(FunctionName=name)
            print(f"exists   {name}")
        except lam.exceptions.ResourceNotFoundException:
            lam.create_function(
                FunctionName=name, Runtime=rt, Role=role_arn,
                Handler="lambda_function.lambda_handler" if py else "index.handler",
                Code={"ZipFile": make_zip("lambda_function.py" if py else "index.mjs",
                                          PY_CODE if py else JS_CODE)},
                MemorySize=m, Timeout=60, Architectures=["x86_64"],
                Environment={"Variables": {"WORKLOAD": w}},
                TracingConfig={"Mode": "Active"})    # <- X-Ray switched on
            print(f"created  {name}")
        lam.get_waiter("function_active_v2").wait(FunctionName=name)


def invoke_all():
    """Cold start + warm calls, CYCLES times per function. Returns one dict per call."""
    calls = []
    for rt, w, m in TEST_CONFIGS:
        name = fn_name(rt, w, m)
        for c in range(CYCLES):
            # force a cold start (same trick as run_experiment.py)
            lam.update_function_configuration(FunctionName=name, Environment={
                "Variables": {"WORKLOAD": w, "COLD_TOKEN": str(time.time())}})
            lam.get_waiter("function_updated_v2").wait(FunctionName=name)
            for i in range(1 + WARM_PER_CYCLE):
                r = lam.invoke(FunctionName=name, LogType="Tail", Payload=b"{}")
                log = base64.b64decode(r["LogResult"]).decode(errors="replace")
                init = re.search(r"Init Duration: ([\d.]+) ms", log)
                trace = json.loads(r["Payload"].read()).get("trace", "")
                root = re.search(r"Root=([^;]+)", trace)
                calls.append({"function": name, "cycle": c, "call": i,
                              "log_cold": int(bool(init)),
                              "log_init_ms": float(init.group(1)) if init else "",
                              "trace_id": root.group(1) if root else "",
                              "sampled": int("Sampled=1" in trace)})
                # X-Ray samples about 1 request per second, so I space the
                # calls out a little to get every call traced.
                time.sleep(1.1)
        print(f"invoked  {name}")
    return calls


def find_init(doc):
    """Search an X-Ray segment (and its subsegments) for the Initialization step.
    Returns its duration in ms, or None if there is no Initialization."""
    for sub in doc.get("subsegments", []):
        if sub.get("name", "").lower().startswith("init"):
            return (sub["end_time"] - sub["start_time"]) * 1000
        found = find_init(sub)
        if found is not None:
            return found
    return None


def fetch_traces(calls):
    """Download the X-Ray trace of every sampled call and check for Initialization."""
    pending = {c["trace_id"]: c for c in calls if c["trace_id"] and c["sampled"]}
    for c in pending.values():
        c["xray_found"] = 0
    deadline = time.time() + 180          # traces can take up to a minute or two
    while pending and time.time() < deadline:
        time.sleep(15)
        ids = list(pending)
        for i in range(0, len(ids), 5):   # the API takes 5 trace ids at a time
            for t in xray.batch_get_traces(TraceIds=ids[i:i + 5])["Traces"]:
                c = pending.get(t["Id"])
                if not c:
                    continue
                docs = [json.loads(s["Document"]) for s in t["Segments"]]
                # the function's own segment = the one with origin AWS::Lambda::Function
                fn_docs = [d for d in docs if d.get("origin") == "AWS::Lambda::Function"]
                if not fn_docs:
                    continue              # trace not complete yet, try again later
                init = None
                for d in fn_docs:
                    init = find_init(d) if init is None else init
                c.update(xray_found=1, xray_cold=int(init is not None),
                         xray_init_ms=round(init, 2) if init is not None else "")
                pending.pop(t["Id"])
        print(f"traces still pending: {len(pending)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleanup", action="store_true", help="delete the test functions")
    a = ap.parse_args()
    if a.cleanup:
        for cfg in TEST_CONFIGS:
            lam.delete_function(FunctionName=fn_name(*cfg))
            print("deleted", fn_name(*cfg))
        return

    setup()
    calls = invoke_all()
    fetch_traces(calls)

    cols = ["function", "cycle", "call", "log_cold", "log_init_ms", "trace_id",
            "sampled", "xray_found", "xray_cold", "xray_init_ms"]
    with open("results/xray_validation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(calls)

    # --- summary ---------------------------------------------------------
    checked = [c for c in calls if c.get("xray_found")]
    agree = [c for c in checked if c["log_cold"] == c["xray_cold"]]
    colds = [c for c in checked if c["log_cold"] and c["xray_cold"]]
    print(f"\nCalls made            : {len(calls)}")
    print(f"Calls with X-Ray trace: {len(checked)}")
    print(f"Agreement (cold/warm) : {len(agree)}/{len(checked)}")
    print(f"Cold starts (both)    : {len(colds)} "
          f"(expected {len(TEST_CONFIGS) * CYCLES})")
    if colds:
        diffs = [abs(c["log_init_ms"] - c["xray_init_ms"]) for c in colds]
        print(f"Init time, log vs X-Ray: mean difference {sum(diffs) / len(diffs):.1f} ms")
    print("Details: results/xray_validation.csv")


if __name__ == "__main__":
    main()
