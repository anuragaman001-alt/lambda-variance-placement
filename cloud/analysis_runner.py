# ---------------------------------------------------------------------------
# cloud/analysis_runner.py
# Runs the whole ANALYSIS inside AWS, as a Lambda function
# ("variance-analysis-runner" in eu-north-1).
#
# What one run does:
#   1. downloads the scientific libraries (scikit-learn, scipy, matplotlib)
#      from S3 into /tmp. They are too big for a normal Lambda upload, so
#      deploy_analysis.py stores them in S3 once.
#   2. downloads the latest code from GitHub (main branch)
#   3. downloads the cloud runs from s3://<bucket>/results/ (test runs skipped)
#   4. runs train_gpr.py, stats_tests.py, placement.py, case_study2.py
#   5. uploads every result (tables, plots, model, full log) to
#      s3://<bucket>/analysis/<UTC time>/
#
# Start it from the Lambda console: Test tab -> event {} -> Test.
# ---------------------------------------------------------------------------
import glob
import io
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

import boto3

BUCKET = os.environ["RESULTS_BUCKET"]
DEPS_KEY = os.environ["DEPS_KEY"]
REPO_ZIP = os.environ["REPO_ZIP"]
SCRIPTS = ["train_gpr.py", "stats_tests.py", "placement.py", "case_study2.py"]
OUTPUTS = ["*.png", "*.joblib", "summary*.csv", "gpr_*.csv", "model_*.csv",
           "stats_*.csv", "placement_*.csv", "case_study*.csv"]
DEPS_DIR = "/tmp/deps"
s3 = boto3.client("s3")


def ensure_deps():
    """Download and unpack the libraries once per Lambda container."""
    if os.path.isdir(DEPS_DIR):
        return
    s3.download_file(BUCKET, DEPS_KEY, "/tmp/deps.zip")
    with zipfile.ZipFile("/tmp/deps.zip") as z:
        z.extractall(DEPS_DIR)
    os.remove("/tmp/deps.zip")


def get_code(work):
    """Download the repository from GitHub and return its folder."""
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    with urllib.request.urlopen(REPO_ZIP, timeout=120) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(work)
    return glob.glob(os.path.join(work, "*"))[0]


def get_cloud_runs(repo):
    """Copy the cloud runs from S3 into the repo's results/ folder."""
    os.makedirs(os.path.join(repo, "results"), exist_ok=True)
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="results/"):
        for obj in page.get("Contents", []):
            name = os.path.basename(obj["Key"])
            if name.startswith("raw_") and name.endswith(".csv"):
                s3.download_file(BUCKET, obj["Key"], os.path.join(repo, "results", name))
                n += 1
    return n


def handler(event, context):
    t0 = time.time()
    ensure_deps()
    repo = get_code("/tmp/work")
    n = get_cloud_runs(repo)
    data_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(repo, "results", "raw_*.csv")))
    log = [f"Cloud runs downloaded from S3: {n}",
           f"Data files used: {', '.join(data_files)}", ""]

    env = dict(os.environ, PYTHONPATH=DEPS_DIR, MPLCONFIGDIR="/tmp/mpl", MPLBACKEND="Agg")
    status = "Succeeded"
    for script in SCRIPTS:
        r = subprocess.run([sys.executable, script], cwd=repo, env=env,
                           capture_output=True, text=True)
        log.append(f"===== {script} =====\n{r.stdout}")
        print(f"===== {script} =====\n{r.stdout}")      # also visible in CloudWatch
        if r.returncode != 0:
            log.append(f"ERROR in {script}:\n{r.stderr[-4000:]}")
            print(r.stderr[-4000:])
            status = f"Failed in {script}"
            break

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    prefix = f"analysis/{stamp}/"
    uploaded = 0
    for pattern in OUTPUTS:
        for p in glob.glob(os.path.join(repo, "results", pattern)):
            s3.upload_file(p, BUCKET, prefix + os.path.basename(p))
            uploaded += 1
    s3.put_object(Bucket=BUCKET, Key=prefix + "analysis_log.txt", Body="\n".join(log).encode())

    return {"status": status, "results": f"s3://{BUCKET}/{prefix}",
            "files_uploaded": uploaded + 1, "data_files": len(data_files),
            "seconds": round(time.time() - t0)}
