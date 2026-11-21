# ---------------------------------------------------------------------------
# cloud/deploy_analysis.py
# One-time setup so that the ANALYSIS also runs in AWS (not on my laptop).
# (First version used CodeBuild, but new accounts get a CodeBuild limit of 0,
#  so the analysis runs in Lambda instead, which already works for the data.)
#
# Creates in eu-north-1 (safe to run again):
#   1. the scientific libraries for Lambda (Linux, Python 3.11), zipped and
#      stored in S3: s3://<bucket>/deps/analysis-deps-py311.zip
#   2. IAM role        variance-analysis-role   (logs + my results bucket)
#   3. Lambda function variance-analysis-runner (cloud/analysis_runner.py)
#
# Run once from the repo root:  python3 cloud/deploy_analysis.py
# Needs internet on the laptop to download the libraries (about 85 MB).
# ---------------------------------------------------------------------------
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = "eu-north-1"
FN = "variance-analysis-runner"
ROLE = "variance-analysis-role"
REPO_ZIP = "https://github.com/anuragaman001-alt/lambda-variance-placement/archive/refs/heads/main.zip"
DEPS_KEY = "deps/analysis-deps-py311.zip"
LIBS = ["scikit-learn", "scipy", "matplotlib", "joblib"]
HERE = Path(__file__).resolve().parent

ACCT = boto3.client("sts").get_caller_identity()["Account"]
BUCKET = f"lv-results-{ACCT}-{REGION}"
iam = boto3.client("iam")
s3 = boto3.client("s3", region_name=REGION)
lam = boto3.client("lambda", region_name=REGION)


def upload_deps():
    """Download Linux / Python 3.11 versions of the libraries, zip, upload to S3."""
    try:
        s3.head_object(Bucket=BUCKET, Key=DEPS_KEY)
        print(f"Libraries already in s3://{BUCKET}/{DEPS_KEY}")
        return
    except ClientError:
        pass
    tmp = Path(tempfile.mkdtemp())
    print("Downloading libraries for Lambda (Linux, Python 3.11)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "--platform", "manylinux2014_x86_64", "--implementation", "cp",
                    "--python-version", "3.11", "--only-binary=:all:",
                    "--target", str(tmp / "deps"), *LIBS], check=True)
    zip_path = shutil.make_archive(str(tmp / "deps"), "zip", tmp / "deps")
    print(f"Uploading {Path(zip_path).stat().st_size / 1e6:.0f} MB to S3...")
    s3.upload_file(zip_path, BUCKET, DEPS_KEY)
    shutil.rmtree(tmp)


def ensure_role():
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
             "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": "*"},
        {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{BUCKET}"},
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": f"arn:aws:s3:::{BUCKET}/*"}]}
    try:
        iam.get_role(RoleName=ROLE)
        new = False
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust))
        new = True
    else:
        # the role may exist from the CodeBuild version: make Lambda allowed to use it
        iam.update_assume_role_policy(RoleName=ROLE, PolicyDocument=json.dumps(trust))
        new = True
    iam.put_role_policy(RoleName=ROLE, PolicyName=f"{ROLE}-policy", PolicyDocument=json.dumps(policy))
    if new:
        print("Role ready; waiting 15 s for IAM...")
        time.sleep(15)
    return iam.get_role(RoleName=ROLE)["Role"]["Arn"]


def code_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("analysis_runner.py")
        info.external_attr = 0o644 << 16
        z.writestr(info, (HERE / "analysis_runner.py").read_text())
    return buf.getvalue()


def ensure_function(role_arn):
    kw = dict(Role=role_arn, Handler="analysis_runner.handler", Runtime="python3.11",
              Timeout=900, MemorySize=3008, EphemeralStorage={"Size": 2048},
              Environment={"Variables": {"RESULTS_BUCKET": BUCKET, "DEPS_KEY": DEPS_KEY,
                                         "REPO_ZIP": REPO_ZIP}})
    try:
        lam.get_function(FunctionName=FN)
        lam.update_function_code(FunctionName=FN, ZipFile=code_zip())
        lam.get_waiter("function_updated_v2").wait(FunctionName=FN)
        lam.update_function_configuration(FunctionName=FN, **kw)
    except lam.exceptions.ResourceNotFoundException:
        for attempt in range(6):
            try:
                lam.create_function(FunctionName=FN, Code={"ZipFile": code_zip()},
                                    Architectures=["x86_64"], **kw)
                break
            except ClientError as e:
                if "cannot be assumed" in str(e) and attempt < 5:
                    time.sleep(10)
                    continue
                raise
    lam.get_waiter("function_active_v2").wait(FunctionName=FN)
    lam.get_waiter("function_updated_v2").wait(FunctionName=FN)
    print(f"Lambda function ready: {FN}")


def main():
    upload_deps()
    ensure_function(ensure_role())
    print("\nReady. Run the analysis in the console: Lambda > Functions > "
          f"{FN} > Test tab > event {{}} > Test  (takes a few minutes)")
    print(f"Results go to s3://{BUCKET}/analysis/<time>/")


if __name__ == "__main__":
    main()
