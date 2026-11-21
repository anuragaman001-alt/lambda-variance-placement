# ---------------------------------------------------------------------------
# cloud/deploy_xray.py
# One-time setup for the cloud X-Ray validation (run from the repo root):
#     python3 cloud/deploy_xray.py
#
# Creates (safe to run again):
#   1. in each of the 4 regions: 6 test functions with X-Ray switched on,
#      "lv-xray-<py|node>-<cpu|memory|io>-512". Same workloads as the main
#      functions (code taken from xray_validate.py); they also return their
#      X-Ray trace id so each call's trace can be found.
#   2. IAM role  variance-xray-role   (call and update the lv-xray-* functions,
#      read X-Ray traces, save results to S3)
#   3. Lambda    variance-xray-validator  in eu-north-1 (cloud/xray_validator.py)
#
# The 120 main functions are not changed.
# Delete the test functions later with:  python3 cloud/deploy_xray.py --cleanup
# ---------------------------------------------------------------------------
import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import ROLE_NAME                       # noqa: E402  (role of the test functions)
from xray_validate import JS_CODE, PY_CODE         # noqa: E402  (same workloads)

HOME = "eu-north-1"
REGIONS = ["eu-north-1", "eu-west-1", "us-east-1", "ap-southeast-1"]
RUNTIMES = {"python3.11": "py", "nodejs20.x": "node"}
WORKLOADS = ["cpu", "memory", "io"]
MEMORY = 512
FN = "variance-xray-validator"
ROLE = "variance-xray-role"

ACCT = boto3.client("sts").get_caller_identity()["Account"]
BUCKET = f"lv-results-{ACCT}-{HOME}"
iam = boto3.client("iam")


def zip_one(filename, text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo(filename)
        info.external_attr = 0o644 << 16
        z.writestr(info, text)
    return buf.getvalue()


def test_functions():
    """6 traced test functions per region."""
    iam.attach_role_policy(RoleName=ROLE_NAME,
                           PolicyArn="arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess")
    role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
    for region in REGIONS:
        lam = boto3.client("lambda", region_name=region)
        for rt, short in RUNTIMES.items():
            py = short == "py"
            for w in WORKLOADS:
                name = f"lv-xray-{short}-{w}-{MEMORY}"
                try:
                    lam.get_function(FunctionName=name)
                    print(f"[{region}] exists  {name}")
                    continue
                except lam.exceptions.ResourceNotFoundException:
                    pass
                for attempt in range(6):
                    try:
                        lam.create_function(
                            FunctionName=name, Runtime=rt, Role=role_arn,
                            Handler="lambda_function.lambda_handler" if py else "index.handler",
                            Code={"ZipFile": zip_one("lambda_function.py" if py else "index.mjs",
                                                     PY_CODE if py else JS_CODE)},
                            MemorySize=MEMORY, Timeout=60, Architectures=["x86_64"],
                            Environment={"Variables": {"WORKLOAD": w}},
                            TracingConfig={"Mode": "Active"})      # X-Ray on
                        break
                    except ClientError as e:
                        if "cannot be assumed" in str(e) and attempt < 5:
                            time.sleep(10)
                            continue
                        raise
                lam.get_waiter("function_active_v2").wait(FunctionName=name)
                print(f"[{region}] created {name}")


def validator_role():
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
             "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": "*"},
        {"Effect": "Allow", "Action": ["lambda:InvokeFunction", "lambda:GetFunction",
                                       "lambda:GetFunctionConfiguration",
                                       "lambda:UpdateFunctionConfiguration"],
         "Resource": f"arn:aws:lambda:*:{ACCT}:function:lv-xray-*"},
        {"Effect": "Allow", "Action": ["xray:BatchGetTraces"], "Resource": "*"},
        {"Effect": "Allow", "Action": "s3:PutObject", "Resource": f"arn:aws:s3:::{BUCKET}/results/*"}]}
    try:
        iam.get_role(RoleName=ROLE)
        new = False
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust))
        new = True
    iam.put_role_policy(RoleName=ROLE, PolicyName=f"{ROLE}-policy", PolicyDocument=json.dumps(policy))
    if new:
        print("Created role; waiting 15 s for IAM...")
        time.sleep(15)
    return iam.get_role(RoleName=ROLE)["Role"]["Arn"]


def validator(role_arn):
    lam = boto3.client("lambda", region_name=HOME)
    code = zip_one("xray_validator.py", (ROOT / "cloud" / "xray_validator.py").read_text())
    kw = dict(Role=role_arn, Handler="xray_validator.handler", Runtime="python3.11",
              Timeout=900, MemorySize=512, Environment={"Variables": {"RESULTS_BUCKET": BUCKET}})
    try:
        lam.get_function(FunctionName=FN)
        lam.update_function_code(FunctionName=FN, ZipFile=code)
        lam.get_waiter("function_updated_v2").wait(FunctionName=FN)
        lam.update_function_configuration(FunctionName=FN, **kw)
    except lam.exceptions.ResourceNotFoundException:
        for attempt in range(6):
            try:
                lam.create_function(FunctionName=FN, Code={"ZipFile": code},
                                    Architectures=["x86_64"], **kw)
                break
            except ClientError as e:
                if "cannot be assumed" in str(e) and attempt < 5:
                    time.sleep(10)
                    continue
                raise
    lam.get_waiter("function_active_v2").wait(FunctionName=FN)
    lam.get_waiter("function_updated_v2").wait(FunctionName=FN)
    print(f"Validator ready: {FN} ({HOME})")


def cleanup():
    for region in REGIONS:
        lam = boto3.client("lambda", region_name=region)
        for short in RUNTIMES.values():
            for w in WORKLOADS:
                name = f"lv-xray-{short}-{w}-{MEMORY}"
                try:
                    lam.delete_function(FunctionName=name)
                    print(f"[{region}] deleted {name}")
                except lam.exceptions.ResourceNotFoundException:
                    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleanup", action="store_true", help="delete the 24 X-Ray test functions")
    if ap.parse_args().cleanup:
        cleanup()
        return
    test_functions()
    validator(validator_role())
    print("\nReady. Run it in the console: Lambda > Functions > variance-xray-validator "
          "> Test tab > event {} > Test  (takes about 10 minutes)")
    print(f"Results go to s3://{BUCKET}/results/xray_validation_<time>.csv")


if __name__ == "__main__":
    main()
