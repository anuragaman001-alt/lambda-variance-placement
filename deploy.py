"""Create the IAM role and the 30 Lambda functions (safe to re-run)."""
import io
import json
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from config import REGION, ROLE_NAME, RUNTIMES, all_configs

HERE = Path(__file__).parent
SOURCES = {
    "py": ("lambda_function.lambda_handler", HERE / "functions/python/lambda_function.py"),
    "node": ("index.handler", HERE / "functions/node/index.mjs"),
}


def make_zip(path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo(path.name)
        info.external_attr = 0o644 << 16
        z.writestr(info, path.read_text())
    return buf.getvalue()


def ensure_role():
    iam = boto3.client("iam")
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"}]}
    try:
        arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        print(f"Role exists: {arn}")
        return arn
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=ROLE_NAME,
                              AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=(
            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"))
        print(f"Created role {arn}; waiting 15 s for IAM to propagate...")
        time.sleep(15)
        return arn


def main():
    role_arn = ensure_role()
    lam = boto3.client("lambda", region_name=REGION)
    zips = {k: make_zip(p) for k, (_, p) in SOURCES.items()}

    for name, rk, w, mem in all_configs():
        handler = SOURCES[rk][0]
        try:
            lam.get_function(FunctionName=name)
            lam.update_function_code(FunctionName=name, ZipFile=zips[rk])
            lam.get_waiter("function_updated_v2").wait(FunctionName=name)
            lam.update_function_configuration(
                FunctionName=name, MemorySize=mem, Timeout=60,
                Environment={"Variables": {"WORKLOAD": w}})
            print(f"updated  {name}")
        except lam.exceptions.ResourceNotFoundException:
            for attempt in range(6):      # role may still be propagating
                try:
                    lam.create_function(
                        FunctionName=name, Runtime=RUNTIMES[rk], Role=role_arn,
                        Handler=handler, Code={"ZipFile": zips[rk]},
                        MemorySize=mem, Timeout=60, Architectures=["x86_64"],
                        Environment={"Variables": {"WORKLOAD": w}})
                    break
                except ClientError as e:
                    if "cannot be assumed" in str(e) and attempt < 5:
                        time.sleep(10)
                        continue
                    raise
            print(f"created  {name}")
        lam.get_waiter("function_active_v2").wait(FunctionName=name)
    print("\nAll 30 functions ready in", REGION)


if __name__ == "__main__":
    main()
