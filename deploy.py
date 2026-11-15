# ---------------------------------------------------------------------------
# deploy.py
# Creates everything the experiment needs in AWS:
#   1. one IAM role (permission for Lambda to run and write logs)
#   2. 30 Lambda functions, one per configuration in config.py
# Safe to run again: if a function already exists it is updated, not duplicated.
#
# Run with:  python3 deploy.py
# ---------------------------------------------------------------------------
import io
import json
import time
import zipfile
from pathlib import Path

import boto3                                   # AWS SDK for Python
from botocore.exceptions import ClientError    # error type boto3 raises

from config import REGION, ROLE_NAME, RUNTIMES, all_configs

# Folder this script is in, so the paths work from anywhere.
HERE = Path(__file__).parent

# For each runtime: (handler name Lambda should call, path to the code file).
# "lambda_function.lambda_handler" means: file lambda_function.py,
# function lambda_handler inside it.
SOURCES = {
    "py": ("lambda_function.lambda_handler", HERE / "functions/python/lambda_function.py"),
    "node": ("index.handler", HERE / "functions/node/index.mjs"),
}


def make_zip(path):
    """Lambda needs the code uploaded as a .zip file.
    This builds the zip in memory (no file saved on disk) and returns its bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo(path.name)
        # Give the file normal read permissions (rw-r--r--), otherwise Lambda
        # may not be able to read it.
        info.external_attr = 0o644 << 16
        z.writestr(info, path.read_text())
    return buf.getvalue()


def ensure_role():
    """Return the IAM role ARN, creating the role first if it does not exist."""
    iam = boto3.client("iam")

    # "Trust policy": says that the Lambda service is allowed to use this role.
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"}]}

    try:
        # Role already there (e.g. second run) -> just reuse it.
        arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        print(f"Role exists: {arn}")
        return arn
    except iam.exceptions.NoSuchEntityException:
        # Role not found -> create it.
        arn = iam.create_role(RoleName=ROLE_NAME,
                              AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        # Attach AWS's ready-made policy that only allows writing CloudWatch logs.
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=(
            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"))
        # New IAM roles take a few seconds to become usable everywhere in AWS.
        print(f"Created role {arn}; waiting 15 s for IAM to propagate...")
        time.sleep(15)
        return arn


def main():
    role_arn = ensure_role()
    lam = boto3.client("lambda", region_name=REGION)

    # Build the two zip files once (one Python, one Node) and reuse them.
    zips = {k: make_zip(p) for k, (_, p) in SOURCES.items()}

    for name, rk, w, mem in all_configs():
        handler = SOURCES[rk][0]
        try:
            # Does the function already exist? If yes -> update it.
            lam.get_function(FunctionName=name)
            lam.update_function_code(FunctionName=name, ZipFile=zips[rk])
            # Wait until the code update is finished before changing settings,
            # otherwise AWS refuses a second update at the same time.
            lam.get_waiter("function_updated_v2").wait(FunctionName=name)
            lam.update_function_configuration(
                FunctionName=name, MemorySize=mem, Timeout=60,
                Environment={"Variables": {"WORKLOAD": w}})
            print(f"updated  {name}")
        except lam.exceptions.ResourceNotFoundException:
            # Function does not exist -> create it.
            # Retry up to 6 times because a brand-new role can still be
            # "not ready" for a few seconds.
            for attempt in range(6):
                try:
                    lam.create_function(
                        FunctionName=name,
                        Runtime=RUNTIMES[rk],         # python3.11 or nodejs20.x
                        Role=role_arn,
                        Handler=handler,
                        Code={"ZipFile": zips[rk]},
                        MemorySize=mem,               # 128 ... 2048 MB
                        Timeout=60,                   # seconds before Lambda stops it
                        Architectures=["x86_64"],     # same CPU type for all tests
                        Environment={"Variables": {"WORKLOAD": w}})
                    break
                except ClientError as e:
                    if "cannot be assumed" in str(e) and attempt < 5:
                        time.sleep(10)
                        continue
                    raise
            print(f"created  {name}")

        # Wait until the function is "Active" (ready to be called).
        lam.get_waiter("function_active_v2").wait(FunctionName=name)

    print("\nAll 30 functions ready in", REGION)


if __name__ == "__main__":
    main()
