# ---------------------------------------------------------------------------
# cloud/deploy_analysis.py
# One-time setup so that the ANALYSIS also runs in AWS (not on my laptop).
#
# Creates in eu-north-1 (safe to run again):
#   - IAM role        variance-analysis-role   (logs + read/write my results bucket)
#   - CodeBuild       variance-analysis        (a cloud machine that runs the scripts)
#
# Each build (started from the CodeBuild console with "Start build"):
#   1. clones this repository from GitHub
#   2. downloads the cloud runs from s3://<bucket>/results/
#   3. runs train_gpr.py, stats_tests.py, placement.py, case_study2.py
#   4. uploads every result (CSV, PNG, model) to
#      s3://<bucket>/analysis/<build number>/
#
# Run once from the repo root:  python3 cloud/deploy_analysis.py
# ---------------------------------------------------------------------------
import json
import time

import boto3

REGION = "eu-north-1"
PROJECT = "variance-analysis"
ROLE = "variance-analysis-role"
REPO = "https://github.com/anuragaman001-alt/lambda-variance-placement.git"

ACCT = boto3.client("sts").get_caller_identity()["Account"]
BUCKET = f"lv-results-{ACCT}-{REGION}"
iam = boto3.client("iam")
cb = boto3.client("codebuild", region_name=REGION)

BUILDSPEC = f"""version: 0.2
phases:
  install:
    runtime-versions:
      python: 3.11
    commands:
      - pip install --quiet scikit-learn scipy matplotlib joblib
  build:
    commands:
      - git clone --depth 1 {REPO} repo
      - cd repo
      - aws s3 sync s3://{BUCKET}/results/ results/ --exclude "test_*"
      - ls results/raw_*.csv
      - python3 train_gpr.py
      - python3 stats_tests.py
      - python3 placement.py
      - python3 case_study2.py
  post_build:
    commands:
      - cd $CODEBUILD_SRC_DIR/repo
      - aws s3 cp results/ s3://{BUCKET}/analysis/$CODEBUILD_BUILD_NUMBER/ --recursive --exclude "*" --include "*.png" --include "*.joblib" --include "summary*.csv" --include "gpr_*.csv" --include "model_*.csv" --include "stats_*.csv" --include "placement_*.csv" --include "case_study*.csv"
      - echo "Results in s3://{BUCKET}/analysis/$CODEBUILD_BUILD_NUMBER/"
"""


def ensure_role():
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
             "Principal": {"Service": "codebuild.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": "*"},
        {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{BUCKET}"},
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": f"arn:aws:s3:::{BUCKET}/*"}]}
    try:
        arn = iam.get_role(RoleName=ROLE)["Role"]["Arn"]
        new = False
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        new = True
    iam.put_role_policy(RoleName=ROLE, PolicyName=f"{ROLE}-policy", PolicyDocument=json.dumps(policy))
    if new:
        print("Created role; waiting 15 s for IAM...")
        time.sleep(15)
    return arn


def main():
    role = ensure_role()
    args = dict(
        name=PROJECT,
        description="Runs the GPR training, statistics and case studies in AWS",
        source={"type": "NO_SOURCE", "buildspec": BUILDSPEC},
        artifacts={"type": "NO_ARTIFACTS"},
        environment={"type": "LINUX_CONTAINER",
                     "image": "aws/codebuild/amazonlinux2-x86_64-standard:5.0",
                     "computeType": "BUILD_GENERAL1_SMALL"},
        serviceRole=role, timeoutInMinutes=30,
        logsConfig={"cloudWatchLogs": {"status": "ENABLED"}})
    for attempt in range(6):
        try:
            if cb.batch_get_projects(names=[PROJECT])["projects"]:
                cb.update_project(**args)
                print(f"Updated CodeBuild project {PROJECT}")
            else:
                cb.create_project(**args)
                print(f"Created CodeBuild project {PROJECT}")
            break
        except cb.exceptions.InvalidInputException as e:
            if "not authorized to perform" in str(e) or "role" in str(e).lower():
                time.sleep(10)
                continue
            raise
    print("\nReady. Start it in the console: CodeBuild > Build projects > "
          f"{PROJECT} > Start build")
    print(f"Results will be uploaded to s3://{BUCKET}/analysis/<build number>/")


if __name__ == "__main__":
    main()
