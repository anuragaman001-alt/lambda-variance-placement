# ---------------------------------------------------------------------------
# cloud/deploy_cloud.py
# One-time setup of the cloud runner (run from the repo root):
#     python3 cloud/deploy_cloud.py
#     python3 cloud/deploy_cloud.py --schedule W5=2026-11-23T12:00:00 W6=2026-11-25T12:00:00
#
# Creates in eu-north-1 (safe to run again; existing parts are updated):
#   1. S3 bucket            lv-results-<account>-eu-north-1   (CSV results)
#   2. Lambda function      variance-orchestrator             (runs one region)
#   3. Step Functions       variance-window-runner            (4 regions in parallel)
#   4. optional EventBridge Scheduler one-time schedules for future windows
#      (times are Europe/Dublin local time)
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
from config import HOME_REGION, REGIONS   # noqa: E402

FN = "variance-orchestrator"
SM = "variance-window-runner"
ORCH_ROLE = "variance-orchestrator-role"
SFN_ROLE = "variance-stepfunctions-role"
SCHED_ROLE = "variance-scheduler-role"

ACCT = boto3.client("sts").get_caller_identity()["Account"]
BUCKET = f"lv-results-{ACCT}-{HOME_REGION}"
iam = boto3.client("iam")
s3 = boto3.client("s3", region_name=HOME_REGION)
lam = boto3.client("lambda", region_name=HOME_REGION)
sfn = boto3.client("stepfunctions", region_name=HOME_REGION)


def ensure_role(name, service, policy, managed=()):
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
             "Principal": {"Service": service}, "Action": "sts:AssumeRole"}]}
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        new = False
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        new = True
    for m in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=m)
    iam.put_role_policy(RoleName=name, PolicyName=f"{name}-policy", PolicyDocument=json.dumps(policy))
    if new:
        print(f"Created role {name}; waiting 15 s for IAM...")
        time.sleep(15)
    return arn


def ensure_bucket():
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": HOME_REGION})
        print(f"Created bucket {BUCKET}")


def code_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [ROOT / "cloud" / "orchestrator.py", ROOT / "run_experiment.py", ROOT / "config.py"]:
            info = zipfile.ZipInfo(p.name)
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_text())
    return buf.getvalue()


def ensure_function(role_arn):
    kw = dict(Role=role_arn, Handler="orchestrator.handler", Runtime="python3.11",
              Timeout=900, MemorySize=512, Environment={"Variables": {"RESULTS_BUCKET": BUCKET}})
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
    arn = lam.get_function(FunctionName=FN)["Configuration"]["FunctionArn"]
    print(f"Function ready: {arn}")
    return arn


def definition(fn_arn):
    return {
        "Comment": "Runs one time window: all regions in parallel, 30 configs each",
        "StartAt": "Regions",
        "States": {"Regions": {
            "Type": "Map", "ItemsPath": "$.regions", "MaxConcurrency": 4,
            "ItemSelector": {"region.$": "$$.Map.Item.Value", "window.$": "$.window",
                             "invocations.$": "$.invocations"},
            "ItemProcessor": {"ProcessorConfig": {"Mode": "INLINE"}, "StartAt": "RunRegion",
                              "States": {"RunRegion": {
                                  "Type": "Task", "Resource": "arn:aws:states:::lambda:invoke",
                                  "Parameters": {"FunctionName": fn_arn, "Payload.$": "$"},
                                  "OutputPath": "$.Payload", "TimeoutSeconds": 900,
                                  "Retry": [{"ErrorEquals": ["Lambda.ServiceException",
                                                             "Lambda.TooManyRequestsException"],
                                             "IntervalSeconds": 10, "MaxAttempts": 2}],
                                  "End": True}}},
            "End": True}},
    }


def ensure_state_machine(fn_arn, role_arn):
    defn = json.dumps(definition(fn_arn))
    for sm in sfn.list_state_machines()["stateMachines"]:
        if sm["name"] == SM:
            sfn.update_state_machine(stateMachineArn=sm["stateMachineArn"], definition=defn, roleArn=role_arn)
            print(f"Updated state machine {sm['stateMachineArn']}")
            return sm["stateMachineArn"]
    for attempt in range(6):
        try:
            arn = sfn.create_state_machine(name=SM, definition=defn, roleArn=role_arn)["stateMachineArn"]
            print(f"Created state machine {arn}")
            return arn
        except ClientError as e:
            if "AccessDenied" in str(e) and attempt < 5:
                time.sleep(10)
                continue
            raise


def window_input(window):
    return {"window": window, "invocations": 200, "regions": REGIONS}


def ensure_schedules(specs, sm_arn):
    role = ensure_role(SCHED_ROLE, "scheduler.amazonaws.com", {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "states:StartExecution", "Resource": sm_arn}]})
    sch = boto3.client("scheduler", region_name=HOME_REGION)
    for spec in specs:
        window, when = spec.split("=", 1)
        name = f"variance-{window}"
        args = dict(Name=name, ScheduleExpression=f"at({when})",
                    ScheduleExpressionTimezone="Europe/Dublin",
                    FlexibleTimeWindow={"Mode": "OFF"}, ActionAfterCompletion="DELETE",
                    Target={"Arn": sm_arn, "RoleArn": role, "Input": json.dumps(window_input(window))})
        try:
            sch.create_schedule(**args)
        except sch.exceptions.ConflictException:
            sch.update_schedule(**args)
        print(f"Scheduled {window} at {when} (Europe/Dublin)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", nargs="*", default=[],
                    help="e.g. W5=2026-11-23T12:00:00 (Europe/Dublin time)")
    a = ap.parse_args()

    ensure_bucket()
    orch_role = ensure_role(ORCH_ROLE, "lambda.amazonaws.com", {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["lambda:InvokeFunction", "lambda:GetFunction",
                                       "lambda:GetFunctionConfiguration",
                                       "lambda:UpdateFunctionConfiguration"],
         "Resource": f"arn:aws:lambda:*:{ACCT}:function:lv-*"},
        {"Effect": "Allow", "Action": "s3:PutObject", "Resource": f"arn:aws:s3:::{BUCKET}/results/*"}]},
        managed=["arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"])
    fn_arn = ensure_function(orch_role)
    sfn_role = ensure_role(SFN_ROLE, "states.amazonaws.com", {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": [fn_arn, f"{fn_arn}:*"]}]})
    sm_arn = ensure_state_machine(fn_arn, sfn_role)
    if a.schedule:
        ensure_schedules(a.schedule, sm_arn)

    print("\nReady. Start a window from the Step Functions console with this input:")
    print(json.dumps(window_input("W4")))
    print(f"Results land in s3://{BUCKET}/results/")


if __name__ == "__main__":
    main()
