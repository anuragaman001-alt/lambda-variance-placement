# ---------------------------------------------------------------------------
# sagemaker_train.py
# Runs train_gpr.py as an Amazon SageMaker training job instead of on my Mac
# (as planned in the proposal architecture).
#
# What happens:
#   1. Creates an IAM role that SageMaker uses (first run only)
#   2. Uploads the raw CSVs to S3 (SageMaker's default bucket)
#   3. Starts a training job on a small ml.m5.large machine, using AWS's
#      ready-made scikit-learn container, and runs train_gpr.py there
#   4. Streams the job's output here (the same table as on my Mac)
#   5. Downloads the results (model, predictions, plot) to results/sagemaker/
#
# Cost: billed per second; a job of a few minutes on ml.m5.large is cents.
#
# Run with:  python3 sagemaker_train.py
# Needs:     pip install "sagemaker>=2.200,<3"
# ---------------------------------------------------------------------------
import glob
import json
import os
import shutil
import tarfile
import tempfile
import time

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

from config import REGION

SM_ROLE = "lv-sagemaker-role"
INSTANCE = "ml.m5.large"

boto_session = boto3.Session(region_name=REGION)
session = sagemaker.Session(boto_session=boto_session)


def ensure_role():
    """IAM role that SageMaker uses to run the job and read/write S3."""
    iam = boto3.client("iam")
    try:
        return iam.get_role(RoleName=SM_ROLE)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        trust = {"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"}]}
        arn = iam.create_role(RoleName=SM_ROLE,
                              AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
        # AWS-managed policy: run training jobs + access buckets whose name
        # contains "sagemaker" (the default bucket below does).
        iam.attach_role_policy(RoleName=SM_ROLE, PolicyArn=(
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"))
        print(f"Created role {arn}; waiting 15 s for IAM to propagate...")
        time.sleep(15)
        return arn


def main():
    role = ensure_role()

    # 1. Upload the data. default_bucket() = sagemaker-<region>-<account id>,
    #    created automatically the first time.
    bucket = session.default_bucket()
    files = sorted(glob.glob("results/raw_*.csv"))
    for f in files:
        session.upload_data(f, bucket=bucket, key_prefix="lambda-variance/data")
    data_uri = f"s3://{bucket}/lambda-variance/data"
    print(f"Uploaded {len(files)} CSV files to {data_uri}")

    # 2. Package only what the job needs: the training script + its extra
    #    library (matplotlib is not in AWS's scikit-learn container).
    src = tempfile.mkdtemp()
    shutil.copy("train_gpr.py", src)
    with open(os.path.join(src, "requirements.txt"), "w") as fh:
        fh.write("matplotlib\n")

    # 3. Define and start the training job.
    est = SKLearn(entry_point="train_gpr.py", source_dir=src,
                  framework_version="1.4-2",          # scikit-learn version
                  py_version="py3",
                  instance_type=INSTANCE, instance_count=1,
                  role=role, sagemaker_session=session,
                  base_job_name="lv-gpr",
                  output_path=f"s3://{bucket}/lambda-variance/output")
    est.fit({"train": data_uri}, wait=True, logs="All")   # shows the job's output here

    # 4. Download the results: SageMaker packs SM_MODEL_DIR into model.tar.gz.
    out = "results/sagemaker"
    os.makedirs(out, exist_ok=True)
    tar_path = os.path.join(out, "model.tar.gz")
    b, key = est.model_data.replace("s3://", "").split("/", 1)
    boto_session.client("s3").download_file(b, key, tar_path)
    with tarfile.open(tar_path) as t:
        t.extractall(out)
    os.remove(tar_path)
    print(f"\nTraining job : {est.latest_training_job.name}")
    print(f"Results in   : {out}/  ({', '.join(sorted(os.listdir(out)))})")


if __name__ == "__main__":
    main()
