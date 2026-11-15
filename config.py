# ---------------------------------------------------------------------------
# config.py
# All the experiment settings live in this one file, so every other script
# (deploy, run, summarize) uses exactly the same list of configurations.
# If I want to change the experiment, I only change it here.
# ---------------------------------------------------------------------------

# AWS regions where the 30 Lambda functions are deployed and measured.
# The proposal planned eu-west-1, us-east-1 and ap-southeast-1. My account was
# first limited to eu-north-1 (Stockholm), where the first 18,000 calls ran.
# After the region policy was opened (15 Nov), all four regions are used:
# Stockholm is kept because its data was already collected and a fourth
# region gives the GPR model more to learn about region effects.
REGIONS = ["eu-north-1", "eu-west-1", "us-east-1", "ap-southeast-1"]

# Local time zone of each region's data centres, used to work out the LOCAL
# hour of day of every call (busy hours depend on local time, not on Dublin's).
REGION_TZ = {
    "eu-north-1": "Europe/Stockholm",
    "eu-west-1": "Europe/Dublin",
    "us-east-1": "America/New_York",
    "ap-southeast-1": "Asia/Singapore",
}

# "Home" region: where the shared services live (DynamoDB results table,
# SageMaker training, X-Ray validation). Only the Lambda tests use all regions.
HOME_REGION = "eu-north-1"
REGION = HOME_REGION          # older scripts import this name

# Name of the IAM role that Lambda uses to run the functions.
# The role only gives permission to write logs to CloudWatch.
ROLE_NAME = "lambda-variance-role"

# The two runtimes I compare.
# Key = short name used inside function names, value = AWS runtime ID.
RUNTIMES = {"py": "python3.11", "node": "nodejs20.x"}

# The three types of work a function can do (see the handler code):
#   cpu    -> loop of maths, tests processor speed
#   memory -> fill a 32 MB block, tests memory speed
#   io     -> write and read a 5 MB file, tests disk speed
WORKLOADS = ["cpu", "memory", "io"]

# The five memory sizes in MB. In Lambda, more memory also means more CPU,
# so this is the main "knob" a developer can turn.
MEMORY_TIERS = [128, 256, 512, 1024, 2048]

# How many times each function is called in one run.
# Call number 0 is the cold start, calls 1..199 are warm.
# 30 configurations x 200 calls = 6,000 calls per region per time window.
INVOCATIONS_PER_CONFIG = 200


def all_configs():
    """Give back every combination of runtime x workload x memory (30 in total).

    Each item is a tuple: (function_name, runtime_key, workload, memory_mb)
    Example: ("lv-py-cpu-128", "py", "cpu", 128)
    """
    for rk in RUNTIMES:                    # 2 runtimes
        for w in WORKLOADS:                # x 3 workloads
            for m in MEMORY_TIERS:         # x 5 memory sizes = 30
                # "lv" = lambda variance, a short prefix so my functions are
                # easy to spot in the AWS console
                yield f"lv-{rk}-{w}-{m}", rk, w, m
