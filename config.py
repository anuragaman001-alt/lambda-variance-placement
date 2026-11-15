# ---------------------------------------------------------------------------
# config.py
# All the experiment settings live in this one file, so every other script
# (deploy, run, summarize) uses exactly the same list of configurations.
# If I want to change the experiment, I only change it here.
# ---------------------------------------------------------------------------

# AWS region where the Lambda functions are created and called.
# Originally planned eu-west-1 (Ireland), but the AWS Organization policy on
# my account only allows eu-north-1 (Stockholm), so this run uses Stockholm.
REGION = "eu-north-1"

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
# 30 configurations x 200 calls = 6,000 calls per time window.
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
