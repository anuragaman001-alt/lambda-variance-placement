"""Shared experiment configuration (one-region slice of the 90-config design)."""
REGION = "eu-north-1"              # Ireland
ROLE_NAME = "lambda-variance-role"
RUNTIMES = {"py": "python3.11", "node": "nodejs20.x"}
WORKLOADS = ["cpu", "memory", "io"]
MEMORY_TIERS = [128, 256, 512, 1024, 2048]
INVOCATIONS_PER_CONFIG = 200      # 1 cold + 199 warm  -> 30 configs x 200 = 6,000


def all_configs():
    """Yield (function_name, runtime_key, workload, memory_mb) for all 30 configs."""
    for rk in RUNTIMES:
        for w in WORKLOADS:
            for m in MEMORY_TIERS:
                yield f"lv-{rk}-{w}-{m}", rk, w, m
