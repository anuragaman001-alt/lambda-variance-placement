# ---------------------------------------------------------------------------
# lambda_function.py  (Python 3.11 version of the test function)
#
# This is the code that actually runs INSIDE AWS Lambda.
# One file is used for all 15 Python functions. The environment variable
# WORKLOAD (set by deploy.py) decides which of the 3 tasks it does.
# The Node.js version (index.mjs) does exactly the same work, so the two
# runtimes can be compared fairly.
# ---------------------------------------------------------------------------
import os
import time


def cpu():
    # CPU-heavy task: 1 million multiplications and remainders.
    # No memory or disk use, so the time mostly depends on how much CPU
    # Lambda gives the function (which grows with the memory setting).
    s = 0
    for i in range(1_000_000):
        s += (i * i) % 7
    return s


def memory():
    # Memory-heavy task: reserve a 32 MB block of memory...
    b = bytearray(32 * 1024 * 1024)
    # ...then write one byte in every 4 KB "page". Memory is handed out in
    # pages, so touching each page forces the system to really give us all
    # 32 MB, not just promise it.
    for i in range(0, len(b), 4096):
        b[i] = 1
    return len(b)


def io_task():
    # Disk-heavy task: make 5 MB of random bytes...
    data = os.urandom(5 * 1024 * 1024)
    # ...write them to /tmp (the only folder Lambda lets us write to)...
    with open("/tmp/f.bin", "wb") as f:
        f.write(data)
    # ...and read the file back in.
    with open("/tmp/f.bin", "rb") as f:
        return len(f.read())


# Map the WORKLOAD text to the matching function above.
TASKS = {"cpu": cpu, "memory": memory, "io": io_task}


def lambda_handler(event, context):
    # Lambda calls this function every time the function is invoked.
    # Read which task to run (default "cpu" if the variable is missing).
    w = os.environ.get("WORKLOAD", "cpu")

    # Time only the task itself, as an extra check next to Lambda's own
    # "Duration" number in the logs.
    t0 = time.perf_counter()
    r = TASKS[w]()
    compute_ms = (time.perf_counter() - t0) * 1000

    # Send back a small answer. The real measurements come from the
    # Lambda REPORT log line, which run_experiment.py reads.
    return {"workload": w, "result": r, "compute_ms": compute_ms}
