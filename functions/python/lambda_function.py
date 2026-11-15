import os
import time


def cpu():
    s = 0
    for i in range(1_000_000):
        s += (i * i) % 7
    return s


def memory():
    b = bytearray(32 * 1024 * 1024)          # 32 MB
    for i in range(0, len(b), 4096):         # touch every page
        b[i] = 1
    return len(b)


def io_task():
    data = os.urandom(5 * 1024 * 1024)       # 5 MB
    with open("/tmp/f.bin", "wb") as f:
        f.write(data)
    with open("/tmp/f.bin", "rb") as f:
        return len(f.read())


TASKS = {"cpu": cpu, "memory": memory, "io": io_task}


def lambda_handler(event, context):
    w = os.environ.get("WORKLOAD", "cpu")
    t0 = time.perf_counter()
    r = TASKS[w]()
    return {"workload": w, "result": r, "compute_ms": (time.perf_counter() - t0) * 1000}
