# ---------------------------------------------------------------------------
# summarize.py
# Reads every raw CSV in results/ and works out, for each configuration
# (region x runtime x workload x memory), how fast and how VARIABLE the warm
# calls were.
# Output: a table on screen + results/summary.csv
#
# Run with:  python3 summarize.py
# ---------------------------------------------------------------------------
import csv
import glob
import statistics as st
from collections import defaultdict

# 1. Load all rows from all runs (W1, W2, W3 ... are combined).
#    Skip failed calls and rows with no Duration.
rows = []
for f in glob.glob("results/raw_*.csv"):
    with open(f) as fh:
        rows += [r for r in csv.DictReader(fh) if not r["error"] and r["duration_ms"]]

# 2. Group by configuration. Cold and warm calls are kept apart, because
#    a cold start is a different thing and would distort the warm numbers.
groups = defaultdict(lambda: {"warm": [], "cold_init": []})
for r in rows:
    key = (r["region"], r["runtime"], r["workload"], int(r["memory_mb"]))
    if r["is_cold"] == "1":
        if r["init_ms"]:                                       # setup time
            groups[key]["cold_init"].append(float(r["init_ms"]))
    else:
        groups[key]["warm"].append(float(r["duration_ms"]))    # run time

# 3. Statistics per configuration (warm calls only).
out = []
for (rg, rt, w, m), g in sorted(groups.items()):
    d = g["warm"]
    if len(d) < 2:          # need at least 2 values for a standard deviation
        continue
    # quantiles(n=100) splits the data into 100 parts:
    # q[49] = 50th percentile (median), q[94] = P95, q[98] = P99
    q = st.quantiles(d, n=100)
    mean = st.mean(d)
    out.append({"region": rg, "runtime": rt, "workload": w, "memory_mb": m, "n_warm": len(d),
                "mean_ms": round(mean, 2),
                "std_ms": round(st.stdev(d), 2),
                # CV (coefficient of variation) = std / mean.
                # My main variance measure: it lets me compare a 10 ms function
                # with a 1500 ms function fairly.
                "cv": round(st.stdev(d) / mean, 3),
                "p50_ms": round(q[49], 2),
                "p95_ms": round(q[94], 2),
                "p99_ms": round(q[98], 2),     # tail latency, the focus of the thesis
                "cold_init_ms": round(st.mean(g["cold_init"]), 2) if g["cold_init"] else ""})

# 4. Save the summary table as CSV (used later for the GPR model).
with open("results/summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

# 5. Print a readable table on screen.
print(f"{'region':<16}{'runtime':<12}{'workload':<8}{'mem':>6}{'n':>5}{'mean':>9}{'p95':>9}{'p99':>9}{'CV':>7}{'cold':>9}")
for r in out:
    print(f"{r['region']:<16}{r['runtime']:<12}{r['workload']:<8}{r['memory_mb']:>6}{r['n_warm']:>5}"
          f"{r['mean_ms']:>9}{r['p95_ms']:>9}{r['p99_ms']:>9}{r['cv']:>7}{str(r['cold_init_ms']):>9}")
print(f"\n{len(rows)} valid invocations -> results/summary.csv")
