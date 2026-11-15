"""Per-config statistics (mean, std, CV, P50/P95/P99) from all raw CSVs in results/."""
import csv
import glob
import statistics as st
from collections import defaultdict

rows = []
for f in glob.glob("results/raw_*.csv"):
    with open(f) as fh:
        rows += [r for r in csv.DictReader(fh) if not r["error"] and r["duration_ms"]]

groups = defaultdict(lambda: {"warm": [], "cold_init": []})
for r in rows:
    key = (r["runtime"], r["workload"], int(r["memory_mb"]))
    if r["is_cold"] == "1":
        groups[key]["cold_init"].append(float(r["init_ms"]))
    else:
        groups[key]["warm"].append(float(r["duration_ms"]))

out = []
for (rt, w, m), g in sorted(groups.items()):
    d = g["warm"]
    if len(d) < 2:
        continue
    q = st.quantiles(d, n=100)
    mean = st.mean(d)
    out.append({"runtime": rt, "workload": w, "memory_mb": m, "n_warm": len(d),
                "mean_ms": round(mean, 2), "std_ms": round(st.stdev(d), 2),
                "cv": round(st.stdev(d) / mean, 3), "p50_ms": round(q[49], 2),
                "p95_ms": round(q[94], 2), "p99_ms": round(q[98], 2),
                "cold_init_ms": round(st.mean(g["cold_init"]), 2) if g["cold_init"] else ""})

with open("results/summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

print(f"{'runtime':<12}{'workload':<8}{'mem':>6}{'n':>5}{'mean':>9}{'p95':>9}{'p99':>9}{'CV':>7}{'cold':>9}")
for r in out:
    print(f"{r['runtime']:<12}{r['workload']:<8}{r['memory_mb']:>6}{r['n_warm']:>5}"
          f"{r['mean_ms']:>9}{r['p95_ms']:>9}{r['p99_ms']:>9}{r['cv']:>7}{str(r['cold_init_ms']):>9}")
print(f"\n{len(rows)} valid invocations -> results/summary.csv")
