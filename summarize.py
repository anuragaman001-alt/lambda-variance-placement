# ---------------------------------------------------------------------------
# summarize.py
# Reads every raw CSV in results/ and works out, for each configuration
# (region x runtime x workload x memory), how fast and how VARIABLE the warm
# calls were.
#
# Run with:  python3 summarize.py               -> all windows combined
#                                                  (results/summary.csv)
#            python3 summarize.py --by-window   -> each time window separately,
#                                                  side by side, plus the configs
#                                                  whose P99 changes most between
#                                                  windows (results/summary_by_window.csv)
# ---------------------------------------------------------------------------
import argparse
import csv
import glob
import re
import statistics as st
from collections import defaultdict


def window_group(label):
    """Repeat runs belong to the same window: W1, W1r2, W1r3 -> W1."""
    m = re.match(r"(W\d+)", label)
    return m.group(1) if m else label


def stats(d, cold):
    """Mean, std, CV and P50/P95/P99 of a list of warm durations (ms)."""
    # quantiles(n=100) splits the data into 100 parts:
    # q[49] = 50th percentile (median), q[94] = P95, q[98] = P99
    q = st.quantiles(d, n=100)
    mean = st.mean(d)
    return {"n_warm": len(d),
            "mean_ms": round(mean, 2),
            "std_ms": round(st.stdev(d), 2),
            # CV (coefficient of variation) = std / mean.
            # My main variance measure: it lets me compare a 10 ms function
            # with a 1500 ms function fairly.
            "cv": round(st.stdev(d) / mean, 3),
            "p50_ms": round(q[49], 2),
            "p95_ms": round(q[94], 2),
            "p99_ms": round(q[98], 2),        # tail latency, the focus of the thesis
            "cold_init_ms": round(st.mean(cold), 2) if cold else ""}


# 1. Load all rows from all runs (test_*.csv files are NOT included).
#    Skip failed calls and rows with no Duration.
rows = []
for f in glob.glob("results/raw_*.csv"):
    with open(f) as fh:
        rows += [r for r in csv.DictReader(fh) if not r["error"] and r["duration_ms"]]

ap = argparse.ArgumentParser()
ap.add_argument("--by-window", action="store_true",
                help="compare time windows side by side instead of combining them")
args = ap.parse_args()

# 2. Group by configuration (and by window if asked). Cold and warm calls are
#    kept apart, because a cold start would distort the warm numbers.
groups = defaultdict(lambda: {"warm": [], "cold_init": []})
for r in rows:
    key = (r["region"], r["runtime"], r["workload"], int(r["memory_mb"]))
    if args.by_window:
        key += (window_group(r["window"]),)
    if r["is_cold"] == "1":
        if r["init_ms"]:
            groups[key]["cold_init"].append(float(r["init_ms"]))   # setup time
    else:
        groups[key]["warm"].append(float(r["duration_ms"]))        # run time

summary = {k: stats(g["warm"], g["cold_init"]) for k, g in groups.items()
           if len(g["warm"]) >= 2}       # need 2+ values for a standard deviation

if not args.by_window:
    # ----- all windows combined (as before) ----------------------------------
    out = [{"region": k[0], "runtime": k[1], "workload": k[2], "memory_mb": k[3], **v}
           for k, v in sorted(summary.items())]
    with open("results/summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"{'region':<16}{'runtime':<12}{'workload':<8}{'mem':>6}{'n':>5}"
          f"{'mean':>9}{'p95':>9}{'p99':>9}{'CV':>7}{'cold':>9}")
    for r in out:
        print(f"{r['region']:<16}{r['runtime']:<12}{r['workload']:<8}{r['memory_mb']:>6}"
              f"{r['n_warm']:>5}{r['mean_ms']:>9}{r['p95_ms']:>9}{r['p99_ms']:>9}"
              f"{r['cv']:>7}{str(r['cold_init_ms']):>9}")
    print(f"\n{len(rows)} valid invocations -> results/summary.csv")

else:
    # ----- one column per window, side by side --------------------------------
    windows = sorted({k[4] for k in summary})
    configs = sorted({k[:4] for k in summary})

    out = []
    for c in configs:
        row = {"region": c[0], "runtime": c[1], "workload": c[2], "memory_mb": c[3]}
        p99s = []
        for wdw in windows:
            s = summary.get(c + (wdw,))
            row[f"{wdw}_n"] = s["n_warm"] if s else ""
            row[f"{wdw}_mean_ms"] = s["mean_ms"] if s else ""
            row[f"{wdw}_p99_ms"] = s["p99_ms"] if s else ""
            row[f"{wdw}_cv"] = s["cv"] if s else ""
            if s:
                p99s.append(s["p99_ms"])
        # How much the tail moves between windows: largest P99 / smallest P99.
        # 1.0 = identical in every window; 2.0 = the tail doubles at some time.
        row["p99_ratio_max_min"] = round(max(p99s) / min(p99s), 2) if len(p99s) >= 2 else ""
        out.append(row)

    with open("results/summary_by_window.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    # Table: P99 and CV per window for every config
    head = f"{'region':<16}{'runtime':<12}{'workload':<8}{'mem':>6}"
    for wdw in windows:
        head += f"{wdw + ' p99':>11}{wdw + ' CV':>8}"
    print(head + f"{'ratio':>7}")
    for r in out:
        line = f"{r['region']:<16}{r['runtime']:<12}{r['workload']:<8}{r['memory_mb']:>6}"
        for wdw in windows:
            line += f"{str(r[f'{wdw}_p99_ms']):>11}{str(r[f'{wdw}_cv']):>8}"
        print(line + f"{str(r['p99_ratio_max_min']):>7}")

    # The configs whose tail latency changes most with the time of day
    ranked = sorted([r for r in out if r["p99_ratio_max_min"] != ""],
                    key=lambda r: r["p99_ratio_max_min"], reverse=True)
    print(f"\nWindows found: {', '.join(windows)}")
    print("Biggest P99 change between windows (top 10):")
    for r in ranked[:10]:
        vals = ", ".join(f"{wdw} {r[f'{wdw}_p99_ms']}" for wdw in windows
                         if r[f"{wdw}_p99_ms"] != "")
        print(f"  {r['p99_ratio_max_min']:>5}x  {r['region']} {r['runtime']} "
              f"{r['workload']} {r['memory_mb']} MB  ({vals})")
    print(f"\n{len(rows)} valid invocations -> results/summary_by_window.csv")
