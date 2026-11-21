# ---------------------------------------------------------------------------
# placement.py  -  Case Study 1: SLO-based placement
#
# Question: for a function with a P99 latency target (SLO), which region and
# memory size should it use? Four methods choose, then each choice is judged
# against what was REALLY measured.
#
# Fair setup (same as Experiment 2): every method only "knows" the benchmark
# results for 128, 512 and 2048 MB. The 256 and 1024 MB results are hidden
# from all methods and used only to judge their choices. So the GPR has to
# predict P99 for setups it never saw, exactly what a real user would need.
#
# Methods
#   1. AWS default        128 MB in eu-west-1, no tuning
#   2. Power-Tuning-style eu-west-1 only; tries the measured memory sizes and
#                         picks the best balance of AVERAGE speed and cost
#                         (an emulation of AWS Lambda Power Tuning "balanced")
#   3. Mean-based         all 4 regions x 5 memory sizes; uses the AVERAGE run
#                         time instead of P99: cheapest option whose average
#                         meets the SLO (memory sizes not benchmarked are
#                         estimated by a straight line on a log-log scale)
#   4. GPR (proposed)     all 4 regions x 5 memory sizes; uses the GPR's
#                         predicted P99: cheapest option whose predicted P99
#                         meets the SLO
#   4b. GPR (cautious)    as 4, but uses the TOP of the GPR's 95% interval
#                         (sensitivity check: safer, maybe more expensive)
#
# Scenarios: 6 functions (2 runtimes x 3 workloads) x 4 P99 targets
#            (50, 100, 200, 500 ms) = 24 scenarios per method.
#
# Input : results/raw_*.csv, results/gpr_model.joblib (from train_gpr.py)
# Output: results/placement_choices.csv  (every scenario, every method)
#         results/placement_summary.csv  (one line per method)
#
# Run with:  python3 placement.py
# ---------------------------------------------------------------------------
import csv
import glob
from collections import defaultdict

import joblib
import numpy as np

TRAIN_MEM = [128, 512, 2048]              # memory sizes every method may "see"
ALL_MEM = [128, 256, 512, 1024, 2048]
SLOS_MS = [50, 100, 200, 500]             # P99 targets
DEFAULT = ("eu-west-1", 128)              # AWS default: 128 MB, home region

# AWS Lambda x86 prices, first tier (USD). The US East price is the public
# list price; CHECK the other regions on https://aws.amazon.com/lambda/pricing/
# (choose the region in the price table) and update before final results.
PRICE = {   # region: (price per GB-second, price per request)
    "us-east-1": (0.0000166667, 0.20 / 1e6),
    "eu-west-1": (0.0000166667, 0.20 / 1e6),
    "eu-north-1": (0.0000166667, 0.20 / 1e6),
    "ap-southeast-1": (0.0000166667, 0.20 / 1e6),
}

# ---------------------------------------------------------------------------
# 1. Measured truth for every configuration (all windows, warm calls only)
# ---------------------------------------------------------------------------
dur = defaultdict(list)                   # run time (ms) of each warm call
billed = defaultdict(list)                # billed time (ms), for the real cost
for f in glob.glob("results/raw_*.csv"):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if r["error"] or not r["duration_ms"] or r["is_cold"] == "1":
                continue
            k = (r["region"], r["runtime"], r["workload"], int(r["memory_mb"]))
            dur[k].append(float(r["duration_ms"]))
            billed[k].append(float(r["billed_ms"]))

REGIONS = sorted({k[0] for k in dur})
RUNTIMES = sorted({k[1] for k in dur})
WORKLOADS = ["cpu", "memory", "io"]
p99 = {k: np.percentile(v, 99, method="weibull") for k, v in dur.items()}
mean = {k: float(np.mean(v)) for k, v in dur.items()}
mean_billed = {k: float(np.mean(v)) for k, v in billed.items()}


def cost_per_million(region, mem_mb, billed_ms):
    """Cost (USD) of one million calls = compute (GB-seconds) + requests."""
    gbs, req = PRICE[region]
    return 1e6 * (mem_mb / 1024 * billed_ms / 1000 * gbs + req)


def loglog_estimate(values, region, runtime, workload, mem_mb):
    """Estimate a value at any memory size from the 3 benchmarked sizes:
    a straight line through log(value) vs log2(memory). For benchmarked
    sizes it returns the measured value itself."""
    if mem_mb in TRAIN_MEM:
        return values[(region, runtime, workload, mem_mb)]
    xs = np.log2(TRAIN_MEM)
    ys = np.log([values[(region, runtime, workload, m)] for m in TRAIN_MEM])
    b, a = np.polyfit(xs, ys, 1)
    return float(np.exp(a + b * np.log2(mem_mb)))


# ---------------------------------------------------------------------------
# 2. The trained GPR (trained WITHOUT 256 and 1024 MB, see train_gpr.py)
# ---------------------------------------------------------------------------
saved = joblib.load("results/gpr_model.joblib")
gpr, G_REG, G_RT, G_WL = saved["model"], saved["regions"], saved["runtimes"], saved["workloads"]


def features(region, runtime, workload, memory_mb):     # same as train_gpr.py
    return ([1.0 if region == g else 0.0 for g in G_REG]
            + [1.0 if runtime == r else 0.0 for r in G_RT]
            + [1.0 if workload == w else 0.0 for w in G_WL]
            + [np.log2(memory_mb)])


def gpr_p99(region, runtime, workload, mem_mb):
    """Predicted P99 (ms) and the top of its 95% interval."""
    m, s = gpr.predict(np.array([features(region, runtime, workload, mem_mb)]), return_std=True)
    return float(np.exp(m[0])), float(np.exp(m[0] + 1.96 * s[0]))


# ---------------------------------------------------------------------------
# 3. The four methods. Each returns (region, memory, estimated latency used)
# ---------------------------------------------------------------------------
def choose_cheapest(cands, slo):
    """cands = list of (region, mem, est_latency_ms, est_cost).
    Cheapest candidate whose estimate meets the SLO; if none does, the one
    with the lowest estimate (and flag that the SLO looked impossible)."""
    ok = [c for c in cands if c[2] <= slo]
    if ok:
        return min(ok, key=lambda c: (c[3], c[2])), False
    return min(cands, key=lambda c: c[2]), True


def m_default(rt, wl, slo):
    return (*DEFAULT, None), False


def m_power_tuning(rt, wl, slo):
    # Power Tuning "balanced": normalise average time and average cost of
    # each tested memory size to 0..1 and pick the lowest sum. Region fixed.
    region = DEFAULT[0]
    t = np.array([mean[(region, rt, wl, m)] for m in TRAIN_MEM])
    c = np.array([cost_per_million(region, m, mean_billed[(region, rt, wl, m)]) for m in TRAIN_MEM])
    norm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-12)
    i = int(np.argmin(norm(t) + norm(c)))
    return (region, TRAIN_MEM[i], float(t[i])), False


def m_mean_based(rt, wl, slo):
    cands = []
    for g in REGIONS:
        for m in ALL_MEM:
            est = loglog_estimate(mean, g, rt, wl, m)
            bill = loglog_estimate(mean_billed, g, rt, wl, m)
            cands.append((g, m, est, cost_per_million(g, m, bill)))
    (g, m, est, _), infeasible = choose_cheapest(cands, slo)
    return (g, m, est), infeasible


def m_gpr(rt, wl, slo, cautious=False):
    cands = []
    for g in REGIONS:
        for m in ALL_MEM:
            pred, upper = gpr_p99(g, rt, wl, m)
            bill = loglog_estimate(mean_billed, g, rt, wl, m)   # expected billed time
            cands.append((g, m, upper if cautious else pred, cost_per_million(g, m, bill)))
    (g, m, est, _), infeasible = choose_cheapest(cands, slo)
    return (g, m, est), infeasible


METHODS = [("AWS default", m_default),
           ("Power-Tuning-style", m_power_tuning),
           ("Mean-based", m_mean_based),
           ("GPR (proposed)", m_gpr),
           ("GPR (cautious)", lambda rt, wl, slo: m_gpr(rt, wl, slo, cautious=True))]

# ---------------------------------------------------------------------------
# 4. Run all scenarios and judge each choice with the MEASURED truth
# ---------------------------------------------------------------------------
# The best possible choice (an "oracle" that knows the real P99 of every
# option) is reported too, to show how far each method is from ideal.
choices = []
for rt in RUNTIMES:
    for wl in WORKLOADS:
        for slo in SLOS_MS:
            truth = [(g, m, p99[(g, rt, wl, m)],
                      cost_per_million(g, m, mean_billed[(g, rt, wl, m)]))
                     for g in REGIONS for m in ALL_MEM]
            (og, om, _, _), o_inf = choose_cheapest(truth, slo)
            o_cost = cost_per_million(og, om, mean_billed[(og, rt, wl, om)])
            for name, fn in METHODS:
                (g, m, est), infeasible = fn(rt, wl, slo)
                k = (g, rt, wl, m)
                real = p99[k]
                cost = cost_per_million(g, m, mean_billed[k])
                choices.append({
                    "method": name, "runtime": rt, "workload": wl, "slo_ms": slo,
                    "region": g, "memory_mb": m,
                    "estimate_ms": "" if est is None else round(est, 1),
                    "real_p99_ms": round(real, 1),
                    "slo_met": int(real <= slo),
                    "cost_per_million_usd": round(cost, 4),
                    "oracle_choice": f"{og}/{om}", "oracle_cost_usd": round(o_cost, 4),
                    "oracle_possible": int(not o_inf),
                    "method_said_impossible": int(infeasible)})

with open("results/placement_choices.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(choices[0]))
    w.writeheader()
    w.writerows(choices)

# ---------------------------------------------------------------------------
# 5. Summary per method
#   SLO met        scenarios where the real P99 of the choice is under the SLO
#                  (only scenarios where meeting it was possible at all)
#   cost vs best   average cost of the choice / cost of the oracle's choice,
#                  over scenarios where both met the SLO (1.00 = cheapest possible)
#   P99 reduction  how much lower the real P99 is than the AWS default, average %
# ---------------------------------------------------------------------------
print(f"{'method':<22}{'SLO met':>10}{'rate':>8}{'cost vs best':>14}{'P99 vs default':>16}")
summary = []
for name, _ in METHODS:
    rows = [c for c in choices if c["method"] == name and c["oracle_possible"]]
    met = sum(c["slo_met"] for c in rows)
    both = [c for c in rows if c["slo_met"]]
    ratio = np.mean([c["cost_per_million_usd"] / c["oracle_cost_usd"] for c in both]) if both else float("nan")
    red = []
    for c in rows:
        d = next(x for x in choices if x["method"] == "AWS default" and x["runtime"] == c["runtime"]
                 and x["workload"] == c["workload"] and x["slo_ms"] == c["slo_ms"])
        red.append((d["real_p99_ms"] - c["real_p99_ms"]) / d["real_p99_ms"] * 100)
    summary.append({"method": name, "scenarios": len(rows), "slo_met": met,
                    "compliance_pct": round(met / len(rows) * 100, 1),
                    "cost_vs_cheapest_compliant": round(float(ratio), 3),
                    "p99_reduction_vs_default_pct": round(float(np.mean(red)), 1)})
    print(f"{name:<22}{f'{met}/{len(rows)}':>10}{met / len(rows) * 100:>7.0f}%"
          f"{ratio:>13.2f}x{np.mean(red):>15.1f}%")

with open("results/placement_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)

impossible = sum(1 for c in choices if c["method"] == "AWS default" and not c["oracle_possible"])
print(f"\n{impossible} scenario(s) left out: no option at all met that SLO in the real data.")
print("Saved results/placement_choices.csv and results/placement_summary.csv")
