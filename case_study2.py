# ---------------------------------------------------------------------------
# case_study2.py  -  Case Study 2: does the model still work on another day?
#
# AWS can change its machines at any time, so a model trained today might be
# wrong next week. This script tests that with the time windows already
# collected:
#   - take the windows that cover ALL regions (W2, W3, W4, ... ; W1 is left
#     out because it only covered eu-north-1)
#   - train the GPR on all windows except one, test on the window left out
#   - repeat so that every window is the "unseen day" once
#
# Training uses only 128/512/2048 MB (same as Experiment 2), so the test
# window checks both a new day AND memory sizes the model never saw.
#
# For each unseen window it reports:
#   model : MAPE and 95% interval coverage of the predicted P99 against the
#           P99 really measured in that window
#   placement : how often the setups chosen from the OLD windows still meet
#           the SLO in the NEW window (GPR, GPR cautious, mean-based)
#
# Input : results/raw_*.csv            Output: results/case_study2.csv
# Run with:  python3 case_study2.py
# ---------------------------------------------------------------------------
import csv
import glob
import warnings
from collections import defaultdict

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

warnings.filterwarnings("ignore", category=ConvergenceWarning)
rng = np.random.default_rng(42)

TRAIN_MEM = [128, 512, 2048]
ALL_MEM = [128, 256, 512, 1024, 2048]
SLOS_MS = [50, 100, 200, 500]
RUNTIMES = ["nodejs20.x", "python3.11"]
WORKLOADS = ["cpu", "memory", "io"]
PRICE_GBS, PRICE_REQ = 0.0000166667, 0.20 / 1e6     # same as placement.py

# ---------------------------------------------------------------------------
# 1. Load warm calls, keeping the window label of every call
# ---------------------------------------------------------------------------
dur = defaultdict(list)           # (window, region, runtime, workload, mem) -> ms
billed = defaultdict(list)
for f in glob.glob("results/raw_*.csv"):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if r["error"] or not r["duration_ms"] or r["is_cold"] == "1":
                continue
            k = (r["window"], r["region"], r["runtime"], r["workload"], int(r["memory_mb"]))
            dur[k].append(float(r["duration_ms"]))
            billed[k].append(float(r["billed_ms"]))

REGIONS = sorted({k[1] for k in dur})
regions_per_window = defaultdict(set)
for k in dur:
    regions_per_window[k[0]].add(k[1])
WINDOWS = sorted(w for w, rg in regions_per_window.items() if rg == set(REGIONS))
print("Windows used (all regions present):", ", ".join(WINDOWS))
if len(WINDOWS) < 2:
    raise SystemExit("Need at least 2 full windows.")


def pool(windows, region, runtime, workload, mem):
    """All warm run times of one config across the given windows."""
    out = []
    for w in windows:
        out += dur.get((w, region, runtime, workload, mem), [])
    return np.array(out)


def pool_billed(windows, region, runtime, workload, mem):
    out = []
    for w in windows:
        out += billed.get((w, region, runtime, workload, mem), [])
    return float(np.mean(out))


def features(region, runtime, workload, mem):            # same as train_gpr.py
    return ([1.0 if region == g else 0.0 for g in REGIONS]
            + [1.0 if runtime == r else 0.0 for r in RUNTIMES]
            + [1.0 if workload == w else 0.0 for w in WORKLOADS]
            + [np.log2(mem)])


def log_p99_and_boot_var(d):
    """log P99 of one config and its bootstrap variance (as in train_gpr.py)."""
    lp = np.log(np.percentile(d, 99, method="weibull"))
    s = rng.choice(d, size=(1000, len(d)), replace=True)
    return lp, np.log(np.percentile(s, 99, axis=1, method="weibull")).var()


def train(windows):
    """Train the same GPR as train_gpr.py on the given windows, 128/512/2048 MB."""
    X, y, a = [], [], []
    for g in REGIONS:
        for rt in RUNTIMES:
            for wl in WORKLOADS:
                for m in TRAIN_MEM:
                    lp, bv = log_p99_and_boot_var(pool(windows, g, rt, wl, m))
                    X.append(features(g, rt, wl, m)); y.append(lp); a.append(bv)
    X = np.array(X)
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e0)))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=np.array(a), normalize_y=True,
                                   n_restarts_optimizer=10, random_state=42)
    return gpr.fit(X, np.array(y))


def cost(mem, billed_ms):
    return 1e6 * (mem / 1024 * billed_ms / 1000 * PRICE_GBS + PRICE_REQ)


def loglog(vals, mem):
    """Value at any memory size from the 3 benchmarked sizes (log-log line)."""
    if mem in TRAIN_MEM:
        return vals[mem]
    b, a = np.polyfit(np.log2(TRAIN_MEM), np.log([vals[m] for m in TRAIN_MEM]), 1)
    return float(np.exp(a + b * np.log2(mem)))


def choose(cands, slo):
    """Cheapest candidate whose estimate meets the SLO, else lowest estimate."""
    ok = [c for c in cands if c[2] <= slo]
    return min(ok, key=lambda c: (c[3], c[2])) if ok else min(cands, key=lambda c: c[2])


# ---------------------------------------------------------------------------
# 2. Leave one window out, train on the rest, test on it
# ---------------------------------------------------------------------------
rows = []
print(f"\n{'unseen':<8}{'model MAPE':>11}{'coverage':>11}   "
      f"{'GPR':>8}{'GPR caut.':>11}{'mean-based':>12}   (SLO met in the unseen window)")
for test_w in WINDOWS:
    train_w = [w for w in WINDOWS if w != test_w]
    gpr = train(train_w)

    # --- model accuracy against the unseen window, all 120 configs -------
    errs, inside = [], []
    pred = {}                          # (g, rt, wl, m) -> (p99, upper)
    for g in REGIONS:
        for rt in RUNTIMES:
            for wl in WORKLOADS:
                for m in ALL_MEM:
                    mu, sd = gpr.predict(np.array([features(g, rt, wl, m)]), return_std=True)
                    real_lp, bv = log_p99_and_boot_var(pool([test_w], g, rt, wl, m))
                    tot = np.sqrt(sd[0] ** 2 + bv)
                    pred[(g, rt, wl, m)] = (np.exp(mu[0]), np.exp(mu[0] + 1.96 * sd[0]))
                    errs.append(abs(np.exp(mu[0]) - np.exp(real_lp)) / np.exp(real_lp) * 100)
                    inside.append(abs(real_lp - mu[0]) <= 1.96 * tot)
    mape, cov = np.mean(errs), np.mean(inside) * 100

    # --- placement: choose from old windows, judge on the unseen window --
    met = {"GPR": [0, 0], "GPR cautious": [0, 0], "Mean-based": [0, 0]}
    for rt in RUNTIMES:
        for wl in WORKLOADS:
            real = {(g, m): np.percentile(pool([test_w], g, rt, wl, m), 99, method="weibull")
                    for g in REGIONS for m in ALL_MEM}
            for slo in SLOS_MS:
                if min(real.values()) > slo:       # impossible in that window
                    continue
                cands = {"GPR": [], "GPR cautious": [], "Mean-based": []}
                for g in REGIONS:
                    means = {m: float(np.mean(pool(train_w, g, rt, wl, m))) for m in TRAIN_MEM}
                    bills = {m: pool_billed(train_w, g, rt, wl, m) for m in TRAIN_MEM}
                    for m in ALL_MEM:
                        c = cost(m, loglog(bills, m))
                        p, up = pred[(g, rt, wl, m)]
                        cands["GPR"].append((g, m, p, c))
                        cands["GPR cautious"].append((g, m, up, c))
                        cands["Mean-based"].append((g, m, loglog(means, m), c))
                for name, cl in cands.items():
                    g, m, _, _ = choose(cl, slo)
                    met[name][0] += int(real[(g, m)] <= slo)
                    met[name][1] += 1

    rate = {k: f"{v[0]}/{v[1]}" for k, v in met.items()}
    print(f"{test_w:<8}{mape:>10.1f}%{cov:>10.0f}%   {rate['GPR']:>8}"
          f"{rate['GPR cautious']:>11}{rate['Mean-based']:>12}")
    rows.append({"unseen_window": test_w, "train_windows": "+".join(train_w),
                 "mape_pct": round(mape, 2), "coverage_pct": round(cov, 1),
                 **{f"{k}_slo_met": v[0] for k, v in met.items()},
                 "scenarios": met["GPR"][1]})

with open("results/case_study2.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("\nSaved results/case_study2.csv")
