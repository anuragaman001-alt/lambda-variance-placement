# ---------------------------------------------------------------------------
# train_gpr.py  (version 2: bootstrap-calibrated)
#
# Goal: predict the P99 (tail) latency of a configuration from its settings:
#       runtime + workload + memory size  ->  predicted P99 + 95% interval
#
# What is new in v2:
#   In v1 the model treated every measured P99 as an exact number and used one
#   shared noise level for all configs. Its 95% intervals were far too narrow
#   (only 4 of 12 hidden configs fell inside).
#   In v2 I first measure HOW UNCERTAIN each config's P99 is, using
#   bootstrapping (explained below), and give that to the model per config.
#   Noisy configs (e.g. Node.js at small memory) now get wide intervals,
#   stable ones (e.g. Python cpu) get narrow intervals. This is what makes the
#   model "variance-aware".
#
# Test: hide memory sizes 256 MB and 1024 MB, train on 128/512/2048, and check
#       how close the predictions are and whether the real values fall inside
#       the 95% intervals.
#
# Input : results/raw_*.csv  (all runs, made by run_experiment.py)
# Output: results/gpr_predictions.csv  and  results/gpr_p99_plot.png
#
# Run with:  python3 train_gpr.py
# Needs:     pip install scikit-learn matplotlib
# ---------------------------------------------------------------------------
import csv
import glob
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")                      # save the plot to a file, no window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# The optimiser warns when an input turns out not to matter (its length-scale
# hits the upper limit). Expected here: one of the one-hot workload columns is
# redundant. Hide the warning so the output stays readable.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

HELD_OUT_MEMORY = {256, 1024}             # memory sizes the model never sees
RUNTIMES = ["nodejs20.x", "python3.11"]
WORKLOADS = ["cpu", "memory", "io"]
N_BOOT = 1000                             # number of bootstrap resamples
rng = np.random.default_rng(42)           # fixed seed -> same result every run

# ---------------------------------------------------------------------------
# 1. Load all warm calls from every raw CSV, grouped by configuration
# ---------------------------------------------------------------------------
warm = defaultdict(list)
for f in glob.glob("results/raw_*.csv"):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            # skip failed calls and cold starts (cold starts are a separate thing)
            if r["error"] or not r["duration_ms"] or r["is_cold"] == "1":
                continue
            key = (r["runtime"], r["workload"], int(r["memory_mb"]))
            warm[key].append(float(r["duration_ms"]))

configs = sorted(warm)                    # list of (runtime, workload, memory)
print(f"Loaded {sum(len(v) for v in warm.values())} warm calls "
      f"for {len(configs)} configurations")

# ---------------------------------------------------------------------------
# 2. P99 per config + its uncertainty (bootstrapping)
#
# Bootstrapping: from the n measured calls of one config, draw n calls at
# random WITH replacement (the same call can be picked twice) and compute the
# P99 of that sample. Repeat 1,000 times. The spread of those 1,000 P99 values
# shows how much the P99 would change if I re-ran the experiment.
#
# I work in log space (log of ms): latencies range from ~10 ms to ~1600 ms,
# so logs keep slow configs from dominating, and predictions stay positive.
# method="weibull" gives the same percentile definition as summarize.py.
# ---------------------------------------------------------------------------
log_p99, boot_var = [], []
for key in configs:
    d = np.array(warm[key])
    log_p99.append(np.log(np.percentile(d, 99, method="weibull")))
    samples = rng.choice(d, size=(N_BOOT, len(d)), replace=True)
    boot = np.log(np.percentile(samples, 99, axis=1, method="weibull"))
    boot_var.append(boot.var())            # uncertainty of this config's P99
y = np.array(log_p99)
boot_var = np.array(boot_var)


def features(runtime, workload, memory_mb):
    """Turn one configuration into numbers the model can use.

    - runtime and workload are categories, so each becomes 0/1 columns
      ("one-hot encoding"), e.g. workload=io -> [0, 0, 1]
    - memory goes in as log2(memory): 128->7, 256->8, ... 2048->11.
      Each doubling of memory is one equal step, which matches how Lambda
      gives CPU in proportion to memory.
    """
    return ([1.0 if runtime == r else 0.0 for r in RUNTIMES]
            + [1.0 if workload == w else 0.0 for w in WORKLOADS]
            + [np.log2(memory_mb)])


X = np.array([features(*k) for k in configs])
mem = np.array([k[2] for k in configs])
test = np.isin(mem, list(HELD_OUT_MEMORY))      # True for 256 / 1024 MB rows
train = ~test

# ---------------------------------------------------------------------------
# 3. Build and train the GPR
# ---------------------------------------------------------------------------
# Kernel = how the model decides which configs are "similar":
#   ConstantKernel * RBF : smooth relationship; one length-scale per input
#                          (ARD), so the model learns which input matters most
#   WhiteKernel          : small extra noise for anything the smooth curve
#                          can't explain (e.g. sudden jumps between tiers)
# alpha = per-config bootstrap variance. This is the key change in v2: each
#         training point tells the model how much it can be trusted.
kernel = (ConstantKernel(1.0, (1e-2, 1e2))
          * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2))
          + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e0)))

gpr = GaussianProcessRegressor(kernel=kernel, alpha=boot_var[train],
                               normalize_y=True, n_restarts_optimizer=10,
                               random_state=42)
gpr.fit(X[train], y[train])
print("Learned kernel:", gpr.kernel_)

# ---------------------------------------------------------------------------
# 4. Predict and evaluate
# ---------------------------------------------------------------------------
mu, sd = gpr.predict(X, return_std=True)        # model mean + model uncertainty

# The real P99 I compare against is itself a noisy measurement, so the
# interval for it = model uncertainty + that config's own bootstrap noise.
total_sd = np.sqrt(sd ** 2 + boot_var)
pred = np.exp(mu)                                         # back to ms
lo, hi = np.exp(mu - 1.96 * total_sd), np.exp(mu + 1.96 * total_sd)
actual = np.exp(y)

# MAPE = mean absolute percentage error (average % the prediction is off)
mape = np.mean(np.abs(pred[test] - actual[test]) / actual[test]) * 100
# Coverage = share of real values inside the 95% interval (ideal ~95%)
inside = (actual[test] >= lo[test]) & (actual[test] <= hi[test])
print(f"\nHeld-out configs (memory {sorted(HELD_OUT_MEMORY)} MB): {test.sum()}")
print(f"MAPE on held-out P99     : {mape:.1f} %")
print(f"95% interval coverage    : {inside.sum()}/{test.sum()} "
      f"({inside.mean() * 100:.0f} %)\n")

# boot% = how much this config's own P99 wobbles (bootstrap std, as a %)
print(f"{'runtime':<12}{'workload':<8}{'mem':>6}{'actual':>10}{'pred':>10}"
      f"{'95% interval':>22}{'err%':>7}{'boot%':>7}{'in?':>5}")
with open("results/gpr_predictions.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["runtime", "workload", "memory_mb", "split", "actual_p99_ms",
                "pred_p99_ms", "ci95_low_ms", "ci95_high_ms", "abs_err_pct",
                "boot_std_pct", "inside_ci"])
    for i, (rt, wl, m) in enumerate(configs):
        err = abs(pred[i] - actual[i]) / actual[i] * 100
        bpct = np.sqrt(boot_var[i]) * 100     # log-space std ~= relative %
        ok = lo[i] <= actual[i] <= hi[i]
        w.writerow([rt, wl, m, "test" if test[i] else "train",
                    round(actual[i], 2), round(pred[i], 2), round(lo[i], 2),
                    round(hi[i], 2), round(err, 1), round(bpct, 1), int(ok)])
        if test[i]:
            print(f"{rt:<12}{wl:<8}{m:>6}{actual[i]:>10.1f}{pred[i]:>10.1f}"
                  f"{f'[{lo[i]:.1f}, {hi[i]:.1f}]':>22}{err:>7.1f}{bpct:>7.1f}"
                  f"{'yes' if ok else 'NO':>5}")

# ---------------------------------------------------------------------------
# 5. Plot: one small chart per runtime x workload
#    line + band = model prediction with 95% interval across all memory sizes
#    dots + whiskers = measured P99 with its own bootstrap 95% range
#    blue = trained on, orange = hidden from the model (the real test)
# ---------------------------------------------------------------------------
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
grid_mem = np.logspace(np.log2(128), np.log2(2048), 60, base=2)

fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), facecolor="#fcfcfb")
for ri, rt in enumerate(RUNTIMES):
    for wi, wl in enumerate(WORKLOADS):
        ax = axes[ri, wi]
        ax.set_facecolor("#fcfcfb")
        Xg = np.array([features(rt, wl, m) for m in grid_mem])
        gm, gs = gpr.predict(Xg, return_std=True)
        ax.fill_between(grid_mem, np.exp(gm - 1.96 * gs), np.exp(gm + 1.96 * gs),
                        color=BLUE, alpha=0.15, linewidth=0, label="95% interval")
        ax.plot(grid_mem, np.exp(gm), color=BLUE, linewidth=2, label="GPR prediction")

        for split, color, label in [(train, BLUE, "Measured (train)"),
                                    (test, ORANGE, "Measured (held out)")]:
            sel = np.array([(k[0] == rt and k[1] == wl) for k in configs]) & split
            yerr = [actual[sel] - np.exp(y[sel] - 1.96 * np.sqrt(boot_var[sel])),
                    np.exp(y[sel] + 1.96 * np.sqrt(boot_var[sel])) - actual[sel]]
            ax.errorbar(mem[sel], actual[sel], yerr=yerr, fmt="o", ms=7,
                        color=color, mec="#fcfcfb", mew=1.5, elinewidth=1.5,
                        capsize=3, zorder=3, label=label)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([128, 256, 512, 1024, 2048])
        ax.set_xticklabels(["128", "256", "512", "1024", "2048"])
        ax.minorticks_off()
        ax.set_title(f"{rt} · {wl}", fontsize=11, color=INK, loc="left")
        ax.tick_params(colors=MUTED, labelsize=9)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#d6d5d0")
        ax.grid(True, color="#ecebe8", linewidth=0.8)
        if wi == 0:
            ax.set_ylabel("P99 latency (ms, log)", color=MUTED, fontsize=9)
        if ri == 1:
            ax.set_xlabel("Memory (MB)", color=MUTED, fontsize=9)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.suptitle(f"GPR P99 prediction (bootstrap-calibrated) — held-out 256/1024 MB: "
             f"MAPE {mape:.1f}%, 95% interval coverage {inside.sum()}/{test.sum()}",
             x=0.01, y=0.985, ha="left", fontsize=12, color=INK)
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.945),
           ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("results/gpr_p99_plot.png", dpi=160)
print("\nSaved results/gpr_predictions.csv and results/gpr_p99_plot.png")
