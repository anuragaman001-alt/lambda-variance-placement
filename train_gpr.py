# ---------------------------------------------------------------------------
# train_gpr.py
# First version of the GPR (Gaussian Process Regression) model.
#
# Goal: predict the P99 (tail) latency of a configuration from its settings:
#       runtime + workload + memory size  ->  predicted P99 + 95% interval
#
# Test: I hide two memory sizes (256 MB and 1024 MB) from the model, train it
#       on the other three (128, 512, 2048), and check how close it gets to
#       the real P99 of the hidden configs. If it is close, GPR can estimate
#       configurations that were never measured, which is the core idea
#       behind variance-aware placement.
#
# Input : results/summary.csv   (made by summarize.py, one row per config)
# Output: results/gpr_predictions.csv  and  results/gpr_p99_plot.png
#
# Run with:  python3 train_gpr.py
# Needs:     pip install scikit-learn matplotlib
# ---------------------------------------------------------------------------
import csv
import warnings

import matplotlib
matplotlib.use("Agg")                      # save the plot to a file, no window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# The optimiser warns when an input turns out not to matter (its length-scale
# hits the upper limit). That's expected here: one of the one-hot workload
# columns is redundant. Hide the warning so the output stays readable.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

HELD_OUT_MEMORY = {256, 1024}             # memory sizes the model never sees
RUNTIMES = ["nodejs20.x", "python3.11"]
WORKLOADS = ["cpu", "memory", "io"]

# ---------------------------------------------------------------------------
# 1. Load the per-config summary
# ---------------------------------------------------------------------------
with open("results/summary.csv") as fh:
    rows = list(csv.DictReader(fh))


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


X = np.array([features(r["runtime"], r["workload"], int(r["memory_mb"])) for r in rows])

# Target = log of P99. Latencies range from ~10 ms to ~1600 ms, so working in
# log space stops the slow configs from dominating the fit, and keeps every
# prediction positive when converted back to ms.
y = np.log(np.array([float(r["p99_ms"]) for r in rows]))

mem = np.array([int(r["memory_mb"]) for r in rows])
test = np.isin(mem, list(HELD_OUT_MEMORY))      # True for 256 / 1024 MB rows
train = ~test

# ---------------------------------------------------------------------------
# 2. Build and train the GPR
# ---------------------------------------------------------------------------
# Kernel = how the model decides which configs are "similar":
#   ConstantKernel * RBF : smooth relationship; one length-scale per input
#                          (ARD), so the model learns which input matters most
#   WhiteKernel          : allows for measurement noise
kernel = (ConstantKernel(1.0, (1e-2, 1e2))
          * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2))
          + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e0)))

# normalize_y: centre the targets first; n_restarts: try several starting
# points so the optimiser doesn't get stuck in a bad solution.
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                               n_restarts_optimizer=10, random_state=42)
gpr.fit(X[train], y[train])
print("Learned kernel:", gpr.kernel_)

# ---------------------------------------------------------------------------
# 3. Predict the hidden configs and measure the error
# ---------------------------------------------------------------------------
mu, sd = gpr.predict(X, return_std=True)        # mean and std, in log space
pred = np.exp(mu)                               # back to milliseconds
lo, hi = np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd)   # 95% interval
actual = np.exp(y)

# MAPE = mean absolute percentage error (average % the prediction is off)
mape = np.mean(np.abs(pred[test] - actual[test]) / actual[test]) * 100
# Coverage = share of real values that fall inside the 95% interval.
# Ideally close to 95%.
inside = (actual[test] >= lo[test]) & (actual[test] <= hi[test])
print(f"\nHeld-out configs (memory {sorted(HELD_OUT_MEMORY)} MB): {test.sum()}")
print(f"MAPE on held-out P99     : {mape:.1f} %")
print(f"95% interval coverage    : {inside.sum()}/{test.sum()} "
      f"({inside.mean() * 100:.0f} %)\n")

print(f"{'runtime':<12}{'workload':<8}{'mem':>6}{'actual':>10}{'pred':>10}{'95% interval':>22}{'err%':>7}")
with open("results/gpr_predictions.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["runtime", "workload", "memory_mb", "split", "actual_p99_ms",
                "pred_p99_ms", "ci95_low_ms", "ci95_high_ms", "abs_err_pct"])
    for i, r in enumerate(rows):
        err = abs(pred[i] - actual[i]) / actual[i] * 100
        split = "test" if test[i] else "train"
        w.writerow([r["runtime"], r["workload"], r["memory_mb"], split,
                    round(actual[i], 2), round(pred[i], 2),
                    round(lo[i], 2), round(hi[i], 2), round(err, 1)])
        if test[i]:
            print(f"{r['runtime']:<12}{r['workload']:<8}{r['memory_mb']:>6}"
                  f"{actual[i]:>10.1f}{pred[i]:>10.1f}"
                  f"{f'[{lo[i]:.1f}, {hi[i]:.1f}]':>22}{err:>7.1f}")

# ---------------------------------------------------------------------------
# 4. Plot: one small chart per runtime x workload
#    line + band = model prediction with 95% interval (across all memory sizes)
#    blue dots   = configs the model was trained on
#    orange dots = hidden configs (the real test)
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

        sel = np.array([(r["runtime"] == rt and r["workload"] == wl) for r in rows])
        ax.scatter(mem[sel & train], actual[sel & train], s=60, color=BLUE,
                   edgecolor="#fcfcfb", linewidth=2, zorder=3, label="Measured (train)")
        ax.scatter(mem[sel & test], actual[sel & test], s=60, color=ORANGE,
                   edgecolor="#fcfcfb", linewidth=2, zorder=3, label="Measured (held out)")

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
fig.suptitle(f"GPR P99 prediction — held-out 256/1024 MB: MAPE {mape:.1f}%, "
             f"95% interval coverage {inside.sum()}/{test.sum()}",
             x=0.01, y=0.985, ha="left", fontsize=12, color=INK)
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.945),
           ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("results/gpr_p99_plot.png", dpi=160)
print("\nSaved results/gpr_predictions.csv and results/gpr_p99_plot.png")
