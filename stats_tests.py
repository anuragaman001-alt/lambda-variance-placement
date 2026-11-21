# ---------------------------------------------------------------------------
# stats_tests.py  -  Experiment 1: what changes the variance?
#
# Uses the coefficient of variation (CV = std / mean of warm run times) of
# each of the 120 configurations as the measure of variance, then asks:
#   - Does memory size change the CV?   Kruskal-Wallis (5 groups of 24)
#   - Does the region change the CV?    Kruskal-Wallis (4 groups of 30)
#   - Does the workload change the CV?  Kruskal-Wallis (3 groups of 40)
#   - Does the runtime change the CV?   Mann-Whitney U (60 vs 60) + effect size r
# Non-parametric tests are used because run times (and CVs) are not normal.
# p < 0.05 means the factor makes a real difference.
#
# It also compares the average warm CV with the base paper
# (Wen et al., 2025: 8.39% warm, 6.07% cold).
#
# Input : results/raw_*.csv      Output: results/stats_tests.csv
# Run with:  python3 stats_tests.py      (needs: pip install scipy)
# ---------------------------------------------------------------------------
import csv
import glob
from collections import defaultdict

import numpy as np
from scipy import stats

warm = defaultdict(list)
for f in glob.glob("results/raw_*.csv"):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if r["error"] or not r["duration_ms"] or r["is_cold"] == "1":
                continue
            warm[(r["region"], r["runtime"], r["workload"], int(r["memory_mb"]))].append(
                float(r["duration_ms"]))

keys = sorted(warm)
cv = {k: np.std(warm[k], ddof=1) / np.mean(warm[k]) for k in keys}
FACTORS = {"memory_mb": 3, "region": 0, "workload": 2}

rows = []
print(f"{'factor':<11}{'test':<16}{'statistic':>10}{'p-value':>11}  result")
for name, idx in FACTORS.items():
    levels = sorted({k[idx] for k in keys})
    groups = [[cv[k] for k in keys if k[idx] == lv] for lv in levels]
    h, p = stats.kruskal(*groups)
    medians = ", ".join(f"{lv}: {np.median(g) * 100:.1f}%" for lv, g in zip(levels, groups))
    verdict = "differs" if p < 0.05 else "no clear difference"
    print(f"{name:<11}{'Kruskal-Wallis':<16}{h:>10.2f}{p:>11.4f}  {verdict}  (median CV {medians})")
    rows.append({"factor": name, "test": "Kruskal-Wallis", "statistic": round(h, 3),
                 "p_value": round(p, 5), "effect_size_r": "", "median_cv_by_level": medians})

py = [cv[k] for k in keys if k[1] == "python3.11"]
nd = [cv[k] for k in keys if k[1] == "nodejs20.x"]
u, p = stats.mannwhitneyu(py, nd, alternative="two-sided")
# effect size r = |Z| / sqrt(N); Z from the normal approximation of U
n1, n2 = len(py), len(nd)
z = (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
r = abs(z) / np.sqrt(n1 + n2)
size = "large" if r >= 0.5 else "medium" if r >= 0.3 else "small"
medians = f"python3.11: {np.median(py) * 100:.1f}%, nodejs20.x: {np.median(nd) * 100:.1f}%"
print(f"{'runtime':<11}{'Mann-Whitney U':<16}{u:>10.1f}{p:>11.4f}  "
      f"{'differs' if p < 0.05 else 'no clear difference'}, r = {r:.2f} ({size})  (median CV {medians})")
rows.append({"factor": "runtime", "test": "Mann-Whitney U", "statistic": round(u, 3),
             "p_value": round(p, 5), "effect_size_r": round(r, 3), "median_cv_by_level": medians})

mean_cv = np.mean(list(cv.values())) * 100
print(f"\nAverage warm CV over {len(keys)} configurations: {mean_cv:.2f}% "
      f"(base paper, Wen et al. 2025: 8.39% warm)")

with open("results/stats_tests.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("Saved results/stats_tests.csv")
