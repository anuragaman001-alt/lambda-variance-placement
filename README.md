# Performance Variance-Aware Function Placement in AWS Lambda

MSc Cloud Computing research project (NCI Dublin) — measuring latency variance across
AWS Lambda configurations to train a Gaussian Process Regression placement model.

## Experiment slice (this run)
- Region: eu-north-1
- 2 runtimes (Python 3.11, Node.js 20.x) x 3 workloads (cpu, memory, io) x 5 memory tiers (128–2048 MB) = 30 configs
- 200 invocations per config (1 forced cold start + 199 warm) = **6,000 invocations**
- Metrics per call: Lambda-reported duration, billed duration, init (cold-start) duration, max memory, client round-trip latency

## Run (AWS CloudShell)
```bash
python3 deploy.py                 # create IAM role + 30 functions
python3 run_experiment.py --window W1
python3 summarize.py              # per-config mean / std / CV / P50 / P95 / P99
```

## Layout
- `functions/` – Lambda handler code (Python + Node.js, identical workloads)
- `config.py` – experiment matrix
- `deploy.py`, `run_experiment.py`, `summarize.py`
- `results/` – raw CSVs and summary
