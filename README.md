# Kaal Barish: Multiverse of Monsoon

Climate digital twin of India for the Yugaantar DevForge "Multiverse of Monsoon" track (Round 1).
Given 3 days of weather over India, `serve.py` forecasts the next 1 to 30 days, and it can replay
the same forecast with a "what if" shock added (for example a warmer Bay of Bengal or a stalled monsoon).
The dashboard puts both futures side by side on a live map.

- Grid: 32 x 36 cells of 1°, lat 6.5–37.5 N, lon 65.5–100.5 E.
- Channels: t2m (K), tp (mm/day), u10, v10 (m/s), msl (Pa), tcwv (kg/m²), in physical units on the wire.
- Data: ERA5 daily fields from the organisers' kit, 2005–2021. Train 2005–2019, val 2020, test 2021.

## How it works

- **Model** (`model/`): a residual CNN that steps the standardised state forward one day at a time
  (rain goes through log1p first) and predicts the change from the previous day. Its receptive field is
  small, so a shock stays local on day 1 and spreads at a physical pace after that. It was trained on
  rollouts of 1, 4 and then 8 steps. The shipped weights are a 2-member deterministic ensemble (r2 and
  r4, the same recipe with a different seed) whose rollouts are averaged. A rain output gain, tuned on
  2020, improves heavy-rain detection. A response limiter keeps a shocked run within 4.5 channel stds of
  the unshocked one.
- **Fallback** (`engine/baseline.py`): a per-cell linear regression of future anomalies against
  climatology, with neighbour smoothing. `serve.py` uses it if the CNN fails to load, and for any single
  request where the CNN fails or gives invalid output.
- **Server** (`serve.py`): FastAPI, offline, CPU only. It loads and warms up both models before it
  starts listening, so `/health` only answers once it is ready.
- **Dashboard** (`dashboard/`): React + Leaflet frontend and a small backend that serves the 2021 sample
  windows. Every forecast on screen is a live `/simulate` call. See `dashboard/README.md`.

## Repo layout

```
serve.py              inference server (POST /health, POST /simulate)
model/                CNN definition (net.py) and inference entry point (rollout.py)
engine/baseline.py    fallback model
assets/               weights.pt (plain git file), clim.npy, static.npy, stats.json, baseline.npz
train/                data access, asset building, training, calibration, evaluation
eval/                 local score harness, judge request replay, server self-check, submission precheck
dashboard/            backend (port 8001) and frontend (port 5173)
CONTRACT.md           shapes, file paths and endpoints shared by the team
data/                 raw kit data, gitignored, never committed
```

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Download `india_climate_train.nc` and `stats.json` from the
[kit release](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0) into `data/`.
Install the kit wheel (`monsoon_r1`) from the same release to use the validator and the eval scripts.

## Train

```bash
.venv/bin/python -m train.build_assets                      # assets/clim.npy, static.npy, stats.json (2005-2019)
.venv/bin/python -m train.train --out data/runs/r2.pt       # curriculum: 15 ep 1-step, 6 ep 4-step, 3 ep 8-step (batch 16)
.venv/bin/python -m train.train --seed 1 --out data/runs/r4.pt                # second ensemble member, same recipe
.venv/bin/python -m train.make_ensemble data/runs/r2.pt data/runs/r4.pt --out data/runs/ens.pt
.venv/bin/python -m train.calibrate data/runs/ens.pt --out assets/weights.pt  # rain gain chosen on 2020 only (1.2)
.venv/bin/python -m train.shock_report --weights assets/weights.pt          # per-shock causality checks, 2020 and 2021
.venv/bin/python -m train.evaluate --split val              # kit scoring during training
.venv/bin/python eval/fit_baseline.py                       # fallback coefficients -> assets/baseline.npz
```

The shipped r2 was trained in two runs (the first was stopped by low memory after the 4-step stage):
`--stages 1:15,4:6`, then `--init <that checkpoint> --stages 8:3 --batch 16`. The default `--stages 1:15,4:6,8:3:16`
is the same schedule in one run. Training is CPU-only (about 1 to 1.5 hours per model; peak memory about 2.8 GB working set, 3.4 GB committed).

## Run

```bash
.venv/bin/python serve.py --port 8000
.venv/bin/python -m monsoon_r1.schema_validator http://localhost:8000
```

## Local checks

```bash
.venv/bin/python eval/score_local.py --split test           # auto-score estimate, in-process
.venv/bin/python eval/judge_sim.py http://localhost:8000    # grader's request sequence against a running server
.venv/bin/python eval/test_serve.py                         # serve.py edge cases and fallback
```

Local auto-score of the shipped ensemble from `eval/score_local.py` and `eval/drift_check.py`. They use the
kit's scoring functions on 40 windows per year. 2020 was used to choose the model and 2021 to check it.
2022 (built with the kit's own dataset builder) played no part in any choice, so it is the cleanest
held-out estimate. The brief's 15 points are marked by hand.

| part | val 2020 | test 2021 | held-out 2022 |
|---|---|---|---|
| Forecast RMSE skill /30 | 30.00 (SS 0.132) | 30.00 (SS 0.123) | 30.00 (SS 0.129) |
| Heavy-rain CSI /15 | 13.16 | 15.00 | 14.16 |
| Anti-blur /10 | 10.00 | 10.00 | 10.00 |
| Perturbation causality /15 | 13.83 | 13.31 | 13.78 |
| Repo scan /15 | 15.00 | 15.00 | 15.00 |
| **Auto total /85** | **81.99** | **83.31** | **82.94** |

The judge scores only 5 windows, and the RMSE points are capped at the mean, so one weak window costs
points. Over 2000 random 5-window draws, the expected RMSE + CSI points are 38.6 (2020), 40.3 (2021)
and 40.1 (2022) of 45, so a realistic auto total is about 77 to 79 of 85.

In a judge-like offline Docker run (`--network none`, 4 CPUs, 6 GB) from a fresh clone, `/health` was
ready in 2.5 s and peak memory was about 330 MB. Each 10-step call took under 0.3 s of server time
(under 0.5 s with torch 2.2). `serve.py` caps the CPU thread pools at 4. Without that cap, an older torch in a
4-CPU container sized its pools from the host and some calls took over 4 s.
