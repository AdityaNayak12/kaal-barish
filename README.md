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
  rollouts of 1, 4 and then 8 steps. A rain output gain, tuned on 2020, improves heavy-rain detection.
  A response limiter keeps a shocked run within 4.5 channel stds of the unshocked one.
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
.venv/bin/python -m train.train --out data/runs/r2.pt       # curriculum 1-, 4-, 8-step rollouts (--stages 1:15,4:8,8:5)
.venv/bin/python -m train.calibrate data/runs/r2.pt --out assets/weights.pt   # rain gain chosen on 2020
.venv/bin/python -m train.evaluate --split val              # kit scoring during training
.venv/bin/python eval/fit_baseline.py                       # fallback coefficients -> assets/baseline.npz
```

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

Local auto-score from `eval/score_local.py`. It uses the kit's scoring functions on 40 windows per year.
The brief's 15 points are marked by hand.

| part | val 2020 | test 2021 |
|---|---|---|
| Forecast RMSE skill /30 | 29.48 (SS 0.118) | 27.28 (SS 0.109) |
| Heavy-rain CSI /15 | 12.98 | 14.46 |
| Anti-blur /10 | 10.00 | 10.00 |
| Perturbation causality /15 | 13.97 | 13.69 |
| Repo scan /15 | 15.00 | 15.00 |
| **Auto total /85** | **81.42** | **80.42** |

In a judge-like offline Docker run (`--network none`, 4 CPUs, 6 GB) from a fresh clone, `/health` was
ready in 4 s, each 10-step call took under 0.3 s, and peak memory was about 250 MB.
