# CONTRACT.md — source of truth

## Data
- Grid: 32 x 36 cells, 1°, lat 6.5..37.5 (south→north), lon 65.5..100.5 (west→east)
- Channels (axis 1, this order): 0 t2m K | 1 tp mm/day | 2 u10 m/s | 3 v10 m/s | 4 msl Pa | 5 tcwv kg/m²
- Raw data: data/india_climate_train.nc + data/stats.json (gitignored)
- Split: train 2005–2019, val 2020, test 2021 (test is only for eval/ and dashboard samples)

## assets/ (owner: pawar)
- stats.json — copy of the kit file (per-channel mean, std)
- clim.npy — float32 [366, 6, 32, 36] day-of-year climatology, physical units
- static.npy — float32 [2, 32, 36] land_sea_mask, orography_m
- weights.pt — CNN weights (Git LFS)
- baseline.npz — fallback model coefficients (owner: nayak)

## model/rollout.py (owner: pawar)
rollout(init_states, doy_sincos, steps, perturbation) -> np.ndarray
- init_states: float32 [3, 6, 32, 36], physical units, oldest first
- doy_sincos: [sin, cos] of the day of year of init_states[-1]
- perturbation: float32 [6, 32, 36] physical delta added to init_states[-1], or None
- returns float32 [steps, 6, 32, 36], physical units; output[k] = day k+1 after the last input
- deterministic, finite, rain >= 0, t2m clamped to 233.15–333.15 K

## engine/baseline.py (owner: nayak)
Same signature as model/rollout.py. serve.py uses rollout() if assets/weights.pt loads and its warm-up rollout is finite, else baseline (logged to stderr).

## Server API (owner: nayak) — python serve.py --port 8000, host 0.0.0.0
- POST /health → {"status": "ready", "model_name": str}
- POST /simulate {init_states, doy_sincos, steps, perturbation|null} → {"rollout": [steps,6,32,36], "inference_ms": float}
- Plain nested JSON lists, physical units. 10 steps must take < 5 s (target < 1 s).

## Dashboard backend (owner: sarthak)
- python dashboard/backend/app.py --port 8001
- GET /samples → list of {id, label, kind, last_input_day, doy_sincos} (windows from the 2021 test set)
- GET /samples/{id} → {id, label, init_states [3,6,32,36], doy_sincos, truth [10,6,32,36], truth_days [10]}
- GET /grid → {lat [32], lon [36], land_mask [32][36] (0/1)}
- POST /simulate and POST /health are passed through unchanged to the model server (MODEL_SERVER, default http://localhost:8000)
- Calls the model server's /simulate for every forecast shown. Nothing pre-rendered.
