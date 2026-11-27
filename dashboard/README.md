# Dashboard: Multiverse of Monsoon

A map of India for the model. Pick a real 2021 window, watch the 10-day forecast, add a shock (a warmer
Bay of Bengal, a stalled monsoon, or your own box) and compare the two futures side by side.

Every forecast on screen comes from a live `POST /simulate` call to the model server. Nothing is pre-rendered.
The dashboard talks to the server over HTTP only and never imports server code.

```
browser (5173)  ->  dashboard backend (8001)  ->  model server serve.py (8000)
                    /samples, /grid                /simulate, /health
```

## Run

From the repo root. The kit data in `data/` (`india_climate_train.nc`, gitignored) is optional.

Without data/, the backend serves the 10 sample windows from dashboard/backend/samples.npz (modified ERA5, Copernicus Climate Change Service, CC BY 4.0; rebuild with python eval/make_dashboard_samples.py).
samples.npz holds the 10 demo windows (120 days of the provided 2021 data) for the dashboard only; serve.py never reads it.

```bash
# 1. model server
.venv/bin/python serve.py --port 8000

# 2. dashboard backend (reads data/ or samples.npz, passes /simulate on to port 8000)
.venv/bin/python dashboard/backend/app.py --port 8001

# 3. frontend
cd dashboard/frontend
npm install
npm run dev          # http://localhost:5173
```

Settings:
- `MODEL_SERVER` (backend env var): model server URL, default `http://localhost:8000`.
- `VITE_API_URL` (frontend env var): backend URL, default `http://localhost:8001`.

## What is on the page

- **Explore map**: the 32 x 36 grid as a heatmap, channel switcher, day slider 1-10 with play,
  wind arrows from u10/v10, and a forecast vs truth toggle. Hover a cell to read its value.
- **Shock builder**: draw a box on the map (it snaps to grid cells) or type its edges, then set the change:
  +K for t2m, % of tcwv, % of wind (u10 and v10 together). The change is added to the last input day,
  same as the judge's shocks. Presets:
  - Warm Bay of Bengal: lat 8-20, lon 82-94, +3 K t2m, +15% tcwv
  - Monsoon stall: lat 8-20, lon 65-75, -40% u10 and v10
- **Multiverse**: baseline | shocked | difference maps for the selected day. The difference map keeps one
  colour range for all 10 days, so pressing play shows the shock spreading. The bar row counts the cells
  the shock has reached on each day.
- **Impact panel**: per region (rough lat/lon boxes, land cells only), heatwave risk as cell-days with
  daily mean t2m above a threshold you choose, ranked for baseline and shocked; and the rain deficit of the
  shocked run vs the baseline, ranked with the biggest deficit first.

## Sample windows

`backend/samples.py` lists 10 windows from 2021 (the test year): cyclones Tauktae, Yaas and Gulab,
two June heatwaves, the June monsoon surge, the July rain peak, the October Kerala and Uttarakhand rain,
and two winter cold spells. Each one is 3 input days plus the 10 true days that follow, cut from
`data/india_climate_train.nc` at start-up, or from `backend/samples.npz` when `data/` is missing.

## Note for the repo scan

`monsoon_r1.repo_scan` treats any `.py` file that contains the `/simulate` route as inference code, and
then also the local files it imports. That is why the forecast pass-through sits alone in
`backend/model_proxy.py`, with no local imports and no dates. Keep it that way: the sample dates live in
`samples.py`, which the proxy must never import.
