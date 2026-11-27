"""Write dashboard/backend/samples.npz: only the days the dashboard's 10 sample windows need (3 input + 10 truth days
each), plus lat, lon and land mask, so the dashboard runs from a fresh clone without data/india_climate_train.nc.
  python eval/make_dashboard_samples.py          (needs data/india_climate_train.nc)
Data: modified ERA5 (Copernicus Climate Change Service, Hersbach et al. 2020; CC BY 4.0), via the organisers' kit.
"""
import sys
from pathlib import Path
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent / "dashboard" / "backend"
sys.path.insert(0, str(BACKEND))
from samples import INPUT_DAYS, TRUTH_DAYS, WINDOWS, SampleStore  # noqa: E402

store = SampleStore()
idx = sorted({i for w in WINDOWS for i in range(store._index_of(w["last_input_day"]) - INPUT_DAYS + 1,
                                                  store._index_of(w["last_input_day"]) + TRUTH_DAYS + 1)})
np.savez_compressed(BACKEND / "samples.npz", values=store.values[idx], days=np.array(store.days[idx].strftime("%Y-%m-%d"), "U10"),
                    lat=np.array(store.lat, "f4"), lon=np.array(store.lon, "f4"), land_mask=np.array(store.land_mask, "u1"))
print(f"wrote {BACKEND / 'samples.npz'}: {len(idx)} days for {len(WINDOWS)} windows")
