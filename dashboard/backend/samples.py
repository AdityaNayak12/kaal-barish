"""Sample windows for the dashboard, taken from the 2021 test year of data/india_climate_train.nc.

Each sample is 3 input days (init_states) and the 10 true days that follow (truth).
The full data stays in data/ (gitignored). Without it (a fresh clone), the store reads samples.npz next to this file:
only the 120 days these 10 windows need, written by eval/make_dashboard_samples.py.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "data" / "india_climate_train.nc"
BUNDLE = Path(__file__).resolve().parent / "samples.npz"  # only the 10 windows' days (eval/make_dashboard_samples.py)

INPUT_DAYS = 3
TRUTH_DAYS = 10

# Each window is named by the LAST input day. The event happens inside the 10 forecast days.
# These dates were picked by looking at the 2021 data (rain peaks, highest temperatures, lowest pressure).
WINDOWS = [
    {"id": "cyclone-tauktae", "kind": "cyclone", "last_input_day": "2021-05-13",
     "label": "Cyclone Tauktae, Arabian Sea (mid May)"},
    {"id": "cyclone-yaas", "kind": "cyclone", "last_input_day": "2021-05-22",
     "label": "Cyclone Yaas, Bay of Bengal (late May)"},
    {"id": "heatwave-june", "kind": "heatwave", "last_input_day": "2021-06-05",
     "label": "Pre-monsoon heatwave, north-west India (early June)"},
    {"id": "monsoon-onset", "kind": "monsoon", "last_input_day": "2021-06-12",
     "label": "Monsoon surge over central India (mid June)"},
    {"id": "heatwave-break", "kind": "heatwave", "last_input_day": "2021-06-26",
     "label": "Monsoon break heat, north India (late June)"},
    {"id": "monsoon-peak", "kind": "monsoon", "last_input_day": "2021-07-16",
     "label": "Heavy monsoon rain, peak of the season (mid July)"},
    {"id": "cyclone-gulab", "kind": "cyclone", "last_input_day": "2021-09-23",
     "label": "Cyclone Gulab, Bay of Bengal (late September)"},
    {"id": "october-rain", "kind": "rain", "last_input_day": "2021-10-13",
     "label": "Kerala and Uttarakhand extreme rain (mid October)"},
    {"id": "winter-january", "kind": "winter", "last_input_day": "2021-01-27",
     "label": "Winter cold over north India (late January)"},
    {"id": "winter-december", "kind": "winter", "last_input_day": "2021-12-16",
     "label": "December cold wave, north India"},
]


def doy_sincos(day):
    """[sin, cos] of the day of year, same formula as the kit: 2*pi*doy/365."""
    doy = day.dayofyear
    angle = 2.0 * math.pi * doy / 365.0
    return [math.sin(angle), math.cos(angle)]


class SampleStore:
    """Loads the 2021 part of the dataset once and cuts windows out of it."""

    def __init__(self, data_file=DATA_FILE):
        if not data_file.exists() and BUNDLE.exists():  # fresh clone without data/: use the bundled windows
            z = np.load(BUNDLE)
            self.days = pd.to_datetime(z["days"])
            self.values = z["values"]
            self.lat, self.lon = z["lat"].tolist(), z["lon"].tolist()
            self.land_mask = z["land_mask"].astype(int).tolist()
            return
        if not data_file.exists():
            raise FileNotFoundError(
                f"{data_file} not found. Download india_climate_train.nc from the kit release into data/."
            )
        ds = xr.open_dataset(data_file)
        year = ds["data"].sel(time=slice("2021-01-01", "2021-12-31")).load()
        self.days = pd.to_datetime(year["time"].values)
        self.values = year.values.astype(np.float32)  # [365, 6, 32, 36]
        self.lat = ds["lat"].values.tolist()
        self.lon = ds["lon"].values.tolist()
        land_sea = ds["static"].sel(static_channel="land_sea_mask").values
        self.land_mask = (land_sea > 0.5).astype(int).tolist()
        ds.close()

    def _index_of(self, day_text):
        day = pd.Timestamp(day_text)
        for i in range(len(self.days)):
            if self.days[i] == day:
                return i
        raise KeyError(day_text)

    def list_samples(self):
        result = []
        for window in WINDOWS:
            last_day = pd.Timestamp(window["last_input_day"])
            result.append({
                "id": window["id"],
                "label": window["label"],
                "kind": window["kind"],
                "last_input_day": window["last_input_day"],
                "doy_sincos": doy_sincos(last_day),
            })
        return result

    def get_sample(self, sample_id):
        window = None
        for w in WINDOWS:
            if w["id"] == sample_id:
                window = w
        if window is None:
            return None

        last = self._index_of(window["last_input_day"])
        first_input = last - INPUT_DAYS + 1
        last_truth = last + TRUTH_DAYS
        if first_input < 0 or last_truth >= len(self.days):
            raise ValueError(f"window {sample_id} does not fit inside 2021")

        init_states = self.values[first_input:last + 1]
        truth = self.values[last + 1:last_truth + 1]
        truth_days = []
        for i in range(last + 1, last_truth + 1):
            truth_days.append(str(self.days[i].date()))

        return {
            "id": window["id"],
            "label": window["label"],
            "init_states": init_states.tolist(),
            "doy_sincos": doy_sincos(self.days[last]),
            "truth": truth.tolist(),
            "truth_days": truth_days,
        }

    def grid(self):
        return {"lat": self.lat, "lon": self.lon, "land_mask": self.land_mask}
