"""Training data access. Reads data/ (raw kit files, gitignored); never imported by serve.py or model/."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASSETS = ROOT / "assets"

SPLITS = {"train": (2005, 2019), "val": (2020, 2020), "test": (2021, 2021), "all": (2005, 2021),
          "trainval": (2005, 2020)}


def load_archive():
    """-> data [N,6,32,36] f32 physical, times (DatetimeIndex), static [2,32,36], lat, lon."""
    ds = xr.open_dataset(DATA / "india_climate_train.nc")
    data = ds["data"].values.astype("f4")
    times = pd.DatetimeIndex(ds["time"].values)
    static = ds["static"].values.astype("f4")
    return data, times, static, ds["lat"].values, ds["lon"].values


def year_mask(times: pd.DatetimeIndex, split: str) -> np.ndarray:
    y0, y1 = SPLITS[split]
    return (times.year >= y0) & (times.year <= y1)


def doy(times: pd.DatetimeIndex) -> np.ndarray:
    """1-based day of year, the convention doy_sincos uses on the wire."""
    return np.asarray(times.day_of_year, dtype="f4")


def clim_index(doy_value) -> np.ndarray:
    """Same mapping model/rollout.py uses: day of year (1-based, may exceed 365) -> row of the 366-day climatology."""
    return (np.rint(np.asarray(doy_value)).astype(int) - 1) % 366


def kit_stats() -> dict:
    return json.loads((DATA / "stats.json").read_text())
