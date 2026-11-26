"""Build the static inference assets from the raw kit data.

  python -m train.build_assets [--split train]

Writes assets/clim.npy [366,6,32,36] (15-day smoothed day-of-year climatology, physical units, same recipe as
the kit's monsoon_r1.stats.climatology), assets/static.npy [2,32,36] and a copy of stats.json.
"""
from __future__ import annotations
import argparse, shutil
import numpy as np
from monsoon_r1.stats import climatology

from train.data import ASSETS, DATA, load_archive, year_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", help="years used for the climatology (train | trainval | all)")
    a = ap.parse_args()
    data, times, static, _, _ = load_archive()
    m = year_mask(times, a.split)
    dates = np.asarray(times[m].strftime("%Y%m%d").astype(int))
    clim = climatology(data[m], dates)
    ASSETS.mkdir(exist_ok=True)
    np.save(ASSETS / "clim.npy", clim.astype("f4"))
    np.save(ASSETS / "static.npy", static.astype("f4"))
    shutil.copyfile(DATA / "stats.json", ASSETS / "stats.json")
    print(f"clim {clim.shape} from {m.sum()} days ({a.split}); static {static.shape}; stats.json copied")


if __name__ == "__main__":
    main()
