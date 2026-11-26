"""Fit engine/baseline.py's coefficients on the train years and write assets/baseline.npz. Offline only, never imported
by the server.   python eval/fit_baseline.py [--smooth 4] [--kappa 1.0] [--ridge 1e-3]

Needs assets/clim.npy (pawar). With --make-clim it builds a dev copy with the kit's climatology() first.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.baseline import smooth                                     # noqa: E402

DATA = ROOT / "data" / "india_climate_train.nc"
TRAIN_END = "2019-12-31"                                               # CONTRACT.md split: train 2005-2019
LEADS = 14                                                             # contract MAX_STEPS


def load(path=DATA):
    """Daily fields [N,6,32,36] f4, yyyymmdd ints, climatology rows 0..365 (Feb 29 has its own row, like the kit)."""
    ds = xr.open_dataset(path)
    t = ds["time"].to_index()
    ymd = (t.year * 10000 + t.month * 100 + t.day).to_numpy()
    idx = (t.day_of_year - 1 + ((~t.is_leap_year) & (t.month > 2))).to_numpy()   # non-leap Mar 1 -> row 60
    return ds["data"].values.astype("f4"), ymd, idx, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", type=int, default=4); ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--ridge", type=float, default=1e-3); ap.add_argument("--make-clim", action="store_true")
    a = ap.parse_args()
    data, ymd, idx, t = load()
    n_tr = int((t <= TRAIN_END).sum())
    clim_path = ROOT / "assets" / "clim.npy"
    if a.make_clim:
        from monsoon_r1.stats import climatology
        clim_path.parent.mkdir(exist_ok=True)
        np.save(clim_path, climatology(data[:n_tr], ymd[:n_tr])); print("wrote dev", clim_path)
    clim = np.load(clim_path).astype("f4")
    anom = data[:n_tr] - clim[idx[:n_tr]]                              # [N,6,32,36]

    t0 = np.arange(2, n_tr - LEADS)                                    # last input day of each training sample
    coef = np.zeros((LEADS, 3, 6, 32, 36), "f4")
    sm = anom
    for k in range(LEADS):
        for _ in range(a.smooth):
            sm = smooth(sm, a.kappa)
        X = np.stack([sm[t0 - 2], sm[t0 - 1], sm[t0]], axis=1)         # [S,3,6,32,36], oldest first like the engine
        y = anom[t0 + k + 1]                                           # [S,6,32,36]
        xtx = np.einsum("slcij,smcij->cijlm", X, X, optimize=True)     # [6,32,36,3,3]
        xty = np.einsum("slcij,scij->cijl", X, y, optimize=True)
        lam = a.ridge * np.trace(xtx, axis1=-2, axis2=-1)[..., None, None] / 3
        b = np.linalg.solve(xtx + lam * np.eye(3), xty[..., None])[..., 0]   # [6,32,36,3]
        coef[k] = np.moveaxis(b, -1, 0)
        print(f"lead {k + 1:2d}: mean coef (t-2, t-1, t) =", np.round(coef[k].mean(axis=(1, 2, 3)), 3))
    out = ROOT / "assets" / "baseline.npz"
    np.savez(out, coef=coef, smooth=np.int32(a.smooth), kappa=np.float32(a.kappa))
    print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
