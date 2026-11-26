"""Fallback model: per-cell linear regression of future anomalies (vs climatology) on the 3 input-day anomalies.

Before each lead the anomalies get one more round of neighbour smoothing, so a shock spreads about one cell per
day instead of staying put (perturbation propagation) and long leads relax towards climatology.
Coefficients come from eval/fit_baseline.py. Same signature as model/rollout.py (see CONTRACT.md).
"""
from pathlib import Path
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"
_S = {}


def _load() -> dict:
    if not _S:
        z = np.load(ASSETS / "baseline.npz")
        _S.update(coef=z["coef"].astype("f4"), smooth=int(z["smooth"]), kappa=float(z["kappa"]),
                  clim=np.load(ASSETS / "clim.npy").astype("f4"))
    return _S


def doy_index(doy_sincos) -> int:
    """[sin, cos] of 2*pi*doy/365 -> climatology row 0..365. Takes doy as 1-based; around Feb 29 it can be 1 day off."""
    ang = float(np.arctan2(doy_sincos[0], doy_sincos[1])) % (2 * np.pi)
    doy = int(round(ang * 365 / (2 * np.pi))) % 365
    return (doy - 1) % 366


def smooth(a: np.ndarray, kappa: float) -> np.ndarray:
    """Blend each cell with the mean of its 4 neighbours (edge cells reuse themselves). Works on [..., 32, 36]."""
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    nb = 0.25 * (p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2] + p[..., 1:-1, 2:])
    return (1 - kappa) * a + kappa * nb


def rollout(init_states, doy_sincos, steps, perturbation=None) -> np.ndarray:
    s = _load(); coef, clim = s["coef"], s["clim"]
    x = np.array(init_states, dtype="f4")
    if perturbation is not None:
        x[-1] += np.asarray(perturbation, dtype="f4")
    i0 = doy_index(doy_sincos)
    a = x - clim[[(i0 - 2) % 366, (i0 - 1) % 366, i0]]            # anomalies, oldest first
    out = np.empty((steps,) + x.shape[1:], "f4")
    for k in range(steps):
        for _ in range(s["smooth"]):
            a = smooth(a, s["kappa"])
        c = coef[min(k, len(coef) - 1)]                               # [3, 6, 32, 36]
        out[k] = clim[(i0 + k + 1) % 366] + (c * a).sum(axis=0)
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    out[:, 0] = np.clip(out[:, 0], 233.15, 333.15)
    return out
