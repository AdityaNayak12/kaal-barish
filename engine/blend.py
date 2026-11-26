"""Post-processing of the CNN's unperturbed forecast: blend it with the fallback regression (engine/baseline.py).

For t2m, u10, msl and tcwv at lead k: out = clim + a[k] * (cnn - clim) + b[k] * (baseline - clim). The CNN is
the better forecast for the first days; the per-cell regression adds skill at days 4-8 (most of it for t2m and tcwv).
Weights come from eval/fit_blend.py (weighted least squares on the validation year). Rain and v10 are never touched,
so the heavy-rain scores stay those of the CNN. serve.py adds the CNN's own shock response on top of the blended base.
"""
import json
from pathlib import Path
import numpy as np
from engine.baseline import _load as _baseline_state, doy_index

_W = {}


def _load() -> dict:
    if not _W:
        z = json.loads((Path(__file__).resolve().parent / "blend.json").read_text())
        _W.update(ch=np.array(z["channels"]), a=np.array(z["a"], "f4"), b=np.array(z["b"], "f4"))
    return _W


def clim_rows(doy_sincos, steps: int) -> np.ndarray:
    """Climatology for forecast days 1..steps, the same rows engine/baseline.py uses. [steps, 6, 32, 36]"""
    i0 = doy_index(doy_sincos)
    return _baseline_state()["clim"][[(i0 + k + 1) % 366 for k in range(steps)]]


def blend(cnn: np.ndarray, base: np.ndarray, doy_sincos) -> np.ndarray:
    """cnn, base: unperturbed rollouts [steps, 6, 32, 36]. Leads past the fitted ones reuse the last weights."""
    w = _load(); ch = w["ch"]; c = clim_rows(doy_sincos, len(cnn))[:, ch]
    k = np.minimum(np.arange(len(cnn)), len(w["a"]) - 1)
    a, b = w["a"][k][:, :, None, None], w["b"][k][:, :, None, None]
    out = cnn.copy()
    out[:, ch] = c + a * (cnn[:, ch] - c) + b * (base[:, ch] - c)
    out[:, 0] = np.clip(out[:, 0], 233.15, 333.15)
    return out
