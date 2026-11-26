"""Inference entry point (see CONTRACT.md, model/rollout.py section).

Loads assets/weights.pt, assets/clim.npy and assets/static.npy once, on the first call. Never reads data/.
Deterministic: CPU, no dropout, no sampling.
"""
from __future__ import annotations
from pathlib import Path
import threading
import numpy as np
import torch

from model.net import StepNet, clim_rows, doy_features, doy_from_sincos, from_z, static_features, to_z

ASSETS = Path(__file__).resolve().parents[1] / "assets"
WEIGHTS = ASSETS / "weights.pt"
T_MIN, T_MAX = 233.15, 333.15

_state = None
_lock = threading.Lock()


def _load():
    global _state
    with _lock:
        if _state is not None:
            return _state
        ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        net = StepNet(**ck["config"])
        net.load_state_dict(ck["state_dict"])
        net.eval()
        mu, sd = ck["mu"].float(), ck["sd"].float()
        clim = torch.from_numpy(np.load(ASSETS / "clim.npy").astype("f4"))
        static = torch.from_numpy(np.load(ASSETS / "static.npy").astype("f4"))
        rain_gain = float(ck.get("calib", {}).get("rain_gain", 1.0))   # output-only heavy-rain calibration
        _state = {"net": net, "mu": mu, "sd": sd, "z_clim": to_z(clim, mu, sd), "static": static_features(static),
                  "rain_gain": rain_gain}
        return _state


def model_name() -> str:
    return "kaal-barish-rescnn"


def rollout(init_states, doy_sincos, steps, perturbation=None) -> np.ndarray:
    """init_states [3,6,32,36] physical, oldest first; perturbation [6,32,36] physical delta or None.
    Returns float32 [steps,6,32,36] physical; output[k] is day k+1 after init_states[-1]."""
    st = _load()
    steps = int(steps)
    x = np.array(init_states, dtype="f4", copy=True)
    if perturbation is not None:
        x[-1] = x[-1] + np.asarray(perturbation, dtype="f4")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    d0 = doy_from_sincos(float(doy_sincos[0]), float(doy_sincos[1]))
    rows = clim_rows(d0, steps)
    doy_feat = doy_features(torch.tensor([d0 + k for k in range(1, steps + 1)], dtype=torch.float64)).float()

    with torch.inference_mode():
        z = to_z(torch.from_numpy(x), st["mu"], st["sd"])[None]
        zc = st["z_clim"][rows][None]
        out = st["net"].rollout_z(z, zc, st["static"], doy_feat[None])[0]
        y = from_z(out, st["mu"], st["sd"]).numpy().astype("f4")

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y[:, 1] = np.maximum(y[:, 1] * st["rain_gain"], 0.0)
    y[:, 0] = np.clip(y[:, 0], T_MIN, T_MAX)
    return y
