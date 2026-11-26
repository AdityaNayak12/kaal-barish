"""Inference entry point (see CONTRACT.md, model/rollout.py section).

Loads assets/weights.pt, assets/clim.npy and assets/static.npy once, on the first call. Never reads data/.
Deterministic: CPU, no dropout, no sampling.
"""
from __future__ import annotations
from pathlib import Path
import json
import threading
import numpy as np
import torch

from model.net import StepNet, clim_rows, doy_features, doy_from_sincos, from_z, static_features, to_z

ASSETS = Path(__file__).resolve().parents[1] / "assets"
WEIGHTS = ASSETS / "weights.pt"
T_MIN, T_MAX = 233.15, 333.15
# Shock response limiter: a perturbed run may differ from the unperturbed run by at most this many channel stds
# (kit stats.json) per cell, enforced at every step on the state that is fed forward. It bounds the chaotic
# rain blow-ups a small shock can set off (scenario runs only; unperturbed forecasts are untouched).
RESP_LIMIT = 4.5

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
        calib = ck.get("calib", {})
        rain_gain = float(calib.get("rain_gain", 1.0))                  # output-only heavy-rain calibration
        kit_std = torch.tensor(json.loads((ASSETS / "stats.json").read_text())["std"], dtype=torch.float32)
        lim = float(calib.get("resp_limit", RESP_LIMIT)) * kit_std
        lim[1] = lim[1] / rain_gain                                       # limit applies after the rain gain
        _state = {"net": net, "mu": mu, "sd": sd, "z_clim": to_z(clim, mu, sd), "static": static_features(static),
                  "rain_gain": rain_gain, "lim": lim[:, None, None]}
        return _state


def model_name() -> str:
    return "kaal-barish-rescnn"


def rollout(init_states, doy_sincos, steps, perturbation=None) -> np.ndarray:
    """init_states [3,6,32,36] physical, oldest first; perturbation [6,32,36] physical delta or None.
    Returns float32 [steps,6,32,36] physical; output[k] is day k+1 after init_states[-1]."""
    st = _load()
    steps = int(steps)
    x = np.nan_to_num(np.array(init_states, dtype="f4", copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    pert = None
    if perturbation is not None:
        pert = np.nan_to_num(np.asarray(perturbation, dtype="f4"), nan=0.0, posinf=0.0, neginf=0.0)
        if not pert.any():
            pert = None                                   # null shock: exactly the unperturbed computation

    d0 = doy_from_sincos(float(doy_sincos[0]), float(doy_sincos[1]))
    rows = clim_rows(d0, steps)
    doy_feat = doy_features(torch.tensor([d0 + k for k in range(1, steps + 1)], dtype=torch.float64)).float()
    zc = st["z_clim"][rows][None]

    with torch.inference_mode():
        base = from_z(st["net"].rollout_z(to_z(torch.from_numpy(x), st["mu"], st["sd"])[None], zc, st["static"],
                                          doy_feat[None])[0], st["mu"], st["sd"])
        if pert is None:
            y = base
        else:
            xs = x.copy(); xs[-1] = xs[-1] + pert
            y = _limited_rollout(st, xs, base, zc, doy_feat)

    return _finish(y.numpy().astype("f4"), st["rain_gain"])


def _limited_rollout(st, x, base, zc, doy_feat):
    """Perturbed rollout whose per-step state stays within +-lim of the unperturbed one (physical units)."""
    net, mu, sd, lim = st["net"], st["mu"], st["sd"], st["lim"]
    z = to_z(torch.from_numpy(x), mu, sd)[None]
    outs = []
    for k in range(base.shape[0]):
        nz = net(z, zc[:, k], st["static"], doy_feat[None, k])
        y = from_z(nz, mu, sd)[0]
        y = base[k] + torch.maximum(torch.minimum(y - base[k], lim), -lim)
        outs.append(y)
        z = torch.cat([z[:, 1:], to_z(y, mu, sd)[None, None]], dim=1)
    return torch.stack(outs)


def _finish(y: np.ndarray, rain_gain: float) -> np.ndarray:
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y[:, 1] = np.maximum(y[:, 1] * rain_gain, 0.0)
    y[:, 0] = np.clip(y[:, 0], T_MIN, T_MAX)
    return y
