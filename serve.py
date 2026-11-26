"""Kaal Barish inference server (built from the Round 1 kit starter). The judge runs: python serve.py --port 8000

Rules: listen on 0.0.0.0 (not 127.0.0.1), load weights from files next to this one (there is no internet), keep
the model loaded in memory (a 10-step rollout must return in under 5 s), return finite numbers in physical units.
Uses the CNN in model/rollout.py when assets/weights.pt loads, else the fallback in engine/baseline.py (CONTRACT.md).
The CNN's unperturbed forecast is blended with the fallback's (engine/blend.py); a shock adds the CNN's own response.
A rejected or invalid answer scores 0 for that window, so the server repairs what it can instead of refusing.
"""
import argparse, hashlib, os, sys
# Judge container: 4 CPUs on a bigger host. Older torch sizes its thread pools from the host's core count and
# set_num_threads(4) alone left 3 s latency spikes, so cap the pools before numpy/torch load.
os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from engine.baseline import rollout as baseline
from engine.blend import blend

app = FastAPI()
SHAPE = (6, 32, 36)
MAX_STEPS = 30
WARMUP = np.broadcast_to(np.array([295.0, 4.0, 2.0, 1.0, 100500.0, 35.0], "f4")[:, None, None], (3,) + SHAPE)


def _valid(out, steps) -> bool:
    """Same gate as the grader's validity check: shape, finite, rain >= 0, t2m in 233.15..333.15 K."""
    return (out.shape == (steps,) + SHAPE and bool(np.isfinite(out).all()) and out[:, 1].min() >= 0
            and 233.15 <= out[:, 0].min() and out[:, 0].max() <= 333.15)


def _pick_model():
    """CNN if its weights load and give a valid warm-up rollout, else the baseline (a crash at startup scores 0).
    Warm-up runs before uvicorn starts, so /health only answers once the models are in memory."""
    baseline(WARMUP, [0.0, 1.0], 1, None)                             # always loaded: it is the per-request fallback
    try:
        if not (Path(__file__).resolve().parent / "assets" / "weights.pt").exists():
            raise FileNotFoundError("assets/weights.pt")
        import torch
        torch.set_num_threads(4)                                       # judge container: 4 CPUs
        from model.rollout import rollout
        if not _valid(np.asarray(rollout(WARMUP, [0.0, 1.0], 1, None), "f4"), 1):
            raise ValueError("invalid warm-up output")
        return rollout, "kaal-barish-cnn"
    except Exception as e:                                             # e.g. a Git LFS pointer instead of weights
        print(f"CNN unavailable ({type(e).__name__}: {str(e)[:200]}), serving the baseline", file=sys.stderr)
        return baseline, "kaal-barish-baseline"


rollout, MODEL_NAME = _pick_model()


def _arr(x, shape):
    """Accept nested lists (the contract) or {'b64': ..., 'shape': [...]} (tolerated). Rejects only a wrong shape."""
    if isinstance(x, dict):
        import base64
        a = np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    else:
        a = np.asarray(x, dtype="f4")
    if a.shape != shape:
        raise ValueError(f"expected shape {shape}, got {a.shape}")
    return a


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """init_states [3,6,32,36] physical units, oldest first. perturbation [6,32,36] delta or None.
    Return [steps,6,32,36] physical units. doy_sincos = [sin, cos] of the day of year of init_states[-1]."""
    init_states = np.asarray(init_states, dtype="f4")
    fin = np.isfinite(init_states)
    if not fin.all():                                                  # fill gaps with the channel's mean over all 3 days
        fill = np.where(fin, init_states, 0).sum(axis=(0, 2, 3), keepdims=True) / np.maximum(fin.sum(axis=(0, 2, 3), keepdims=True), 1)
        init_states = np.where(fin, init_states, fill).astype("f4")
    if perturbation is not None:
        perturbation = np.nan_to_num(np.asarray(perturbation, dtype="f4"), nan=0.0, posinf=0.0, neginf=0.0)
        if not perturbation.any():
            perturbation = None                                        # all-zeros = null shock: identical to no shock
    if rollout is not baseline:
        try:
            cnn, out = _unperturbed(init_states, doy_sincos, steps)
            if perturbation is None:
                return out.copy()
            shocked = np.asarray(rollout(init_states, doy_sincos, steps, perturbation), dtype="f4")
            if _valid(shocked, steps):
                out = out + (shocked - cnn)                            # the CNN's own shock response on the blended base
                out[:, 1] = shocked[:, 1]                              # rain is not blended: keep it bit-identical
                if _valid(out, steps):
                    return out
            print("CNN output invalid, baseline used for this request", file=sys.stderr)
        except Exception as e:
            print(f"CNN failed ({type(e).__name__}: {str(e)[:200]}), baseline used for this request", file=sys.stderr)
    return np.asarray(baseline(init_states, doy_sincos, steps, perturbation), dtype="f4")


_RECENT = {}                                                           # last unperturbed runs, so a shock call reuses its base


def _unperturbed(init_states, doy_sincos, steps):
    """(CNN rollout, blended rollout) without perturbation. Raises if the CNN output is invalid."""
    key = (hashlib.sha1(init_states.tobytes()).hexdigest(), tuple(float(v) for v in doy_sincos), int(steps))
    if key not in _RECENT:
        cnn = np.asarray(rollout(init_states, doy_sincos, steps, None), dtype="f4")
        if not _valid(cnn, steps):
            raise ValueError("invalid unperturbed CNN rollout")
        try:
            out = blend(cnn, np.asarray(baseline(init_states, doy_sincos, steps, None), dtype="f4"), doy_sincos)
        except Exception as e:                                         # the post-processing must never cost the CNN
            print(f"blend failed ({type(e).__name__}: {str(e)[:200]}), plain CNN used", file=sys.stderr)
            out = cnn
        if len(_RECENT) >= 8:
            _RECENT.pop(next(iter(_RECENT)))
        _RECENT[key] = (cnn, out if _valid(out, steps) else cnn)
    return _RECENT[key]


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    import time
    t0 = time.perf_counter(); body = await request.json()
    try:
        init = _arr(body["init_states"], (3,) + SHAPE)
        pert = None if body.get("perturbation") is None else _arr(body["perturbation"], SHAPE)
        doy = [float(v) for v in body["doy_sincos"]]; steps = int(body["steps"])
        if len(doy) != 2 or not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"need doy_sincos [sin, cos] and 1 <= steps <= {MAX_STEPS}")
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"bad request: {e}")
    out = simulate(init, doy, steps, pert)
    return {"rollout": out.tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
