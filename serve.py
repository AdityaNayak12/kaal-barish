"""Kaal Barish inference server (built from the Round 1 kit starter). The judge runs: python serve.py --port 8000

Rules: listen on 0.0.0.0 (not 127.0.0.1), load weights from files next to this one (there is no internet), keep
the model loaded in memory (a 10-step rollout must return in under 5 s), return finite numbers in physical units.
Uses the CNN in model/rollout.py when assets/weights.pt exists, else the fallback in engine/baseline.py (CONTRACT.md).
"""
import argparse
from pathlib import Path
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

if (Path(__file__).resolve().parent / "assets" / "weights.pt").exists():
    from model.rollout import rollout
    MODEL_NAME = "kaal-barish-cnn"
else:
    from engine.baseline import rollout
    MODEL_NAME = "kaal-barish-baseline"

app = FastAPI()
SHAPE = (6, 32, 36)
MAX_STEPS = 14


def _arr(x, shape):
    """Accept nested lists (the contract) or {'b64': ..., 'shape': [...]} (tolerated). Rejects the wrong shape."""
    if isinstance(x, dict):
        import base64
        a = np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    else:
        a = np.asarray(x, dtype="f4")
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"expected finite array of shape {shape}, got {a.shape}")
    return a


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """init_states [3,6,32,36] physical units, oldest first. perturbation [6,32,36] delta or None.
    Return [steps,6,32,36] physical units. doy_sincos = [sin, cos] of the day of year of init_states[-1]."""
    return np.asarray(rollout(init_states, doy_sincos, steps, perturbation), dtype="f4")


# warm-up so /health only answers once the model is loaded (both models load their assets on first call)
simulate(np.broadcast_to(np.array([295.0, 4.0, 2.0, 1.0, 100500.0, 35.0], "f4")[:, None, None], (3,) + SHAPE), [0.0, 1.0], 1, None)


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
