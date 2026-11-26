"""Kaal Barish inference server (built from the Round 1 kit starter). The judge runs: python serve.py --port 8000

Rules: listen on 0.0.0.0 (not 127.0.0.1), load weights from files next to this one (there is no internet), keep
the model loaded in memory (a 10-step rollout must return in under 5 s), return finite numbers in physical units.
Uses the CNN in model/rollout.py when assets/weights.pt exists, else the fallback in engine/baseline.py (CONTRACT.md).
"""
import argparse, sys
from pathlib import Path
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()
SHAPE = (6, 32, 36)
MAX_STEPS = 14
WARMUP = np.broadcast_to(np.array([295.0, 4.0, 2.0, 1.0, 100500.0, 35.0], "f4")[:, None, None], (3,) + SHAPE)


def _pick_model():
    """CNN if its weights load and give a valid warm-up rollout, else the baseline (a crash at startup scores 0).
    The warm-up also makes /health answer only once the model is in memory (both models load lazily)."""
    try:
        if not (Path(__file__).resolve().parent / "assets" / "weights.pt").exists():
            raise FileNotFoundError("assets/weights.pt")
        from model.rollout import rollout
        if not np.isfinite(rollout(WARMUP, [0.0, 1.0], 1, None)).all():
            raise ValueError("non-finite warm-up output")
        return rollout, "kaal-barish-cnn"
    except Exception as e:                                             # e.g. a Git LFS pointer instead of weights
        print(f"CNN unavailable ({type(e).__name__}: {str(e)[:200]}), serving the baseline", file=sys.stderr)
        from engine.baseline import rollout
        rollout(WARMUP, [0.0, 1.0], 1, None)
        return rollout, "kaal-barish-baseline"


rollout, MODEL_NAME = _pick_model()


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
