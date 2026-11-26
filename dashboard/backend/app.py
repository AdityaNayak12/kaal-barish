"""Dashboard backend, port 8001.

GET  /samples        list of sample windows from the 2021 test year
GET  /samples/{id}   init_states [3,6,32,36], doy_sincos, truth [10,6,32,36]
GET  /grid           lat, lon and land mask of the 32 x 36 grid
POST /health         and the forecast route are passed through to the model server (model_proxy.py)

Run from the repo root:  .venv/bin/python dashboard/backend/app.py
"""
import argparse
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_proxy  # noqa: E402
from samples import SampleStore  # noqa: E402

app = FastAPI(title="Kaal Barish dashboard backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(model_proxy.router)

store = SampleStore()


@app.get("/samples")
def list_samples():
    return store.list_samples()


@app.get("/samples/{sample_id}")
def get_sample(sample_id: str):
    sample = store.get_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"unknown sample {sample_id}")
    return sample


@app.get("/grid")
def grid():
    return store.grid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
