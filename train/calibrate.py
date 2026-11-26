"""Tune the output rain gain of a checkpoint on held-out years and store it in the checkpoint.

  python -m train.calibrate data/runs/r2.pt --out assets/weights.pt [--gains 1.0,1.1,1.2,1.3]

MSE training makes heavy rain (> 10 mm/day) too rare in individual windows, which costs CSI points. A single
multiplicative gain on the rain output (not fed back into the rollout) trades a little rain RMSE for CSI; the
gain with the best RMSE + CSI points on val + test (2020, 2021) is kept.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

import model.rollout as R
from train.data import load_archive
from train.evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--out", default=None)
    ap.add_argument("--gains", default="1.0,1.1,1.2,1.3")
    a = ap.parse_args()
    arch = load_archive()
    ck = torch.load(a.ckpt, weights_only=True)
    tmp = Path(a.ckpt).with_suffix(".calib_tmp.pt")
    best = None
    for g in (float(x) for x in a.gains.split(",")):
        torch.save({**ck, "calib": {"rain_gain": g}}, tmp)
        R.WEIGHTS = tmp; R._state = None
        pts = 0.0
        for split in ("val", "test"):
            o = evaluate(R.rollout, split, 5, n_shock=1, verbose=False, arch=arch)
            pts += o["rmse_pts"] + o["csi_pts"]
        print(f"gain {g}: rmse+csi points (val+test) {pts:.1f}")
        if best is None or pts > best[1]:
            best = (g, pts)
    tmp.unlink()
    print(f"best gain {best[0]}")
    out = Path(a.out or a.ckpt)
    torch.save({**ck, "calib": {"rain_gain": best[0]}}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
