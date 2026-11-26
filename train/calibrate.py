"""Tune the output rain gain of a checkpoint on held-out years and store it in the checkpoint.

  python -m train.calibrate data/runs/r2.pt --out assets/weights.pt [--gains 1.0,1.1,1.2,1.3]

MSE training makes heavy rain (> 10 mm/day) too rare in individual windows, which costs CSI points. A single
multiplicative gain on the rain output (not fed back into the rollout) trades a little rain RMSE for CSI; the
gain with the best RMSE + CSI points on val (2020) is kept. 2021 (test) is only reported, never used to choose.
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
        o = evaluate(R.rollout, "val", 5, n_shock=1, verbose=False, arch=arch)
        pts = o["rmse_pts"] + o["csi_pts"]
        ot = evaluate(R.rollout, "test", 5, n_shock=1, verbose=False, arch=arch)
        print(f"gain {g}: rmse+csi points val 2020 {pts:.1f} (selection) | test 2021 {ot['rmse_pts'] + ot['csi_pts']:.1f} (report only)")
        if best is None or pts > best[1]:
            best = (g, pts)
    tmp.unlink()
    print(f"best gain {best[0]}")
    out = Path(a.out or a.ckpt)
    torch.save({**ck, "calib": {"rain_gain": best[0]}}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
