"""Combine checkpoints into one deterministic ensemble weights file (rollouts are averaged in model/rollout.py).

  python -m train.make_ensemble data/runs/r2.pt data/runs/r4.pt --out data/runs/ens.pt
  python -m train.calibrate data/runs/ens.pt --out assets/weights.pt        # rain gain chosen on 2020
"""
from __future__ import annotations
import argparse
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpts", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    members = []
    for p in a.ckpts:
        ck = torch.load(p, weights_only=True)
        members.append({"config": {"norm": "group", **ck["config"]}, "state_dict": ck["state_dict"], "mu": ck["mu"], "sd": ck["sd"]})
    torch.save({"members": members, "calib": {"rain_gain": 1.0}}, a.out)
    print(f"saved {a.out} with {len(members)} members")


if __name__ == "__main__":
    main()
