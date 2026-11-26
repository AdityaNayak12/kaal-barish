"""Per-shock causality report (kit scoring.causality_checks) for a checkpoint, per split.

  python -m train.shock_report [--weights data/runs/x.pt] [--stride 7]

Prints, for warm_ocean and monsoon_stall separately, the share of windows passing localized / propagates /
stable / no_teleport and the mean points per shock (out of 7.5).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from monsoon_r1 import scoring as S
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, doy_sincos

from train.data import ASSETS, doy, load_archive
from train.evaluate import windows

SHOCKS = (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA))


def report(fn, arch, splits=("val", "test"), stride=7) -> dict:
    data, times, _, _, _ = arch
    std = np.array(json.loads((ASSETS / "stats.json").read_text())["std"], "f4"); dy = doy(times)
    out = {}
    for split in splits:
        res = {n: [] for n, _ in SHOCKS}
        for i in windows(times, split, stride):
            init = data[i - 2:i + 1]; sc = doy_sincos(int(dy[i])); base = fn(init, sc, 10, None)
            for n, box in SHOCKS:
                res[n].append(S.causality_checks(base, fn(init, sc, 10, S.make_shock(n, init[-1], std)), box, std))
        for n, rs in res.items():
            f = lambda k: float(np.mean([r[k] for r in rs]))
            out[(split, n)] = {k: f(k) for k in ("localized", "propagates", "stable", "no_teleport")}
            out[(split, n)]["pts"] = float(np.mean([S.causality_points([r], True) / 2 for r in rs]))
            o = out[(split, n)]
            print(f"  {split:4s} {n:14s} n={len(rs)} localized {o['localized']:.2f} propagates {o['propagates']:.2f} "
                  f"stable {o['stable']:.2f} no_teleport {o['no_teleport']:.2f} | {o['pts']:.2f}/7.5")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", default=None); ap.add_argument("--stride", type=int, default=7)
    a = ap.parse_args()
    import model.rollout as R
    if a.weights:
        R.WEIGHTS = Path(a.weights)
    report(R.rollout, load_archive(), stride=a.stride)


if __name__ == "__main__":
    main()
