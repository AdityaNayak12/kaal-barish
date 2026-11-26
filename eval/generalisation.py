"""Overfitting / generalisation check of the CNN (model.rollout, assets/weights.pt) with the kit's scoring.
  python eval/generalisation.py [--windows 40] [--season-stride 5]

1. Train vs held-out: the eval/score_local.py recipe (same window count and stride) on training years 2012 and 2017,
   val 2020 and test 2021. Flags possible overfitting if train beats held-out by mean SS > 0.03 or CSI points > 2.
2. Shocks per season (2020 + 2021 windows): warm_ocean, monsoon_stall and null with the kit's causality checks,
   plus warm_ocean day-1 cells with |dt2m| > 0.1 K (all, and outside the box + 3-cell border).
3. Leakage, numerically: checkpoint mu/sd and assets/clim.npy recomputed from 2005-2019 vs from wider year ranges.
Calls model.rollout directly (not serve.py), so a CNN failure is not hidden by the baseline fallback.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from monsoon_r1 import contract as C, scoring as S                    # noqa: E402
from monsoon_r1.stats import climatology                               # noqa: E402
from fit_baseline import load                                         # noqa: E402
from score_local import score_windows, year_windows                   # noqa: E402
from model.rollout import rollout                                     # noqa: E402

SPLITS = (("train 2012", 2012, True), ("train 2017", 2017, True), ("val 2020", 2020, False), ("test 2021", 2021, False))
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJAS": (6, 7, 8, 9), "ON": (10, 11)}
SHOCKS = (("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA))
CHECKS = ("nonzero", "localized", "propagates", "stable", "no_teleport")


def train_vs_heldout(data, idx, t, clim, std, n):
    print(f"== 1. train vs held-out ({n} windows per year, same stride)")
    print(f"{'split':11s} {'mean SS':>14s} {'RMSE /30':>8s} {'CSI skill':>9s} {'CSI /15':>7s} {'blur /10':>8s} {'causal /15':>10s}")
    res = {}
    for name, year, _ in SPLITS:
        r = score_windows(rollout, data, idx, t, clim, std, year_windows(t, year, n)); fp = r["fp"]
        res[name] = r
        print(f"{name:11s} {fp['mean_ss']:7.3f}        {fp['rmse_pts']:8.2f} {fp['mean_csi_skill']:9.3f} {fp['csi_pts']:7.2f} "
              f"{r['blur']:8.2f} {r['caus_pts']:10.2f}   " + " ".join(f"{k}={v:.0%}" for k, v in r["checks"].items()))
    mean = lambda key, train: np.mean([res[nm]["fp"][key] for nm, _, tr in SPLITS if tr == train])
    ss_gap, csi_gap = mean("mean_ss", True) - mean("mean_ss", False), mean("csi_pts", True) - mean("csi_pts", False)
    flag = ss_gap > 0.03 or csi_gap > 2
    print(f"gap train - held-out: mean SS {ss_gap:+.3f} (flag > 0.03), CSI points {csi_gap:+.2f} (flag > 2) "
          f"=> {'POSSIBLE OVERFITTING' if flag else 'no overfitting flag'}\n")


def shocks_by_season(data, t, std, stride):
    near = S.near_mask(C.BOX_BAY_OF_BENGAL)
    print(f"== 2. shocks per season (2020 + 2021, every {stride}th day; box + 3-cell border = {near.sum()} cells)")
    print(f"{'season':6s} {'shock':14s} {'n':>3s} " + " ".join(f"{k:>11s}" for k in CHECKS) + f" {'null_ok':>8s}")
    i = np.arange(len(t))
    for season, months in SEASONS.items():
        wins = np.where(np.isin(t.year, (2020, 2021)) & np.isin(t.month, months) & (i >= 2) & (i + C.SCORE_STEPS < len(t)))[0][::stride]
        rates = {n: [] for n, _ in SHOCKS}; null_ok = []; cells = []; outside = []
        for w in wins:
            init = data[w - 2:w + 1]; doy = C.doy_sincos(int(t.day_of_year[w]))
            base = rollout(init, doy, C.SCORE_STEPS, None)
            null_ok.append(S.null_consistent(base, rollout(init, doy, C.SCORE_STEPS, np.zeros(init.shape[1:], "f4")), std))
            for n, box in SHOCKS:
                shocked = rollout(init, doy, C.SCORE_STEPS, S.make_shock(n, init[-1], std))
                rates[n].append(S.causality_checks(base, shocked, box, std))
                if n == "warm_ocean":
                    moved = np.abs(shocked[0, 0] - base[0, 0]) > 0.1
                    cells.append(int(moved.sum())); outside.append(int((moved & ~near).sum()))
        for n, _ in SHOCKS:
            print(f"{season:6s} {n:14s} {len(wins):3d} " + " ".join(f"{np.mean([r[k] for r in rates[n]]):11.0%}" for k in CHECKS)
                  + f" {np.mean(null_ok):8.0%}")
        print(f"{'':6s} warm_ocean day-1 cells |dt2m|>0.1K: mean {np.mean(cells):.0f} max {max(cells)}; "
              f"outside box+3: mean {np.mean(outside):.1f} median {np.median(outside):.0f} max {max(outside)}")
    print()


def leakage(data, ymd, t):
    print("== 3. leakage (numeric): which years reproduce the shipped statistics")
    ck = torch.load(ROOT / "assets" / "weights.pt", map_location="cpu", weights_only=True)
    clim = np.load(ROOT / "assets" / "clim.npy")
    x = torch.from_numpy(data)
    for y1 in (2019, 2020, 2021):
        m = torch.from_numpy(np.asarray(t.year <= y1))
        xt = x[m].clone(); xt[:, 1] = torch.log1p(xt[:, 1].clamp_min(0))       # same transform as train/train.py
        mu, sd = xt.mean(dim=(0, 2, 3)), xt.std(dim=(0, 2, 3))
        d_ms = max(float((mu - ck["mu"]).abs().max()), float((sd - ck["sd"]).abs().max()))
        n = int(m.sum()); d_clim = float(np.abs(climatology(data[:n], ymd[:n]) - clim).max())
        print(f"  2005-{y1}: max |mu/sd - checkpoint| {d_ms:.2e}   max |clim - assets/clim.npy| {d_clim:.2e}")
    print(f"  checkpoint calib: {ck.get('calib')}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=40); ap.add_argument("--season-stride", type=int, default=5)
    a = ap.parse_args()
    data, ymd, idx, t = load()
    clim = np.load(ROOT / "assets" / "clim.npy").astype("f4")
    std = np.array(json.loads((ROOT / "assets" / "stats.json").read_text())["std"], "f4")
    train_vs_heldout(data, idx, t, clim, std, a.windows)
    shocks_by_season(data, t, std, a.season_stride)
    leakage(data, ymd, t)


if __name__ == "__main__":
    main()
