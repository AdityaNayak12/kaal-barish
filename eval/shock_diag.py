"""Why do shocks fail 'propagates'? Diagnostics for the CNN's shock response, per season (2020 + 2021 windows).
  python eval/shock_diag.py [--stride 5]

For each shock and season: kit pass rates, where the response energy sits (near the box / within the CNN's 7-cell
daily reach / beyond), how fast it decays, where its centroid drifts, which channels carry it, and the share of
the grid's edge band. Every shock is also rerun with FROZEN GroupNorm statistics (the shocked run reuses the
unshocked run's per-sample statistics, patched in-process here; model/ is not edited), which removes the only
path for a far-field response. The difference between normal and frozen is the GroupNorm leak.
Energy = sum over channels of (delta / kit std)^2, as in monsoon_r1.scoring.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from monsoon_r1 import contract as C, scoring as S                    # noqa: E402
from fit_baseline import load                                         # noqa: E402
import model.rollout as R                                             # noqa: E402

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJAS": (6, 7, 8, 9), "ON": (10, 11)}
SHOCKS = (("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA))
REACH = 7                                                              # cells per step: 3x3 input conv + 3 blocks x 2 convs
CTL = {"mode": "off", "n": 0, "i": 0, "stats": []}


def _gn_forward(m, x):
    """GroupNorm that records per-sample statistics in each member's unshocked rollout and replays them in that
    member's shocked rollout. rollout() runs, per ensemble member, n calls unshocked then n calls shocked."""
    if CTL["mode"] == "off":
        return F.group_norm(x, m.num_groups, m.weight, m.bias, m.eps)
    i = CTL["i"]; CTL["i"] += 1
    xg = x.reshape(x.shape[0], m.num_groups, -1)
    block, j = divmod(i, CTL["n"]) if CTL["mode"] == "freeze" else (0, i)
    if block % 2 == 0:
        mean, var = xg.mean(-1, keepdim=True), xg.var(-1, unbiased=False, keepdim=True)
        if CTL["mode"] == "freeze":
            if j == 0:
                CTL["stats"] = []
            CTL["stats"].append((mean, var))
    else:
        mean, var = CTL["stats"][j]
    y = ((xg - mean) / torch.sqrt(var + m.eps)).reshape_as(x)
    return y * m.weight[None, :, None, None] + m.bias[None, :, None, None]


def _run(mode, init, doy, pert):
    CTL.update(mode=mode, i=0, stats=[])
    try:
        return R.rollout(init, doy, C.SCORE_STEPS, pert)
    finally:
        CTL["mode"] = "off"


def frozen(init, doy, pert):
    """Shocked rollout whose GroupNorm statistics are the unshocked run's, member by member."""
    CTL["n"] = 1; _run("count", init, doy, None); CTL["n"] = CTL["i"] // len(R._load()["members"])
    return _run("freeze", init, doy, pert)


def cheb_dist(box):
    ii, jj = np.indices((C.NLAT, C.NLON)); bi, bj = np.where(C.box_mask(box))
    return np.maximum(np.abs(ii[..., None] - bi), np.abs(jj[..., None] - bj)).min(axis=-1)


def stats(base, shocked, box, std):
    d = shocked - base; e = ((d / std[None, :, None, None]) ** 2)          # [10,6,32,36]
    et = e.sum(axis=1); near = S.near_mask(box); far = cheb_dist(box) > REACH
    edge = np.zeros((C.NLAT, C.NLON), bool); edge[:2] = edge[-2:] = True; edge[:, :2] = edge[:, -2:] = True
    tot = et.sum(axis=(1, 2)); out = np.array([et[k][~near].sum() for k in range(len(et))])
    late = out[3:8].mean()
    lat_c = [(et[k].sum(axis=1) * C.LAT).sum() / max(tot[k], 1e-12) for k in (0, 3, 7)]
    lon_c = [(et[k].sum(axis=0) * C.LON).sum() / max(tot[k], 1e-12) for k in (0, 3, 7)]
    ch_late = e[3:8][:, :, ~near].sum(axis=(0, 2))
    c = S.causality_checks(base, shocked, box, std)
    return {"propagates": c["propagates"], "no_teleport": c["no_teleport"], "grows": late > 1.5 * out[0] + 1e-12,
            "big": late > 0.02 * tot.max(), "late/out1": late / max(out[0], 1e-12), "late/peak": late / max(tot.max(), 1e-12),
            "decay8": tot[7] / max(tot[0], 1e-12), "far1": et[0][far].sum() / max(tot[0], 1e-12),
            "edge_late": et[3:8][:, edge].sum() / max(et[3:8].sum(), 1e-12), "lat": lat_c, "lon": lon_c,
            "ch_late": ch_late / max(ch_late.sum(), 1e-12),
            "cells": [int(((np.abs(d[0, 0]) > 0.1) & m).sum()) for m in (near, ~near & ~far, far)]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stride", type=int, default=5); a = ap.parse_args()
    data, _, _, t = load()
    std = np.array(json.loads((ROOT / "assets" / "stats.json").read_text())["std"], "f4")
    for member in R._load()["members"]:
        for m in member["net"].modules():
            if isinstance(m, torch.nn.GroupNorm):
                m.forward = lambda x, m=m: _gn_forward(m, x)
    i = np.arange(len(t))
    # sanity: with a shock that vanishes when added, replayed statistics equal recomputed ones, so frozen == normal.
    # (normal(tiny) != base by up to ~0.02 std: the shocked path round-trips physical units between steps; harmless)
    w0 = int(np.where(t == "2021-07-16")[0][0]); x0 = data[w0 - 2:w0 + 1]; dy0 = C.doy_sincos(int(t.day_of_year[w0]))
    tiny = np.full(x0.shape[1:], 1e-30, "f4")
    print("sanity max |frozen(tiny) - normal(tiny)| / std:",
          float((np.abs(frozen(x0, dy0, tiny) - R.rollout(x0, dy0, C.SCORE_STEPS, tiny)) / std[None, :, None, None]).max()))
    print(f"REACH = {REACH} cells/step (Chebyshev); 'far' = beyond it on day 1, reachable only through GroupNorm\n")
    for n, box in SHOCKS:
        print(f"=== {n}")
        print(f"{'season':6s} {'mode':6s} {'n':>3s} {'prop':>5s} {'grows':>5s} {'>2%pk':>5s} {'noTel':>5s} {'late/out1':>9s} "
              f"{'late/pk':>7s} {'E8/E1':>6s} {'far1':>6s} {'edge':>5s}  centroid lat,lon d1>d4>d8         late outside energy by channel")
        for season, months in SEASONS.items():
            wins = np.where(np.isin(t.year, (2020, 2021)) & np.isin(t.month, months) & (i >= 2) & (i + C.SCORE_STEPS < len(t)))[0][::a.stride]
            rows = {"normal": [], "frozen": []}; wind = []
            for w in wins:
                init = data[w - 2:w + 1]; doy = C.doy_sincos(int(t.day_of_year[w]))
                base = R.rollout(init, doy, C.SCORE_STEPS, None); pert = S.make_shock(n, init[-1], std)
                rows["normal"].append(stats(base, R.rollout(init, doy, C.SCORE_STEPS, pert), box, std))
                rows["frozen"].append(stats(base, frozen(init, doy, pert), box, std))
                wind.append(init[-1, 2:4][:, C.box_mask(box)].mean(axis=1))
            for mode, rs in rows.items():
                g = lambda k: np.mean([r[k] for r in rs], axis=0); med = lambda k: np.median([r[k] for r in rs])
                lat, lon = g("lat"), g("lon")
                print(f"{season:6s} {mode:6s} {len(rs):3d} {g('propagates'):5.0%} {g('grows'):5.0%} {g('big'):5.0%} {g('no_teleport'):5.0%} "
                      f"{med('late/out1'):9.2f} {med('late/peak'):7.3f} {med('decay8'):6.3f} {med('far1'):6.3f} {g('edge_late'):5.0%}  "
                      f"{lat[0]:4.1f},{lon[0]:5.1f} > {lat[1]:4.1f},{lon[1]:5.1f} > {lat[2]:4.1f},{lon[2]:5.1f}  "
                      + " ".join(f"{c}={v:.0%}" for c, v in zip(C.CHANNELS, g("ch_late"))))
            u, v = np.mean(wind, axis=0)
            print(f"{'':6s} box-mean wind in last input day: u10 {u:+.1f} v10 {v:+.1f} m/s")
            if n == "warm_ocean":
                for mode, rs in rows.items():
                    c = np.array([r["cells"] for r in rs])
                    print(f"{'':6s} {mode} day-1 cells |dt2m|>0.1K  near(box+3) {c[:, 0].mean():.0f}  "
                          f"ring(4..{REACH}) mean {c[:, 1].mean():.1f}  beyond {REACH}: mean {c[:, 2].mean():.1f} max {c[:, 2].max()}")
        print()


if __name__ == "__main__":
    main()
