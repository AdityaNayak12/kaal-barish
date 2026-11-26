"""Training-time validation with the kit's own scoring functions (monsoon_r1.scoring).

  python -m train.evaluate [--weights data/runs/x.pt] [--split val] [--stride 5]

Mirrors the judge: 10-step rollouts, RMSE skill vs the better of persistence/climatology at leads 1, 4, 8,
heavy-rain CSI, day-8 variance ratio (anti-blur), and the two shocks + null on monsoon windows.
This is a training aid for pawar; the team's official harness is eval/score_local.py (nayak).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from monsoon_r1 import scoring as S
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, LEAD_DAYS, doy_sincos

from train.data import ASSETS, clim_index, doy, load_archive, year_mask

STEPS = 10


def persistence_fn(init, sc, steps, pert):
    last = init[-1] + (0 if pert is None else pert)
    return np.repeat(last[None], steps, axis=0)


def windows(times, split, stride, steps=STEPS):
    m = year_mask(times, split)
    idx = np.where(m)[0]
    lo, hi = idx.min() + 2, idx.max() - steps
    return list(range(lo, hi + 1, stride))


def evaluate(fn, split="val", stride=5, n_shock=4, verbose=True, arch=None) -> dict:
    data, times, _, _, _ = arch if arch is not None else load_archive()
    std = np.array(json.loads((ASSETS / "stats.json").read_text())["std"], "f4")
    clim = np.load(ASSETS / "clim.npy")
    dy = doy(times)
    ss_w, csi_w, ratios, rm, rp, rc, t_ms, invalid = [], [], [], [], [], [], [], 0
    wins = windows(times, split, stride)
    for i in wins:
        init, truth = data[i - 2:i + 1], data[i + 1:i + 1 + STEPS]
        sc = doy_sincos(int(dy[i]))
        t0 = time.perf_counter(); pred = fn(init, sc, STEPS, None); t_ms.append(1000 * (time.perf_counter() - t0))
        if S.validity(pred, STEPS):
            invalid += 1; ss_w.append(0.0); csi_w.append(0.0); continue
        persist = np.repeat(init[-1][None], STEPS, 0)
        cref = clim[clim_index(dy[i] + np.arange(1, STEPS + 1))]
        L = [l - 1 for l in LEAD_DAYS]
        sk = S.skill_vs_best_ref(pred[L], truth[L], persist[L], cref[L])
        ss_w.append(float(np.nanmean(sk["ss"])))
        rm.append([S.rmse_lat(pred[l], truth[l]) for l in L]); rp.append([S.rmse_lat(persist[l], truth[l]) for l in L])
        rc.append([S.rmse_lat(cref[l], truth[l]) for l in L])
        c_m = S.csi(pred[L, 1], truth[L, 1]); c_p = S.csi(persist[L, 1], truth[L, 1]); c_c = S.csi(cref[L, 1], truth[L, 1])
        csi_w.append(S.csi_skill(c_m, c_p, c_c))
        ratios.append(S.variance_ratio(pred[7], truth[7]))
    fp = S.forecast_points(ss_w, csi_w)
    ratios = np.array(ratios)
    blur = S.anti_blur_points(ratios)

    # shocks on monsoon-season windows (Jun-Sep) of the split
    mon = [i for i in wins if times[i].month in (6, 7, 8, 9)]
    pick = mon[:: max(1, len(mon) // n_shock)][:n_shock] if mon else wins[:n_shock]
    caus, null_ok, shock_detail = [], True, []
    for i in pick:
        init = data[i - 2:i + 1]; sc = doy_sincos(int(dy[i]))
        base = fn(init, sc, STEPS, None)
        null = fn(init, sc, STEPS, S.make_shock("null", init[-1], std))
        null_ok &= S.null_consistent(base, null, std)
        res = []
        for name, box in (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA)):
            r = S.causality_checks(base, fn(init, sc, STEPS, S.make_shock(name, init[-1], std)), box, std)
            res.append(r); shock_detail.append({name: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}})
        caus.append(S.causality_points(res, True))
    caus_pts = float(np.mean(caus)) * (1.0 if null_ok else 0.5)

    rm, rp, rc = np.array(rm).mean(0), np.array(rp).mean(0), np.array(rc).mean(0)   # [lead, ch]
    out = {"windows": len(wins), "invalid": invalid, "mean_ss": fp["mean_ss"], "mean_csi_skill": fp["mean_csi_skill"],
           "rmse_pts": fp["rmse_pts"], "csi_pts": fp["csi_pts"], "blur_pts": blur, "causality_pts": caus_pts,
           "null_ok": bool(null_ok), "total_auto": fp["points"] + blur + caus_pts,
           "blur_ratio_ch": np.nanmean(ratios, 0).round(3).tolist(),
           "ss_lead_ch": (1 - rm / np.minimum(rp, rc)).round(3).tolist(),
           "ms_per_call": float(np.median(t_ms)), "shocks": shock_detail}
    if verbose:
        print(f"[{split}] windows={out['windows']} invalid={invalid} mean_ss={out['mean_ss']:.4f} "
              f"csi_skill={out['mean_csi_skill']:.4f} ms/call={out['ms_per_call']:.0f}")
        print(f"  points: rmse {out['rmse_pts']:.1f}/30  csi {out['csi_pts']:.1f}/15  blur {blur:.1f}/10  "
              f"causality {caus_pts:.1f}/15 (null_ok={null_ok})  => {out['total_auto']:.1f}/70")
        print("  ss per lead (rows 1,4,8) x ch (t2m tp u10 v10 msl tcwv):")
        for l, row in zip(LEAD_DAYS, out["ss_lead_ch"]):
            print(f"    d{l}: " + " ".join(f"{v:+.3f}" for v in row))
        print("  day-8 variance ratio per ch:", out["blur_ratio_ch"])
        for d in shock_detail[:4]:
            print("  ", json.dumps(d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="checkpoint (default assets/weights.pt); 'persistence' for the baseline")
    ap.add_argument("--split", default="val"); ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    if a.weights == "persistence":
        fn = persistence_fn
    else:
        import model.rollout as R
        if a.weights:
            R.WEIGHTS = Path(a.weights)
        fn = R.rollout
    evaluate(fn, a.split, a.stride)


if __name__ == "__main__":
    main()
