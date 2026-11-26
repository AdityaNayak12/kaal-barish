"""Drift check: does the shipped model hold up on 2022, the ERA5 year closest to the hidden (after 2023-01-10) windows?
  python -m monsoon_r1.build_dataset train --start 2021-12-20 --end 2023-01-09 --out data/r1/y2022.npz   (once)
  python eval/drift_check.py [--windows 40]

Same recipe as eval/score_local.py (serve.simulate, kit scoring, 40 evenly spaced windows) on val 2020, test 2021
and 2022, plus per channel: skill SS vs the better reference, the truth's mean anomaly against assets/clim.npy
(2005-2019, so climate drift shows up), and the forecast bias at lead days 1/4/8.
2022 comes from the kit's own builder (WeatherBench2), the same pipeline as india_climate_train.nc. Never fetch dates
after 2023-01-10: the hidden test windows are there.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from monsoon_r1 import contract as C, scoring as S                    # noqa: E402
from fit_baseline import load                                         # noqa: E402
from score_local import score_windows, year_windows                   # noqa: E402
import serve                                                          # noqa: E402


def channel_stats(data, idx, t, clim, wins):
    """Per channel: SS vs best reference, truth anomaly vs clim, forecast bias at leads 1/4/8 (physical units)."""
    L = [d - 1 for d in C.LEAD_DAYS]; ss, anom, bias = [], [], []
    for w in wins:
        init, truth = data[w - 2:w + 1], data[w + 1:w + 1 + C.SCORE_STEPS]
        pred = serve.simulate(init, C.doy_sincos(int(t.day_of_year[w])), C.SCORE_STEPS, None)
        persist = np.repeat(init[-1:], C.SCORE_STEPS, axis=0); cref = clim[idx[w + 1:w + 1 + C.SCORE_STEPS]]
        ss.append(S.skill_vs_best_ref(pred[L], truth[L], persist[L], cref[L])["ss"])
        anom.append((truth[L] - cref[L]).mean(axis=(0, 2, 3))); bias.append((pred[L] - truth[L]).mean(axis=(2, 3)))
    return np.nanmean(ss, axis=0), np.mean(anom, axis=0), np.mean(bias, axis=0)    # [6], [6], [3 leads, 6]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--npz", default=str(ROOT / "data" / "r1" / "y2022.npz")); a = ap.parse_args()
    clim = np.load(ROOT / "assets" / "clim.npy").astype("f4")
    std = np.array(json.loads((ROOT / "assets" / "stats.json").read_text())["std"], "f4")
    nc = load(); extra = load(a.npz)
    years = (("val 2020", nc, 2020), ("test 2021", nc, 2021), ("2022", extra, 2022))
    print(f"model {serve.MODEL_NAME} | {a.windows} windows per year\n")
    print(f"{'year':10s} {'mean SS':>7s} {'RMSE/30':>7s} {'CSI sk':>6s} {'CSI/15':>6s} {'blur/10':>7s} {'caus/15':>7s} {'auto/70':>7s}")
    per = {}
    for name, (data, _, idx, t), year in years:
        wins = year_windows(t, year, a.windows)
        r = score_windows(serve.simulate, data, idx, t, clim, std, wins); fp = r["fp"]
        per[name] = channel_stats(data, idx, t, clim, wins)
        draws = [S.forecast_points(list(np.take(r["ss"], d)), list(np.take(r["csi"], d)))
                 for d in np.random.default_rng(0).integers(0, len(wins), (2000, 5))]   # the judge scores only 5 windows
        e5 = {k: np.mean([x[k] for x in draws]) for k in ("rmse_pts", "csi_pts")}
        p_full = np.mean([x["rmse_pts"] >= 30 for x in draws])
        print(f"{name:10s} {fp['mean_ss']:7.3f} {fp['rmse_pts']:7.2f} {fp['mean_csi_skill']:6.3f} {fp['csi_pts']:6.2f} "
              f"{r['blur']:7.2f} {r['caus_pts']:7.2f} {fp['points'] + r['blur'] + r['caus_pts']:7.2f}   "
              f"5-window draws: E[RMSE] {e5['rmse_pts']:.2f}, P(RMSE=30) {p_full:.0%}, E[CSI] {e5['csi_pts']:.2f}"
              + (f"   VALIDITY FAILURES {len(r['bad'])}" if r["bad"] else ""))
    print("\nper channel (" + " ".join(C.CHANNELS) + ")")
    for name, (ss, anom, bias) in per.items():
        print(f"{name:10s} SS        " + " ".join(f"{v:7.3f}" for v in ss))
        print(f"{'':10s} truth-clim" + " ".join(f"{v:7.2f}" for v in anom) + "   (mean anomaly vs 2005-2019 clim, physical units)")
        for k, d in enumerate(C.LEAD_DAYS):
            print(f"{'':10s} bias d{d:<2d}  " + " ".join(f"{v:7.2f}" for v in bias[k]))


if __name__ == "__main__":
    main()
