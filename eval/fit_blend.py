"""Fit engine/blend.json and check it.   python eval/fit_blend.py [--check] [--stride 2]

Fit: for each lead 1..10 and channel t2m, u10, msl, tcwv, the weights a, b of
out = clim + a * (cnn - clim) + b * (baseline - clim), by latitude-weighted least squares over every window of the
validation year 2020 only (CONTRACT.md split), clipped to [0, 2]. Rain is never blended.
--check: blended serve.simulate vs the plain CNN on 2020, 2021 and 2022 (data/r1/y2022.npz from the kit's builder),
per season with the kit's scoring: forecast skill, heavy-rain CSI, day-8 variance ratio per channel (must stay
>= 0.4), causality (must be identical), the shock deltas themselves, the 5-window expectation and latency.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from monsoon_r1 import contract as C, scoring as S                    # noqa: E402
from fit_baseline import load                                         # noqa: E402
from score_local import score_windows                                 # noqa: E402
from engine.blend import clim_rows                                    # noqa: E402
import serve                                                          # noqa: E402

CH = [0, 2, 4, 5]          # not rain (heavy-rain scores stay the CNN's); not v10: it took v10's Oct-Nov day-8 variance ratio to 0.40
LEADS = C.SCORE_STEPS
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJAS": (6, 7, 8, 9), "ON": (10, 11)}


def windows(t, year, stride=1):
    i = np.arange(len(t))
    return np.where((t.year == year) & (i >= 2) & (i + LEADS < len(t)))[0][::stride]


def plain(init, doy, steps, pert):
    """The CNN as served before the blend."""
    return np.asarray(serve.rollout(init, doy, steps, pert), "f4")


def fit():
    assert serve.MODEL_NAME == "kaal-barish-cnn", serve.MODEL_NAME
    data, _, _, t = load()
    w = C.lat_weights()[:, None]
    A = np.zeros((LEADS, len(CH), 2, 2)); r = np.zeros((LEADS, len(CH), 2))
    for i in windows(t, 2020):
        init = data[i - 2:i + 1]; doy = C.doy_sincos(int(t.day_of_year[i]))
        cl = clim_rows(doy, LEADS)[:, CH].astype("f8")
        x = np.stack([plain(init, doy, LEADS, None)[:, CH], serve.baseline(init, doy, LEADS, None)[:, CH]], 2) - cl[:, :, None]
        y = data[i + 1:i + 1 + LEADS][:, CH] - cl                      # x [lead, ch, 2, 32, 36], y [lead, ch, 32, 36]
        A += np.einsum("kcpij,kcqij->kcpq", x * w, x); r += np.einsum("kcpij,kcij->kcp", x * w, y)
    ab = np.clip(np.linalg.solve(A, r[..., None])[..., 0], 0.0, 2.0)  # [lead, ch, (a, b)]
    out = ROOT / "engine" / "blend.json"
    out.write_text(json.dumps({"channels": CH, "a": ab[..., 0].round(4).tolist(), "b": ab[..., 1].round(4).tolist()}, indent=1) + "\n")
    print("lead  " + "  ".join(f"{C.CHANNELS[c]:>11s}" for c in CH) + "   (a cnn, b baseline)")
    for k in range(LEADS):
        print(f"d{k + 1:<4d}" + "  ".join(f"{a:5.2f},{b:5.2f}" for a, b in ab[k]))
    print("wrote", out)


def draws5(ss, cs, n=4000):
    D = np.random.default_rng(0).integers(0, len(ss), (n, 5)); fp = [S.forecast_points(list(ss[d]), list(cs[d])) for d in D]
    rm = np.array([f["rmse_pts"] for f in fp])
    return rm.mean(), (rm >= 30).mean(), np.mean([f["csi_pts"] for f in fp])


def check(stride):
    clim = np.load(ROOT / "assets" / "clim.npy").astype("f4")
    std = np.array(json.loads((ROOT / "assets" / "stats.json").read_text())["std"], "f4")
    nc = load(); years = ((nc, 2020), (nc, 2021), (load(ROOT / "data" / "r1" / "y2022.npz"), 2022))
    low, caus_diff, dmax, rain_same = [], 0, 0.0, True
    print(f"{'':11s} {'n':>3s} | plain: {'SS':>5s} {'CSIsk':>6s} {'blur':>5s} {'caus':>5s} | blend: {'SS':>5s} {'CSIsk':>6s} {'blur':>5s} {'caus':>5s}"
          f" | blend day-8 variance ratio " + " ".join(C.CHANNELS))
    for (data, _, idx, t), year in years:
        allw = windows(t, year, stride); tot = {"plain": ([], []), "blend": ([], [])}
        for sn, months in SEASONS.items():
            wins = allw[np.isin(t.month[allw], months)]
            rp = score_windows(plain, data, idx, t, clim, std, wins); rb = score_windows(serve.simulate, data, idx, t, clim, std, wins)
            for name, r in (("plain", rp), ("blend", rb)):
                tot[name][0].extend(r["ss"]); tot[name][1].extend(r["csi"])
            caus_diff += int(rp["caus_pts"] != rb["caus_pts"] or rp["checks"] != rb["checks"] or rp["null_ok"] != rb["null_ok"])
            low += [f"{year} {sn} {c}={v:.3f} (plain {vp:.3f})" for c, v, vp in zip(C.CHANNELS, rb["ratio"], rp["ratio"]) if not v >= S.BLUR_MIN_RATIO + 0.01]
            print(f"{year} {sn:5s} {len(wins):3d} | " + " | ".join(
                f"{'':6s}{r['fp']['mean_ss']:5.3f} {r['fp']['mean_csi_skill']:6.3f} {r['blur']:5.2f} {r['caus_pts']:5.2f}" for r in (rp, rb))
                + " | " + " ".join(f"{v:.2f}" for v in rb["ratio"]))
        e = {k: draws5(np.array(v[0]), np.array(v[1])) for k, v in tot.items()}
        print(f"{year} all-season 5-window expectation: plain RMSE {e['plain'][0]:.2f} (P30 {e['plain'][1]:.0%}) CSI {e['plain'][2]:.2f}"
              f"  ->  blend RMSE {e['blend'][0]:.2f} (P30 {e['blend'][1]:.0%}) CSI {e['blend'][2]:.2f}")
        for i in allw[::10]:                                           # the shock deltas themselves
            init = data[i - 2:i + 1]; doy = C.doy_sincos(int(t.day_of_year[i]))
            p0, b0 = plain(init, doy, LEADS, None), serve.simulate(init, doy, LEADS, None)
            for n in ("warm_ocean", "monsoon_stall"):
                pert = S.make_shock(n, init[-1], std)
                dp, db = plain(init, doy, LEADS, pert) - p0, serve.simulate(init, doy, LEADS, pert) - b0
                dmax = max(dmax, float((np.abs(db - dp) / std[None, :, None, None]).max())); rain_same &= bool(np.array_equal(db[:, 1], dp[:, 1]))
    print(f"\nday-8 variance ratio below {S.BLUR_MIN_RATIO + 0.01:.2f} (any season, blended): {low or 'none'}")
    print(f"causality: seasons where points/checks/null differ from the plain CNN: {caus_diff}")
    print(f"shock deltas: max |blend - plain| = {dmax:.2e} std; rain deltas bit-identical: {rain_same}")
    data, _, _, t = nc; i = int(windows(t, 2021)[196]); init = data[i - 2:i + 1]; doy = C.doy_sincos(int(t.day_of_year[i]))
    pert = S.make_shock("monsoon_stall", init[-1], std); zero = np.zeros_like(pert)

    def ms(f, n=20, cold=False):
        ts = []
        for _ in range(n):
            if cold:
                serve._RECENT.clear()
            t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        return 1000 * np.median(ts), 1000 * max(ts)
    rows = {"plain base": ms(lambda: plain(init, doy, LEADS, None)), "plain shocked": ms(lambda: plain(init, doy, LEADS, pert)),
            "blend base (new init)": ms(lambda: serve.simulate(init, doy, LEADS, None), cold=True),
            "blend shocked (base seen)": ms(lambda: serve.simulate(init, doy, LEADS, pert)),
            "blend shocked (base unseen)": ms(lambda: serve.simulate(init, doy, LEADS, pert), cold=True)}
    print("latency ms (median, max of 20): " + "; ".join(f"{k} {a:.0f}/{b:.0f}" for k, (a, b) in rows.items()))
    print("null shock bit-identical to no shock:", bool(np.array_equal(serve.simulate(init, doy, LEADS, zero), serve.simulate(init, doy, LEADS, None))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    check(a.stride) if a.check else fit()
