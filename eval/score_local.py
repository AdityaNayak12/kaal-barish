"""Local estimate of the Round 1 auto-score (85 of 100; the brief's 15 are manual), run in-process on serve.py.
  python eval/score_local.py [--split test|val] [--windows 40]

Uses the kit's scoring functions. Our guesses where the kit does not say how the grader combines them:
window SS = mean over channels at the kit's LEAD_DAYS, CSI over the same leads, reference climatology =
assets/clim.npy, both shocks + the null shock applied to every window. Repo scan runs on the files git would commit.
"""
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from monsoon_r1 import contract as C, scoring as S                    # noqa: E402
from monsoon_r1.repo_scan import scan                                 # noqa: E402
from fit_baseline import load                                         # noqa: E402

SPLIT_YEAR = {"test": 2021, "val": 2020}                               # CONTRACT.md split


def committed_scan() -> dict:
    """repo_scan on a copy of tracked + unignored files, so local data/ and stray files don't skew it."""
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory() as d:
        tar = subprocess.run(["tar", "--null", "-T", "-", "-cf", "-"], cwd=ROOT, input=files, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-xf", "-"], cwd=d, input=tar, check=True)
        return scan(Path(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLIT_YEAR, default="test"); ap.add_argument("--windows", type=int, default=40)
    a = ap.parse_args()
    import serve
    data, _, idx, t = load()
    clim = np.load(ROOT / "assets" / "clim.npy").astype("f4")
    std = np.array(json.loads((ROOT / "data" / "stats.json").read_text())["std"], "f4")
    wins = year_windows(t, SPLIT_YEAR[a.split], a.windows)
    r = score_windows(serve.simulate, data, idx, t, clim, std, wins); fp = r["fp"]; rs = committed_scan()
    rows = [("Forecast RMSE skill", fp["rmse_pts"], 30, f"mean SS {fp['mean_ss']:.3f} (full at {S.SS_REF})"),
            ("Heavy-rain CSI", fp["csi_pts"], 15, f"mean CSI skill {fp['mean_csi_skill']:.3f} (full at {S.CSI_SS_REF})"),
            ("Anti-blur", r["blur"], 10, "day-8 var ratio " + " ".join(f"{c}={v:.2f}" for c, v in zip(C.CHANNELS, r["ratio"]))),
            ("Perturbation causality", r["caus_pts"], 15, " ".join(f"{k}={v:.0%}" for k, v in r["checks"].items()) + f" null_ok={r['null_ok']:.0%}"),
            ("Repo scan", rs["points"], 15, f"{len(rs['flags'])} flags, inference: {', '.join(rs['inference_files'])}")]
    print(f"model {serve.MODEL_NAME} | split {a.split} | {len(wins)} windows | max wall {max(r['walls']):.2f}s (limit {C.TIME_LIMIT_S}s)\n")
    print(f"{'part':24s} {'points':>12s}  notes")
    for name, p, mx, note in rows:
        print(f"{name:24s} {p:6.2f} / {mx:<3d}  {note}")
    print(f"{'AUTO TOTAL':24s} {sum(row[1] for row in rows):6.2f} / 85   (+ brief 15, manual)")
    for f in rs["flags"]:
        print("  scan flag:", f)
    if r["bad"]:
        print(f"  VALIDITY FAILURES ({len(r['bad'])}):", *r["bad"][:5], sep="\n    ")


def year_windows(t, year: int, n: int) -> np.ndarray:
    """n last-input-day indices spread evenly over one calendar year (same stride for every year)."""
    i = np.arange(len(t))
    cand = np.where((t.year == year) & (i >= 2) & (i + C.SCORE_STEPS < len(t)))[0]
    return cand[np.linspace(0, len(cand) - 1, n).astype(int)]


def score_windows(fc, data, idx, t, clim, std, wins) -> dict:
    """Kit scoring of forecaster fc(init, doy_sincos, steps, perturbation) on the given windows (no repo scan)."""
    L = [d - 1 for d in C.LEAD_DAYS]
    ss, csis, ratios, caus, bad, walls = [], [], [], [], [], []
    for w in wins:
        init, truth = data[w - 2:w + 1], data[w + 1:w + 1 + C.SCORE_STEPS]
        doy = C.doy_sincos(int(t.day_of_year[w]))
        t0 = time.perf_counter(); pred = np.asarray(fc(init, doy, C.SCORE_STEPS, None), "f4"); walls.append(time.perf_counter() - t0)
        bad += [f"{t[w].date()}: {b}" for b in S.validity(pred, C.SCORE_STEPS)]
        persist = np.repeat(init[-1:], C.SCORE_STEPS, axis=0); cref = clim[idx[w + 1:w + 1 + C.SCORE_STEPS]]
        ss.append(float(np.nanmean(S.skill_vs_best_ref(pred[L], truth[L], persist[L], cref[L])["ss"])))
        csis.append(S.csi_skill(*(S.csi(p[L, 1], truth[L, 1]) for p in (pred, persist, cref))))
        ratios.append(S.variance_ratio(pred[7], truth[7]))
        null_ok = S.null_consistent(pred, fc(init, doy, C.SCORE_STEPS, np.zeros(init.shape[1:], "f4")), std)
        res = [S.causality_checks(pred, fc(init, doy, C.SCORE_STEPS, S.make_shock(n, init[-1], std)), box, std)
               for n, box in (("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA))]
        caus.append((S.causality_points(res, null_ok), res, null_ok))
    return {"fp": S.forecast_points(ss, csis), "blur": S.anti_blur_points(np.array(ratios)),
            "ratio": np.nanmean(ratios, axis=0), "caus_pts": float(np.mean([c[0] for c in caus])),
            "checks": {k: float(np.mean([r[i][k] for _, r, _ in caus for i in range(2)]))
                       for k in ("nonzero", "localized", "propagates", "stable", "no_teleport")},
            "null_ok": float(np.mean([c[2] for c in caus])), "bad": bad, "walls": walls, "ss": ss, "csi": csis}


if __name__ == "__main__":
    main()
