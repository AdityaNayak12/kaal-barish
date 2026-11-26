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
import serve                                                          # noqa: E402

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
    data, _, idx, t = load()
    clim = np.load(ROOT / "assets" / "clim.npy").astype("f4")
    std = np.array(json.loads((ROOT / "data" / "stats.json").read_text())["std"], "f4")
    cand = np.where((t.year == SPLIT_YEAR[a.split]) & (np.arange(len(t)) >= 2) & (np.arange(len(t)) + C.SCORE_STEPS < len(t)))[0]
    wins = cand[np.linspace(0, len(cand) - 1, a.windows).astype(int)]
    L = [d - 1 for d in C.LEAD_DAYS]
    ss, csis, ratios, caus, bad, walls = [], [], [], [], [], []
    for w in wins:
        init, truth = data[w - 2:w + 1], data[w + 1:w + 1 + C.SCORE_STEPS]
        doy = C.doy_sincos(int(t.day_of_year[w]))
        t0 = time.perf_counter(); pred = serve.simulate(init, doy, C.SCORE_STEPS, None); walls.append(time.perf_counter() - t0)
        bad += [f"{t[w].date()}: {b}" for b in S.validity(pred, C.SCORE_STEPS)]
        persist = np.repeat(init[-1:], C.SCORE_STEPS, axis=0); cref = clim[idx[w + 1:w + 1 + C.SCORE_STEPS]]
        ss.append(float(np.nanmean(S.skill_vs_best_ref(pred[L], truth[L], persist[L], cref[L])["ss"])))
        csis.append(S.csi_skill(*(S.csi(p[L, 1], truth[L, 1]) for p in (pred, persist, cref))))
        ratios.append(S.variance_ratio(pred[7], truth[7]))
        null_ok = S.null_consistent(pred, serve.simulate(init, doy, C.SCORE_STEPS, np.zeros(init.shape[1:], "f4")), std)
        res = [S.causality_checks(pred, serve.simulate(init, doy, C.SCORE_STEPS, S.make_shock(n, init[-1], std)), box, std)
               for n, box in (("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA))]
        caus.append((S.causality_points(res, null_ok), res, null_ok))
    fp = S.forecast_points(ss, csis); blur = S.anti_blur_points(np.array(ratios)); rs = committed_scan()
    cp = float(np.mean([c[0] for c in caus]))
    checks = {k: np.mean([r[i][k] for _, r, _ in caus for i in range(2)]) for k in ("nonzero", "localized", "propagates", "stable")}
    rows = [("Forecast RMSE skill", fp["rmse_pts"], 30, f"mean SS {fp['mean_ss']:.3f} (full at {S.SS_REF})"),
            ("Heavy-rain CSI", fp["csi_pts"], 15, f"mean CSI skill {fp['mean_csi_skill']:.3f} (full at {S.CSI_SS_REF})"),
            ("Anti-blur", blur, 10, "day-8 var ratio " + " ".join(f"{c}={r:.2f}" for c, r in zip(C.CHANNELS, np.nanmean(ratios, axis=0)))),
            ("Perturbation causality", cp, 15, " ".join(f"{k}={v:.0%}" for k, v in checks.items()) + f" null_ok={np.mean([c[2] for c in caus]):.0%}"),
            ("Repo scan", rs["points"], 15, f"{len(rs['flags'])} flags, inference: {', '.join(rs['inference_files'])}")]
    print(f"model {serve.MODEL_NAME} | split {a.split} | {len(wins)} windows | max wall {max(walls):.2f}s (limit {C.TIME_LIMIT_S}s)\n")
    print(f"{'part':24s} {'points':>12s}  notes")
    for name, p, mx, note in rows:
        print(f"{name:24s} {p:6.2f} / {mx:<3d}  {note}")
    print(f"{'AUTO TOTAL':24s} {sum(r[1] for r in rows):6.2f} / 85   (+ brief 15, manual)")
    for f in rs["flags"]:
        print("  scan flag:", f)
    if bad:
        print(f"  VALIDITY FAILURES ({len(bad)}):", *bad[:5], sep="\n    ")


if __name__ == "__main__":
    main()
