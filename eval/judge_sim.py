"""Replay the grader's request sequence against a running server, using the kit's own HTTP client.
  python eval/judge_sim.py [http://localhost:8000]

Polls /health every 2 s (120 s budget), then 8 back-to-back calls with no retries: 5 extreme 2021 windows, then
warm_ocean, monsoon_stall and the null shock on the monsoon-peak window. Each call has the 5 s limit and the
validity gate; shocks also get the kit's causality checks. Exit code 0 only if all 8 pass.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from monsoon_r1 import contract as C, scoring as S                    # noqa: E402
from monsoon_r1.client import call_simulate, wait_ready               # noqa: E402
from fit_baseline import load                                         # noqa: E402

# last input day of each window (same picks as dashboard/backend/samples.py)
WINDOWS = {"cyclone (Tauktae)": "2021-05-13", "heatwave (NW India)": "2021-06-05", "monsoon onset surge": "2021-06-12",
           "heavy rain (monsoon peak)": "2021-07-16", "winter disturbance": "2021-01-27"}
SHOCK_WINDOW = "heavy rain (monsoon peak)"
SHOCKS = (("warm_ocean", C.BOX_BAY_OF_BENGAL), ("monsoon_stall", C.BOX_ARABIAN_SEA), ("null", None))


def main(url: str) -> int:
    ok, why = wait_ready(url, 120, allow_private=True, poll_s=2.0)
    print(f"health: {'ready' if ok else 'NOT READY'} {why}")
    if not ok:
        return 1
    data, _, _, t = load()
    std = np.array(json.loads((ROOT / "assets" / "stats.json").read_text())["std"], "f4")
    calls = []                                                         # (name, init, doy, perturbation)
    for name, day in WINDOWS.items():
        w = int(np.where(t == day)[0][0])
        calls.append((name, data[w - 2:w + 1], C.doy_sincos(int(t.day_of_year[w])), None))
    _, init, doy, _ = calls[list(WINDOWS).index(SHOCK_WINDOW)]
    calls += [(f"shock {n}", init, doy, S.make_shock(n, init[-1], std)) for n, _ in SHOCKS]

    outs, rows, all_ok = {}, [], True
    for name, init, doy, pert in calls:                                # back to back, no retries
        payload = {"init_states": C.encode_list(init), "doy_sincos": doy, "steps": C.SCORE_STEPS,
                   "perturbation": None if pert is None else C.encode_list(pert)}
        r = call_simulate(url, payload, allow_private=True)
        bad = S.validity(r.out, C.SCORE_STEPS) if r.ok else [r.error]
        outs[name] = r.out
        rows.append([name, r.wall_s, bad]); all_ok &= r.ok and not bad
    base = outs[SHOCK_WINDOW]
    for (n, box), row in zip(SHOCKS, rows[len(WINDOWS):]):
        shocked = outs[f"shock {n}"]
        if base is None or shocked is None:
            continue
        if box is None:
            good = S.null_consistent(base, shocked, std); row.append(f"null consistent: {good}")
        else:
            c = S.causality_checks(base, shocked, box, std)
            good = all(c[k] for k in ("nonzero", "localized", "stable"))   # propagates is scored, not pass/fail
            row.append(" ".join(f"{k}={c[k]}" for k in ("nonzero", "localized", "propagates", "stable")))
        if not good:
            row[2] = row[2] + ["shock check failed"]; all_ok = False
    for name, wall, bad, *extra in rows:
        status = "FAIL" if bad or wall > C.TIME_LIMIT_S else "OK  "
        print(f"  {status} {name:28s} {wall:5.2f}s / {C.TIME_LIMIT_S:.0f}s  {'; '.join(bad)} {' '.join(extra)}")
    print("RESULT:", "all 8 calls passed" if all_ok else "FAILURES ABOVE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
