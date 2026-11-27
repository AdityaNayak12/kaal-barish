"""Self-check for serve.py's request handling, in-process:  python eval/test_serve.py"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import serve                                                          # noqa: E402

x = np.load(Path(serve.__file__).parent / "assets" / "clim.npy")[[150, 151, 152]]
doy = (0.5, -0.86)                                                     # tuple on purpose
base = serve.simulate(x, doy, 10, None)
assert serve._valid(base, 10), "base rollout invalid"
assert np.array_equal(base, serve.simulate(x, list(doy), 10, np.zeros(serve.SHAPE, "f4"))), "zeros shock != no shock"
assert np.array_equal(base, serve.simulate(x, doy, 10, None)), "not deterministic"
assert serve._valid(serve.simulate(x, doy, serve.MAX_STEPS, None), serve.MAX_STEPS), "long rollout invalid"
holes = x.copy(); holes[-1, 0, :5] = np.nan; holes[0, 4] = np.inf
assert serve._valid(serve.simulate(holes, doy, 10, None), 10), "NaN/Inf input not repaired"

if serve.MODEL_NAME == "kaal-barish-cnn":                              # blend: a shock moves the output exactly as it moves the CNN
    std = np.array(json.loads((Path(serve.__file__).parent / "assets" / "stats.json").read_text())["std"], "f4")
    pert = np.zeros(serve.SHAPE, "f4"); pert[0, 10:20, 20:30] = 3.0; pert[2, 10:20, 5:12] = -2.0
    shocked = serve.simulate(x, doy, 10, pert)
    d_cnn = serve.rollout(x, doy, 10, pert) - serve.rollout(x, doy, 10, None)
    assert np.array_equal((shocked - base)[:, 1], d_cnn[:, 1]), "rain shock response changed"
    assert (np.abs(shocked - base - d_cnn) / std[:, None, None]).max() < 1e-4, "shock response changed"
    serve._RECENT.clear()
    assert np.array_equal(shocked, serve.simulate(x, doy, 10, pert)), "shock result depends on call order"
    hot = np.zeros(serve.SHAPE, "f4"); hot[0, 10:20, 20:30] = 60.0     # extreme dashboard input must not fall back
    out = serve.simulate(x, doy, 10, hot)
    d_hot = serve.rollout(x, doy, 10, hot) - serve.rollout(x, doy, 10, None)
    assert serve._valid(out, 10) and np.array_equal((out - base)[:, 1], d_hot[:, 1]), "extreme shock fell back to the baseline"

serve._RECENT.clear()                                                  # force a request-time CNN failure
cnn = serve.rollout
serve.rollout = lambda *a: np.full((10,) + serve.SHAPE, np.nan, "f4")
assert np.array_equal(serve.simulate(x, doy, 10, None), serve.baseline(x, doy, 10, None)), "no baseline fallback"
serve.rollout = cnn
print(f"serve.py self-check OK ({serve.MODEL_NAME})")
