"""Self-check for serve.py's request handling, in-process:  python eval/test_serve.py"""
import sys
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

cnn = serve.rollout                                                    # force a request-time CNN failure
serve.rollout = lambda *a: np.full((10,) + serve.SHAPE, np.nan, "f4")
assert np.array_equal(serve.simulate(x, doy, 10, None), serve.baseline(x, doy, 10, None)), "no baseline fallback"
serve.rollout = cnn
print(f"serve.py self-check OK ({serve.MODEL_NAME})")
