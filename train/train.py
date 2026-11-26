"""Train the residual CNN (model/net.py) with a curriculum of rollout lengths.

  python -m train.train --out data/runs/r1.pt [--stages 1:15,4:8,8:5] [--split train]

Each stage "S:E" trains E epochs on S-step autoregressive rollouts (loss averaged over the S steps).
Loss: latitude-weighted MSE in the standardised state space + a physical-rain term (tp / kit std) so heavy rain
is not under-weighted by the log1p transform. After every stage the checkpoint is saved and scored on the val
split with train/evaluate.py.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch

from model.net import RAIN, StepNet, from_z, static_features, to_z
from train.data import ASSETS, load_archive, year_mask


def lat_w(H=32):
    lat = torch.arange(H, dtype=torch.float32) + 6.5
    w = torch.cos(torch.deg2rad(lat)); return (w / w.mean())[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/runs/r1.pt")
    ap.add_argument("--split", default="train")
    ap.add_argument("--stages", default="1:15,4:8,8:5")
    ap.add_argument("--width", type=int, default=64); ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rain-w", type=float, default=0.5); ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-batches", type=int, default=0, help="cap batches per epoch (0 = all; for smoke tests)")
    ap.add_argument("--init", default=None, help="start from this checkpoint")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(a.threads)

    arch = load_archive()
    data, times, static, _, _ = arch
    m = year_mask(times, a.split)
    x = torch.from_numpy(data)
    # transformed-space stats from the training years only
    xt = x[torch.from_numpy(m)].clone(); xt[:, RAIN] = torch.log1p(xt[:, RAIN].clamp_min(0))
    mu, sd = xt.mean(dim=(0, 2, 3)), xt.std(dim=(0, 2, 3))
    z_all = to_z(x, mu, sd)                                              # [N,6,H,W]
    clim = torch.from_numpy(np.load(ASSETS / "clim.npy"))
    z_clim = to_z(clim, mu, sd)                                          # [366,6,H,W]
    stat = static_features(torch.from_numpy(static))
    kit_sd_tp = float(json.loads((ASSETS / "stats.json").read_text())["std"][RAIN])
    d0 = torch.from_numpy(np.asarray(times.day_of_year, "f8") % 365.0)     # what doy_from_sincos recovers
    W = lat_w()

    cfg = {"width": a.width, "n_blocks": a.blocks, "n_static": stat.shape[1]}
    net = StepNet(**cfg)
    if a.init:
        net.load_state_dict(torch.load(a.init, weights_only=True)["state_dict"])
    print(f"params {sum(p.numel() for p in net.parameters())/1e3:.0f}k, train days {m.sum()}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def batch(idx, S):
        idx = torch.as_tensor(idx)
        hist = torch.stack([z_all[idx - 2], z_all[idx - 1], z_all[idx]], 1)
        tgt = torch.stack([z_all[idx + k] for k in range(1, S + 1)], 1)
        dk = d0[idx][:, None] + torch.arange(1, S + 1, dtype=torch.float64)[None]
        rows = (torch.round(dk).long() - 1) % 366
        ang = 2 * math.pi * dk / 365.0
        doy = torch.stack([torch.sin(ang), torch.cos(ang)], -1).float()
        return hist, z_clim[rows], doy, tgt

    def save(path):
        torch.save({"config": cfg, "state_dict": net.state_dict(), "mu": mu, "sd": sd}, path)

    for stage in a.stages.split(","):
        S, E = (int(v) for v in stage.split(":"))
        valid = np.array([i for i in range(2, len(times) - S) if m[i - 2:i + S + 1].all()])
        nb = len(valid) // a.batch if not a.max_batches else min(a.max_batches, len(valid) // a.batch)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr / (1 if S == 1 else 3), weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=E * nb, pct_start=0.1)
        for ep in range(E):
            net.train(); perm = np.random.permutation(valid); tot = 0.0; t0 = time.time()
            for b in range(nb):
                hist, zc, doy, tgt = batch(perm[b * a.batch:(b + 1) * a.batch], S)
                pred = net.rollout_z_grad(hist, zc, stat, doy)
                loss = (((pred - tgt) ** 2) * W).mean()
                if a.rain_w > 0:
                    tp_p = from_z(pred, mu, sd)[:, :, RAIN]; tp_t = from_z(tgt, mu, sd)[:, :, RAIN]
                    loss = loss + a.rain_w * ((((tp_p - tp_t) / kit_sd_tp) ** 2) * W).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
                tot += loss.item()
            print(f"stage S={S} ep {ep + 1}/{E} loss {tot / nb:.4f} ({time.time() - t0:.0f}s)", flush=True)
        save(a.out)
        if not a.no_eval:
            import model.rollout as R
            from train.evaluate import evaluate
            R.WEIGHTS = Path(a.out); R._state = None
            evaluate(R.rollout, "val", 5, arch=arch)
    print("saved", a.out)


if __name__ == "__main__":
    main()
