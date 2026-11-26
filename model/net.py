"""Residual CNN that steps the 6-channel India state forward by one day.

Shared by model/rollout.py (inference) and train/ (training). Pure torch, no I/O.

State space: every channel is transformed (rain -> log1p) and standardised with per-channel mean/std
stored in the checkpoint, so the network sees roughly unit-scale inputs. The network predicts the
increment z[t+1] - z[t]; the receptive field is kept small (radius ~ n_blocks*2 + 1 cells per day) so
a local shock stays local on day 1 and spreads at a physical pace over the rollout.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

N_CH = 6          # t2m, tp, u10, v10, msl, tcwv
T_IN = 3          # input days
RAIN = 1          # index of the precipitation channel


def to_z(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """Physical [..., 6, H, W] -> standardised transformed space (rain via log1p)."""
    x = torch.cat([x[..., :RAIN, :, :], torch.log1p(x[..., RAIN:RAIN + 1, :, :].clamp_min(0.0)),
                   x[..., RAIN + 1:, :, :]], dim=-3)
    return (x - mu[:, None, None]) / sd[:, None, None]


def from_z(z: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """Inverse of to_z, returning physical units with rain >= 0."""
    x = z * sd[:, None, None] + mu[:, None, None]
    return torch.cat([x[..., :RAIN, :, :], torch.expm1(x[..., RAIN:RAIN + 1, :, :].clamp(0.0, 8.0)),
                      x[..., RAIN + 1:, :, :]], dim=-3)


def doy_features(doy: torch.Tensor) -> torch.Tensor:
    """Day of year (float, any shape) -> [..., 2] sin/cos on a 365-day cycle (matches the API's doy_sincos)."""
    a = 2 * math.pi * doy / 365.0
    return torch.stack([torch.sin(a), torch.cos(a)], dim=-1)


def doy_from_sincos(s: float, c: float) -> float:
    """Inverse of the API's doy_sincos: day of year in [0, 365)."""
    return (math.atan2(s, c) * 365.0 / (2 * math.pi)) % 365.0


def clim_rows(doy_last: float, steps: int) -> list[int]:
    """Rows of the 366-day climatology for forecast days 1..steps after the last input day (1-based doy)."""
    return [(int(round(doy_last + k)) - 1) % 366 for k in range(1, steps + 1)]


def static_features(static: torch.Tensor) -> torch.Tensor:
    """static [2,H,W] (land_sea_mask, orography_m) -> [1,4,H,W]: mask, scaled orography, lat and lon coordinates."""
    _, H, W = static.shape
    lat = (torch.arange(H, dtype=torch.float32) + 6.5 - 22.0) / 10.0
    lon = (torch.arange(W, dtype=torch.float32) + 65.5 - 83.0) / 10.0
    return torch.stack([static[0], static[1] / 2000.0,
                        lat[:, None].expand(H, W), lon[None, :].expand(H, W)])[None].float()


class ResBlock(nn.Module):
    def __init__(self, w: int):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1, padding_mode="replicate")
        self.c2 = nn.Conv2d(w, w, 3, padding=1, padding_mode="replicate")
        self.n1 = nn.GroupNorm(8, w)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.c2(self.act(self.c1(self.act(self.n1(x)))))


class StepNet(nn.Module):
    """One-day step. Inputs (all on the 32x36 grid):
      z_hist  [B, 3, 6, H, W]  last three days, oldest first, standardised
      z_clim  [B, 6, H, W]     climatology of the target day, standardised
      static  [B, n_static, H, W]  land-sea mask, orography, lat/lon coordinates (fixed per grid)
      doy     [B, 2]           sin/cos of the target day
    Output: z[t+1] = z_hist[:, -1] + increment.
    """

    def __init__(self, width: int = 64, n_blocks: int = 3, n_static: int = 4):
        super().__init__()
        cin = T_IN * N_CH + N_CH + n_static + 2
        self.inp = nn.Conv2d(cin, width, 3, padding=1, padding_mode="replicate")
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.GELU(), nn.Conv2d(width, N_CH, 1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)   # start as persistence

    def forward(self, z_hist, z_clim, static, doy):
        B, _, _, H, W = z_hist.shape
        last = z_hist[:, -1]
        feats = torch.cat([
            z_hist.reshape(B, T_IN * N_CH, H, W),
            z_clim - last,                                   # pull towards climatology is easy to learn
            static.expand(B, -1, -1, -1),
            doy[:, :, None, None].expand(B, 2, H, W),
        ], dim=1)
        return last + self.out(self.blocks(self.inp(feats)))

    @torch.no_grad()
    def rollout_z(self, z_hist, z_clim_seq, static, doy_seq):
        """z_hist [B,3,6,H,W]; z_clim_seq [B,S,6,H,W]; doy_seq [B,S,2] -> [B,S,6,H,W] (no grad)."""
        return self.rollout_z_grad(z_hist, z_clim_seq, static, doy_seq)

    def rollout_z_grad(self, z_hist, z_clim_seq, static, doy_seq):
        outs = []
        for k in range(z_clim_seq.shape[1]):
            nxt = self(z_hist, z_clim_seq[:, k], static, doy_seq[:, k])
            outs.append(nxt)
            z_hist = torch.cat([z_hist[:, 1:], nxt[:, None]], dim=1)
        return torch.stack(outs, dim=1)
