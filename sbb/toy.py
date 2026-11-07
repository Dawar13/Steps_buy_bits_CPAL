"""Toy linear system used by the CPU tests (E1_INSTRUCTIONS.md A7, items 4 and 5).

Euler flow sampler with a linear "network": v_i = c(sigma_i) * W q(a(sigma_i)), where
    a(sigma)  smooth activation path (n_tokens x d_in), crossing several quantisation levels,
    c(sigma)  smooth scalar sensitivity, so G_i = c_i W varies between steps,
    sigma     a FLUX-style shifted schedule with uneven step sizes.
The quantiser is the real QuantLinear driven by a StepContext, so the toy also exercises the module.

With the carry (SD) the accumulated error obeys exactly
    sum_i h_i G_i e_i = sum_(i < T-1) h_i (G_(i+1) - G_i) r_i - h_(T-1) G_(T-1) r_(T-1)   (0-based)
where e_i = q_i - a_i and r_i is the stored leftover.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .context import StepContext, gammas_from_sigmas
from .quant import QuantLinear


def shifted_sigmas(num_steps: int, shift: float = 3.0) -> np.ndarray:
    """T+1 noise levels from 1 to 0 with the SD3/FLUX time shift sigma = shift u / (1 + (shift - 1) u)."""
    u = np.linspace(1.0, 0.0, num_steps + 1)
    return shift * u / (1.0 + (shift - 1.0) * u)


@dataclass
class ToySystem:
    d_in: int = 128
    d_out: int = 32
    n_tokens: int = 4
    seed: int = 0
    motion: float = 1.0  # size of the activation path relative to its offset
    dtype: torch.dtype = torch.float64

    def __post_init__(self):
        g = torch.Generator().manual_seed(self.seed)
        kw = dict(generator=g, dtype=self.dtype)
        self.W = torch.randn(self.d_out, self.d_in, **kw) / math.sqrt(self.d_in)
        self.A0 = torch.randn(self.n_tokens, self.d_in, **kw)
        self.A1 = torch.randn(self.n_tokens, self.d_in, **kw) * self.motion
        self.A2 = torch.randn(self.n_tokens, self.d_in, **kw) * (0.5 * self.motion)
        self.freq = 1.0 + 2.0 * torch.rand(self.n_tokens, self.d_in, **kw)
        self.x0 = torch.randn(self.n_tokens, self.d_out, **kw)

    def a(self, sigma: float) -> torch.Tensor:
        return self.A0 + sigma * self.A1 + self.A2 * torch.sin(math.pi * self.freq * sigma)

    @staticmethod
    def c(sigma: float) -> float:
        return 1.0 + 0.5 * sigma + 0.25 * math.sin(2.0 * sigma)

    def linear(self) -> nn.Linear:
        lin = nn.Linear(self.d_in, self.d_out, bias=False).to(self.dtype)
        with torch.no_grad():
            lin.weight.copy_(self.W)
        return lin


def toy_run(system: ToySystem, mode: str, num_steps: int, bits: int = 4, group_size: int = 64,
            storage: str = "fp32", shift: float = 3.0, seed: int = 0, gamma_override: float | None = None) -> dict:
    sig = shifted_sigmas(num_steps, shift)
    h = sig[:-1] - sig[1:]
    ctx = StepContext().configure(mode=mode, bits=bits, storage=storage, group_size=group_size)
    ctx.gamma_override = gamma_override
    lin = system.linear()
    qlin = QuantLinear(lin, ctx, "toy")
    ctx.begin_image(seed=seed, num_steps=num_steps, gammas=gammas_from_sigmas(sig))
    x_q = system.x0.clone()
    x_fp = system.x0.clone()
    errors, leftovers, cs = [], [], []
    with torch.no_grad():
        for i in range(num_steps):
            ctx.set_step(i)
            a_i = system.a(float(sig[i]))
            c_i = system.c(float(sig[i]))
            v_q = c_i * qlin(a_i)
            v_fp = c_i * lin(a_i)
            x_q = x_q - h[i] * v_q
            x_fp = x_fp - h[i] * v_fp
            cs.append(c_i)
            if mode in ("SD", "SD_UNSCALED"):
                leftovers.append(qlin.state.store.get().reshape(a_i.shape[0], -1)[:, : a_i.shape[1]].clone())
    delta = x_q - x_fp
    return {"x_q": x_q, "x_fp": x_fp, "delta": delta, "h": h, "c": np.asarray(cs), "sigmas": sig,
            "leftovers": leftovers, "W": system.W,
            "rel_err": float(delta.norm() / x_fp.norm())}


def telescoped_error(run: dict) -> torch.Tensor:
    """Right-hand side of the summation-by-parts identity, mapped to the sampler state.

    x_q - x_fp = -sum_i h_i c_i W e_i = -[sum_(i<T-1) h_i (c_(i+1) - c_i) W r_i - h_(T-1) c_(T-1) W r_(T-1)].
    """
    h, c, r, W = run["h"], run["c"], run["leftovers"], run["W"]
    t = len(h)
    acc = torch.zeros_like(run["delta"])
    for i in range(t - 1):
        acc = acc + h[i] * (c[i + 1] - c[i]) * (r[i] @ W.T)
    acc = acc - h[t - 1] * c[t - 1] * (r[t - 1] @ W.T)
    return -acc
