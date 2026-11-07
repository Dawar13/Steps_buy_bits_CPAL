"""Metrics for E1: relative latent error, PSNR, LPIPS and the error-versus-steps slope.

Two forms of the latent error are recorded:
    rel_err_sq = ||z_q - z_FP||^2 / ||z_FP||^2   (the primary metric named in E1_INSTRUCTIONS.md A5)
    rel_err    = sqrt(rel_err_sq) = ||z_q - z_FP|| / ||z_FP||
Slopes are fitted on rel_err, because the predicted slopes (0, -1/2, -1 in PROJECT_CONTEXT.md 4.2)
are for the size of the error, not its square. A fit on rel_err_sq gives exactly twice the slope.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch


def rel_err_sq(z_q: torch.Tensor, z_fp: torch.Tensor) -> float:
    zq, zf = z_q.detach().double().cpu(), z_fp.detach().double().cpu()
    if zq.shape != zf.shape:
        raise ValueError(f"shape mismatch {tuple(zq.shape)} vs {tuple(zf.shape)}")
    den = float((zf ** 2).sum())
    if den == 0.0:
        raise ValueError("reference latent is zero")
    return float(((zq - zf) ** 2).sum()) / den


def psnr(img: torch.Tensor, ref: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR in dB for images in [0, data_range]. Identical images give +inf."""
    mse = float(((img.detach().double().cpu() - ref.detach().double().cpu()) ** 2).mean())
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(data_range ** 2 / mse)


def _load_lpips_model(net: str = "alex"):
    import lpips  # imported lazily: only installed on the VM

    return lpips.LPIPS(net=net, verbose=False)


class LPIPSMetric:
    """LPIPS (AlexNet) between two images in [0, 1] with shape (B, 3, H, W)."""

    def __init__(self, device: str | torch.device = "cpu", net: str = "alex"):
        self.device = torch.device(device)
        self.model = _load_lpips_model(net).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, img: torch.Tensor, ref: torch.Tensor) -> float:
        a = img.to(self.device, torch.float32) * 2 - 1
        b = ref.to(self.device, torch.float32) * 2 - 1
        return float(self.model(a, b).mean())


def fit_slope(ts: Sequence[float], errs: Sequence[float], t_min: float = 16) -> tuple[float, float]:
    """Least squares of log(err) on log(T) using T >= t_min. Returns (slope, intercept); NaN if < 2 points."""
    pts = [(float(t), float(e)) for t, e in zip(ts, errs) if t >= t_min and e > 0 and math.isfinite(e)]
    if len({t for t, _ in pts}) < 2:
        return float("nan"), float("nan")
    x = np.log([t for t, _ in pts])
    y = np.log([e for _, e in pts])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def bootstrap_median_ci(values: Iterable[float], n_boot: int = 2000, seed: int = 0,
                        level: float = 0.95) -> tuple[float, float, float]:
    """Median and percentile bootstrap interval (resampling the values, e.g. prompts, with replacement)."""
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(v))
    if v.size == 1:
        return med, med, med
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return med, float(lo), float(hi)


def slope_table(df, t_min: float = 16, metric: str = "rel_err", n_boot: int = 2000, seed: int = 0,
                group_cols: Sequence[str] = ("mode", "bits")):
    """Per-prompt slopes, then the median and bootstrap 95% interval over prompts for each group.

    Returns (per_prompt DataFrame, summary DataFrame).
    """
    import pandas as pd

    rows = []
    for keys, g in df.groupby(list(group_cols) + ["prompt_id"]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        g = g.sort_values("T")
        slope, intercept = fit_slope(g["T"].tolist(), g[metric].tolist(), t_min)
        rows.append(dict(zip(list(group_cols) + ["prompt_id"], keys), slope=slope, intercept=intercept))
    per_prompt = pd.DataFrame(rows)
    summ = []
    if not per_prompt.empty:
        for keys, g in per_prompt.groupby(list(group_cols)):
            keys = keys if isinstance(keys, tuple) else (keys,)
            med, lo, hi = bootstrap_median_ci(g["slope"].tolist(), n_boot=n_boot, seed=seed)
            summ.append(dict(zip(group_cols, keys), slope_median=med, ci_low=lo, ci_high=hi,
                             n_prompts=int(np.isfinite(g["slope"]).sum())))
    return per_prompt, pd.DataFrame(summ)
