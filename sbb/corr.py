"""E1.1 recorder: temporal correlation of plain-rounding leftovers between sampling steps.

For each quantised input (one per QuantState key) a fixed random subset of entries is chosen once
(seeded by the key and a global seed). At every step the plain RTN leftover r_i = a_i - RTN(a_i)
is computed on that subset (no carry, whatever the run mode), and stored in a ring buffer of the
last max(lags) + 1 steps. The recorder logs
    cos(r_i, r_(i-lag)) for each lag, and
    drift ratio ||a_i - a_(i-1)|| / ||s_i|| on the subset (s_i: the per-entry quantiser scale).
"""
from __future__ import annotations

import math
import re
import warnings
import zlib
from dataclasses import dataclass, field

import numpy as np
import torch

from .quant import group_scale, qmax_for


def subset_indices(numel: int, n: int, seed: int) -> np.ndarray:
    n = min(int(n), int(numel))
    rng = np.random.default_rng(seed)
    idx = rng.choice(numel, size=n, replace=False, shuffle=False) if n < numel else np.arange(numel)
    return np.sort(idx).astype(np.int64)


def key_seed(key: str, seed: int) -> int:
    return (zlib.crc32(key.encode("utf-8")) ^ (seed * 2654435761)) & 0xFFFFFFFF


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float((a * b).sum()) / (na * nb)


@dataclass
class _Track:
    idx: torch.Tensor  # positions in the grouped, padded, flattened layout
    shape: tuple
    ring: torch.Tensor | None = None
    ring_steps: list = field(default_factory=list)
    prev_a: torch.Tensor | None = None
    prev_step: int | None = None

    def clear(self, depth: int) -> None:
        self.ring = None
        self.ring_steps = [None] * depth
        self.prev_a = None
        self.prev_step = None


class CorrRecorder:
    def __init__(self, n_entries: int = 65536, lags=(1, 2, 5, 10), seed: int = 0):
        self.n_entries = int(n_entries)
        self.lags = tuple(int(l) for l in lags)
        if min(self.lags) < 1:
            raise ValueError("lags must be positive")
        self.depth = max(self.lags) + 1
        self.seed = int(seed)
        self.tracks: dict[str, _Track] = {}
        self.order: list[str] = []
        self.num_steps: int | None = None
        self._cos: dict[str, dict[int, list[float]]] | None = None
        self._drift: dict[str, dict[int, float]] | None = None
        self.images: list[dict] = []

    # ----------------------------------------------------------------- image lifecycle
    def begin_image(self, num_steps: int | None) -> None:
        if self._cos is not None:
            self.end_image()
        self.num_steps = num_steps
        for tr in self.tracks.values():
            tr.clear(self.depth)
        self._cos, self._drift = {}, {}

    def end_image(self) -> dict:
        """Close the current image; returns and stores {"keys", "cos" (L, T, n_lags), "drift" (L, T)}."""
        if self._cos is None:
            raise RuntimeError("end_image() without begin_image()")
        steps_seen = [s for d in self._cos.values() for s in d] + [s for d in self._drift.values() for s in d]
        t = self.num_steps or (max(steps_seen) + 1 if steps_seen else 0)
        keys = list(self.order)
        cos = np.full((len(keys), t, len(self.lags)), np.nan, dtype=np.float64)
        drift = np.full((len(keys), t), np.nan, dtype=np.float64)
        for li, k in enumerate(keys):
            for step, vals in self._cos.get(k, {}).items():
                if step < t:
                    cos[li, step] = vals
            for step, v in self._drift.get(k, {}).items():
                if step < t:
                    drift[li, step] = v
        out = {"keys": keys, "cos": cos, "drift": drift, "lags": list(self.lags)}
        self.images.append(out)
        self._cos = self._drift = None
        return out

    # ----------------------------------------------------------------- observation
    def _track(self, key: str, xg: torch.Tensor, c: int) -> _Track:
        tr = self.tracks.get(key)
        if tr is not None:
            if tr.shape != tuple(xg.shape):
                raise ValueError(f"input shape of {key!r} changed from {tr.shape} to {tuple(xg.shape)}")
            return tr
        g = xg.shape[-1]
        cp = xg.shape[-2] * g
        lead = int(np.prod(xg.shape[:-2])) if xg.dim() > 2 else 1
        orig = subset_indices(lead * c, self.n_entries, key_seed(key, self.seed))
        padded = (orig // c) * cp + (orig % c)
        tr = _Track(idx=torch.as_tensor(padded, device=xg.device), shape=tuple(xg.shape))
        tr.clear(self.depth)
        self.tracks[key] = tr
        self.order.append(key)
        return tr

    @torch.no_grad()
    def observe(self, key: str, xg: torch.Tensor, c: int, step: int, bits: int) -> None:
        """Called by QuantLinear with the grouped input (before any carry) of quantiser `key`."""
        if self._cos is None:
            return  # not inside an image
        tr = self._track(key, xg, c)
        g = xg.shape[-1]
        a = xg.reshape(-1)[tr.idx].float()
        s = group_scale(xg, bits).reshape(-1)[tr.idx // g].float()
        qmax = qmax_for(bits)
        r = a - torch.clamp(torch.round(a / s), -qmax, qmax) * s
        self.observe_leftover(key, r, step, a=a, s=s, _track=tr)

    @torch.no_grad()
    def observe_leftover(self, key: str, r: torch.Tensor, step: int, a: torch.Tensor | None = None,
                         s: torch.Tensor | None = None, _track: _Track | None = None) -> None:
        """Record a leftover vector directly (used by observe() and by tests with synthetic leftovers)."""
        if self._cos is None:
            raise RuntimeError("observe_leftover() outside begin_image()/end_image()")
        tr = _track
        if tr is None:
            tr = self.tracks.get(key)
            if tr is None:
                tr = _Track(idx=torch.arange(r.numel()), shape=(r.numel(),))
                tr.clear(self.depth)
                self.tracks[key] = tr
                self.order.append(key)
        r = r.reshape(-1).float()
        if tr.ring is None:
            tr.ring = torch.zeros((self.depth, r.numel()), dtype=torch.float32, device=r.device)
        slot = step % self.depth
        tr.ring[slot] = r
        tr.ring_steps[slot] = step

        cos_vals = [float("nan")] * len(self.lags)
        rows, which = [], []
        for li, lag in enumerate(self.lags):
            j = step - lag
            if j >= 0 and tr.ring_steps[j % self.depth] == j:
                rows.append(j % self.depth)
                which.append(li)
        if rows:
            past = tr.ring[rows]
            dots = past @ r
            norms = past.norm(dim=1) * r.norm()
            vals = torch.where(norms > 0, dots / norms.clamp_min(1e-30), torch.full_like(dots, float("nan")))
            for li, v in zip(which, vals.cpu().tolist()):
                cos_vals[li] = v
        self._cos.setdefault(key, {})[step] = cos_vals

        if a is not None and s is not None:
            a = a.reshape(-1).float()
            if tr.prev_a is not None and tr.prev_step == step - 1:
                sn = float(s.norm())
                self._drift.setdefault(key, {})[step] = float((a - tr.prev_a).norm()) / sn if sn > 0 else float("nan")
            tr.prev_a = a.clone()
            tr.prev_step = step


# --------------------------------------------------------------------------- aggregation
def _stats(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "mean": None}
    return {"n": int(v.size), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)), "mean": float(np.mean(v))}


def layer_kind(key: str) -> str:
    """'transformer_blocks.7.attn.qkv_in' -> 'transformer_blocks.*.attn.qkv_in'."""
    return re.sub(r"^(transformer_blocks|single_transformer_blocks)\.\d+\.", r"\1.*.", key)


def _nan_to_none(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    return x


def summarize(images: list[dict]) -> dict:
    """Pool per-image recordings (same keys and step count) into the E1.1 summary."""
    if not images:
        raise ValueError("no images recorded")
    keys, lags = images[0]["keys"], list(images[0]["lags"])
    for im in images[1:]:
        if im["keys"] != keys or list(im["lags"]) != lags or im["cos"].shape != images[0]["cos"].shape:
            raise ValueError("images were recorded with different layers, lags or step counts")
    cos = np.stack([im["cos"] for im in images])  # (N, L, T, n_lags)
    drift = np.stack([im["drift"] for im in images])  # (N, L, T)
    out: dict = {"n_images": len(images), "n_layers": len(keys), "n_steps": int(cos.shape[2]), "lags": lags,
                 "keys": keys}
    out["cos_by_lag"] = {str(lag): _stats(cos[..., li]) for li, lag in enumerate(lags)}
    out["drift_ratio"] = _stats(drift)
    if 1 in lags:
        l1 = lags.index(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices (step 0) give NaN
            mat = np.nanmedian(cos[..., l1], axis=0)  # (L, T)
            per_step = np.nanmedian(mat, axis=0)
            per_image = [float(np.nanmedian(cos[n, ..., l1])) for n in range(cos.shape[0])]
        out["lag1_matrix"] = _nan_to_none(mat.tolist())
        out["lag1_median_per_step"] = _nan_to_none(per_step.tolist())
        out["lag1_median_per_image"] = _nan_to_none(per_image)
        kinds: dict[str, list[int]] = {}
        for li, k in enumerate(keys):
            kinds.setdefault(layer_kind(k), []).append(li)
        out["lag1_by_kind"] = {k: _stats(cos[:, idx, :, l1]) for k, idx in kinds.items()}
    return out
