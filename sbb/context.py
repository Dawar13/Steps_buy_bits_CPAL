"""Step context shared by every quantised layer during one sampling run.

Execution order is 0-based in code: step i = 0 .. T-1 uses noise level sigma[i] and step size
h[i] = sigma[i] - sigma[i+1]. The carry factor for step i is gamma[i] = h[i-1] / h[i] (gamma[0] = 0).
This is the same as the 1-based gamma_1 = 0, gamma_i = h_(i-1) / h_i in PROJECT_CONTEXT.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

MODES = ("FP", "RTN", "SR", "SD", "SD_UNSCALED")
CARRY_MODES = ("SD", "SD_UNSCALED")
STORAGES = ("fp32", "bf16", "int8", "int4")

# Offset so that the stochastic-rounding stream never coincides with the latent noise seed.
SR_SEED_OFFSET = 1_000_003


def _as_floats(values: Any) -> list[float]:
    if isinstance(values, torch.Tensor):
        return [float(v) for v in values.detach().cpu().flatten().tolist()]
    return [float(v) for v in values]


def step_sizes(sigmas: Any) -> list[float]:
    """h[i] = sigma[i] - sigma[i+1] for a schedule of T+1 noise levels."""
    s = _as_floats(sigmas)
    if len(s) < 2:
        raise ValueError("need at least two noise levels")
    return [s[i] - s[i + 1] for i in range(len(s) - 1)]


def gammas_from_sigmas(sigmas: Any) -> list[float]:
    """Step-size-matched carry factors.

    `sigmas` holds T+1 noise levels in execution order (sigma_1 > ... > sigma_(T+1), usually ending at 0),
    exactly as `pipe.scheduler.sigmas` after `set_timesteps`. Returns a list of length T with
    gamma[0] = 0 and gamma[i] = h[i-1] / h[i] for i >= 1.
    """
    h = step_sizes(sigmas)
    if any(hi <= 0 for hi in h):
        raise ValueError(f"noise levels must be strictly decreasing, got step sizes {h}")
    return [0.0] + [h[i - 1] / h[i] for i in range(1, len(h))]


@dataclass
class StepContext:
    """Mutable state read by every QuantLinear on every call."""

    mode: str = "FP"
    bits: int = 4
    group_size: int = 64
    storage: str = "fp32"  # leftover buffer precision for the carry modes
    step: int = 0  # 0-based index of the step currently being evaluated
    gamma: float = 0.0  # step-matched carry factor for the current step
    gammas: list[float] | None = None
    num_steps: int | None = None
    gamma_override: float | None = None  # testing only: force a fixed gamma at every step
    image_index: int = 0  # increases at every begin_image(); used to detect stale state
    seed: int = 0
    recorder: Any = None  # optional CorrRecorder
    _sr_generators: dict = field(default_factory=dict, repr=False)

    def configure(self, mode: str | None = None, bits: int | None = None, storage: str | None = None,
                  group_size: int | None = None) -> "StepContext":
        if mode is not None:
            if mode not in MODES:
                raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
            self.mode = mode
        if bits is not None:
            if bits < 2:
                raise ValueError("bits must be at least 2")
            self.bits = int(bits)
        if storage is not None:
            if storage not in STORAGES:
                raise ValueError(f"unknown storage {storage!r}; expected one of {STORAGES}")
            self.storage = storage
        if group_size is not None:
            self.group_size = int(group_size)
        return self

    def begin_image(self, seed: int = 0, num_steps: int | None = None,
                    gammas: Sequence[float] | None = None) -> None:
        """Start a new sampling run. Quantiser buffers must be reset separately (see wrap.reset_quant_state)."""
        self.image_index += 1
        self.seed = int(seed)
        self.num_steps = num_steps
        self.gammas = list(gammas) if gammas is not None else None
        self._sr_generators = {}
        self.set_step(0)
        if self.recorder is not None:
            self.recorder.begin_image(num_steps)

    def set_step(self, i: int) -> None:
        self.step = int(i)
        if self.gammas is not None and 0 <= self.step < len(self.gammas):
            self.gamma = float(self.gammas[self.step])
        else:
            self.gamma = 0.0

    def effective_gamma(self) -> float:
        """Carry factor applied to the stored leftover at the current step."""
        if self.gamma_override is not None:
            return float(self.gamma_override)
        if self.step == 0:
            return 0.0
        if self.mode == "SD_UNSCALED":
            return 1.0
        if self.gammas is None:
            raise RuntimeError("SD mode at step > 0 but the step-size ratios are unknown; "
                               "pass StepCallback to the pipeline or call begin_image(gammas=...)")
        return self.gamma

    def sr_generator(self, device: torch.device | str) -> torch.Generator:
        key = str(torch.device(device))
        gen = self._sr_generators.get(key)
        if gen is None:
            gen = torch.Generator(device=torch.device(device))
            gen.manual_seed(SR_SEED_OFFSET + self.seed)
            self._sr_generators[key] = gen
        return gen


class StepCallback:
    """`callback_on_step_end` for the stock diffusers pipeline.

    The pipeline calls it after step i has finished. It reads `pipe.scheduler.sigmas` (set by
    `set_timesteps` inside the pipeline call), computes the step ratios once, and moves the context
    to step i + 1. `callback_kwargs` are returned unchanged.
    """

    def __init__(self, ctx: StepContext, timing: bool = False):
        self.ctx = ctx
        self.calls = 0
        self.sigmas: list[float] | None = None
        self.timing = timing
        self.times: list[float] = []

    def __call__(self, pipe, i: int, t, callback_kwargs: dict) -> dict:
        if self.timing:
            import time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.times.append(time.perf_counter())
        if self.ctx.gammas is None:
            self.sigmas = _as_floats(pipe.scheduler.sigmas)
            self.ctx.gammas = gammas_from_sigmas(self.sigmas)
        if i != self.ctx.step:
            raise RuntimeError(f"step index out of sync: pipeline step {i}, context step {self.ctx.step}")
        self.calls += 1
        self.ctx.set_step(i + 1)
        return callback_kwargs
