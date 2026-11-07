"""Activation fake-quantisation with an optional leftover carry across sampling steps.

Quantiser (all modes): symmetric uniform, b bits, per token, per group of `group_size` input channels.
    s = max|a'| over the group / (2^(b-1) - 1)
    q = clamp(round(a' / s), -(2^(b-1) - 1), 2^(b-1) - 1) * s
computed in float32 (float64 if the input is float64) and cast back to the input dtype.
The last group is zero-padded when the channel count is not a multiple of the group size; padding
never changes a group maximum and is removed afterwards.

Modes
    FP           pass-through
    RTN          a' = a
    SR           q = floor(a / s + U[0, 1)) * s, fresh U on every call from a seeded generator
    SD           a' = a + gamma_i * r_prev, q = RTN(a'), r = a' - q kept for the next step
    SD_UNSCALED  as SD with gamma_i = 1 for every step after the first

The weight stays in its original precision (BF16 for FLUX); only the input of the linear is quantised.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .context import CARRY_MODES, STORAGES, StepContext


def qmax_for(bits: int) -> int:
    if bits < 2:
        raise ValueError("bits must be at least 2")
    return 2 ** (bits - 1) - 1


def compute_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def to_groups(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    """(..., C) -> (..., ceil(C / g), g), zero-padding the last group. Returns the view and C."""
    c = x.shape[-1]
    pad = (-c) % group_size
    if pad:
        x = F.pad(x, (0, pad))
    return x.reshape(*x.shape[:-1], (c + pad) // group_size, group_size), c


def from_groups(xg: torch.Tensor, c: int) -> torch.Tensor:
    x = xg.reshape(*xg.shape[:-2], xg.shape[-2] * xg.shape[-1])
    return x[..., :c] if x.shape[-1] != c else x


def group_scale(xg: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-group scale s = max|x| / qmax with shape (..., G, 1). All-zero groups get s = 1 (q = 0 exactly)."""
    amax = xg.abs().amax(dim=-1, keepdim=True)
    return torch.where(amax > 0, amax / qmax_for(bits), torch.ones_like(amax))


def rtn_grouped(xg: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = qmax_for(bits)
    s = group_scale(xg, bits)
    q = torch.clamp(torch.round(xg / s), -qmax, qmax) * s
    return q, s


def sr_grouped(xg: torch.Tensor, bits: int, generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = qmax_for(bits)
    s = group_scale(xg, bits)
    u = torch.rand(xg.shape, generator=generator, device=xg.device, dtype=xg.dtype)
    q = torch.clamp(torch.floor(xg / s + u), -qmax, qmax) * s
    return q, s


def fake_quant(x: torch.Tensor, bits: int = 4, group_size: int = 64, mode: str = "RTN",
               generator: torch.Generator | None = None) -> torch.Tensor:
    """Stateless RTN or SR fake quantisation of a tensor (helper for tests and analysis)."""
    xg, c = to_groups(x.to(compute_dtype(x)), group_size)
    if mode == "RTN":
        qg, _ = rtn_grouped(xg, bits)
    elif mode == "SR":
        qg, _ = sr_grouped(xg, bits, generator)
    else:
        raise ValueError("fake_quant supports RTN and SR only")
    return from_groups(qg, c).to(x.dtype)


class ResidualStore:
    """Holds the rounding leftover of one quantiser between steps, in a chosen precision.

    fp32  leftover kept in the compute dtype (float32, or float64 for float64 inputs)
    bf16  leftover cast to bfloat16
    int8  integer codes with step s / 2^(8-1) plus one bf16 scale per group
    int4  as int8 with step s / 2^(4-1), codes packed two per byte

    The int step s / 2^(b_r - 1) follows E1_INSTRUCTIONS.md (A1). Because |r| <= s / 2 the codes only
    reach +-2^(b_r - 2), so one code bit is unused; change STEP_EXPONENT_OFFSET to 2 to use all levels.
    """

    STEP_EXPONENT_OFFSET = 1  # step = s / 2^(b_r - STEP_EXPONENT_OFFSET)

    def __init__(self, storage: str, dtype: torch.dtype):
        if storage not in STORAGES:
            raise ValueError(f"unknown storage {storage!r}; expected one of {STORAGES}")
        self.storage = storage
        self.dtype = dtype
        self.data: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.shape: torch.Size | None = None

    @property
    def code_bits(self) -> int:
        return {"int8": 8, "int4": 4}[self.storage]

    def _step(self, scale_stored: torch.Tensor) -> torch.Tensor:
        return scale_stored.to(self.dtype) / float(2 ** (self.code_bits - self.STEP_EXPONENT_OFFSET))

    def put(self, r: torch.Tensor, s: torch.Tensor) -> None:
        """r: leftover in grouped layout (..., G, g); s: the scale used at this step, shape (..., G, 1)."""
        self.shape = r.shape
        if self.storage == "fp32":
            self.data, self.scale = r.detach().to(self.dtype).clone(), None
            return
        if self.storage == "bf16":
            self.data, self.scale = r.detach().to(torch.bfloat16), None
            return
        cmax = 2 ** (self.code_bits - 1) - 1
        scale_stored = s.detach().to(torch.bfloat16)
        codes = torch.clamp(torch.round(r.detach().to(self.dtype) / self._step(scale_stored)), -cmax, cmax)
        if self.storage == "int8":
            self.data = codes.to(torch.int8)
        else:
            if r.shape[-1] % 2:
                raise ValueError("int4 storage needs an even group size")
            u = (codes + 8).to(torch.uint8)  # 1..15
            self.data = u[..., 0::2] | (u[..., 1::2] << 4)
        self.scale = scale_stored

    def get(self) -> torch.Tensor | None:
        if self.data is None:
            return None
        if self.storage in ("fp32", "bf16"):
            return self.data.to(self.dtype)
        if self.storage == "int8":
            codes = self.data.to(self.dtype)
        else:
            lo = (self.data & 0x0F).to(self.dtype)
            hi = ((self.data >> 4) & 0x0F).to(self.dtype)
            codes = torch.stack((lo, hi), dim=-1).reshape(*self.data.shape[:-1], self.data.shape[-1] * 2) - 8
        return codes * self._step(self.scale)

    def nbytes(self) -> int:
        n = 0
        for t in (self.data, self.scale):
            if t is not None:
                n += t.numel() * t.element_size()
        return n

    def clear(self) -> None:
        self.data = self.scale = None
        self.shape = None


class QuantState:
    """Quantiser state, shared by linears that read the same input tensor (for example to_q, to_k, to_v).

    A real low-bit kernel quantises such an input once (fused QKV), so sharing keeps SR noise and
    leftover buffers faithful to that, and stores one buffer instead of three. Within one step the
    first member computes the quantised input; the others reuse it after checking it is the same
    tensor object.
    """

    def __init__(self, key: str):
        self.key = key
        self.n_members = 0
        self.members: list[str] = []
        self.store: ResidualStore | None = None
        self.image: int | None = None
        self.step: int | None = None
        self.x: torch.Tensor | None = None
        self.q: torch.Tensor | None = None
        self.uses = 0

    def reset(self) -> None:
        if self.store is not None:
            self.store.clear()
        self.store = None
        self.image = self.step = None
        self.x = self.q = None
        self.uses = 0

    def nbytes(self) -> int:
        return self.store.nbytes() if self.store is not None else 0


class QuantLinear(nn.Module):
    """nn.Linear whose input is fake-quantised according to a StepContext; the weight is untouched."""

    def __init__(self, linear: nn.Linear, ctx: StepContext, name: str = "", state: QuantState | None = None):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("QuantLinear wraps nn.Linear only")
        self.linear = linear
        self.ctx = ctx
        self.name = name
        if state is None:
            state = QuantState(name)
        if name not in state.members:
            state.members.append(name)
            state.n_members = len(state.members)
        self.state = state

    # Delegate the attributes that diffusers code may read from a linear layer.
    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def extra_repr(self) -> str:
        return f"name={self.name!r}, group={self.state.key!r}"

    def reset(self) -> None:
        self.state.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = self.ctx
        if ctx.mode == "FP":
            return self.linear(x)
        st = self.state
        same_step = st.image == ctx.image_index and st.step == ctx.step
        if same_step and st.q is not None and st.x is x:
            q = st.q
        else:
            if same_step and ctx.mode in CARRY_MODES:
                raise RuntimeError(
                    f"quantiser {st.key!r} was called twice in step {ctx.step} with different inputs; the leftover "
                    "carry needs exactly one quantisation per step (advance ctx.step between forward passes)")
            q = self._quantize(x)
            st.image, st.step, st.x, st.q, st.uses = ctx.image_index, ctx.step, x, q, 0
        st.uses += 1
        if st.uses >= st.n_members:
            st.x = st.q = None  # every member has read it; release the references
        return self.linear(q)

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        ctx, st = self.ctx, self.state
        cdt = compute_dtype(x)
        xg, c = to_groups(x.to(cdt), ctx.group_size)
        if ctx.recorder is not None:
            ctx.recorder.observe(st.key, xg, c, ctx.step, ctx.bits)
        mode = ctx.mode
        if mode == "RTN":
            qg, _ = rtn_grouped(xg, ctx.bits)
            return from_groups(qg, c).to(x.dtype)
        if mode == "SR":
            qg, _ = sr_grouped(xg, ctx.bits, ctx.sr_generator(x.device))
            return from_groups(qg, c).to(x.dtype)
        if mode in CARRY_MODES:
            gamma = ctx.effective_gamma()
            if st.store is None or st.store.storage != ctx.storage or st.store.dtype != cdt:
                st.store = ResidualStore(ctx.storage, cdt)
            if gamma != 0.0:
                r_prev = st.store.get()
                if r_prev is not None:
                    if r_prev.shape != xg.shape:
                        raise ValueError(f"leftover shape {tuple(r_prev.shape)} does not match input "
                                         f"{tuple(xg.shape)} in {st.key!r}; call reset() between images")
                    xg = xg + gamma * r_prev
            qg, s = rtn_grouped(xg, ctx.bits)
            q = from_groups(qg, c).to(x.dtype)
            # Leftover relative to what the matmul actually sees (q after the cast back to the input dtype).
            r = xg - to_groups(q.to(cdt), ctx.group_size)[0]
            st.store.put(r, s)
            return q
        raise ValueError(f"unknown mode {mode!r}")
