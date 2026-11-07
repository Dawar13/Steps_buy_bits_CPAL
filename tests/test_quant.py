"""A7.1 to A7.3: the quantiser, stochastic rounding, and the leftover carry."""
import pytest
import torch
from torch import nn

from sbb.context import StepContext
from sbb.quant import (QuantLinear, ResidualStore, fake_quant, from_groups, group_scale, qmax_for, rtn_grouped,
                       to_groups)


def _activations(seed=0, tokens=64, channels=200):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, channels, generator=g)
    x[:, 5] *= 30.0  # an outlier channel in the first group
    x[3] *= 1e-3  # a token with tiny values
    return x


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_rtn_error_at_most_half_step_no_clamp_and_level_count(bits):
    x = _activations()
    q = fake_quant(x, bits=bits, group_size=64, mode="RTN")
    xg, c = to_groups(x, 64)
    qg, _ = to_groups(q, 64)
    s = group_scale(xg, bits)
    qmax = qmax_for(bits)
    assert torch.all((qg - xg).abs() <= s / 2 * (1 + 1e-5) + 1e-12)
    assert torch.all((xg / s).abs() <= qmax * (1 + 1e-6))  # never clamps
    codes = torch.round(qg / s)
    for t in range(xg.shape[0]):
        for gi in range(xg.shape[1]):
            assert codes[t, gi].unique().numel() <= 2 ** bits - 1
    assert codes.abs().max() <= qmax


def test_partial_last_group_matches_separate_groups():
    x = _activations(channels=100)
    q = fake_quant(x, bits=4, group_size=64)
    q_a = fake_quant(x[:, :64].contiguous(), bits=4, group_size=64)
    q_b = fake_quant(x[:, 64:].contiguous(), bits=4, group_size=64)
    assert torch.equal(q, torch.cat([q_a, q_b], dim=1))


def test_zero_group_and_dtype():
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    x[1, 70] = 1.5
    q = fake_quant(x, bits=4)
    assert q.dtype == torch.bfloat16
    assert torch.equal(q[0], x[0])
    assert torch.equal(q[1], x[1])  # the group maximum is exactly representable


def test_groups_roundtrip():
    x = _activations(channels=130)
    xg, c = to_groups(x, 64)
    assert xg.shape == (64, 3, 64)
    assert torch.equal(from_groups(xg, c), x)


def test_sr_is_unbiased():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(1, 64, generator=g)
    x = base.expand(100_000, 64).contiguous()
    q = fake_quant(x, bits=4, group_size=64, mode="SR", generator=torch.Generator().manual_seed(1))
    s = float(group_scale(to_groups(base, 64)[0], 4).max())
    err = q - x
    assert err.abs().max() < s * (1 + 1e-5)
    # Each entry's error has std <= s/2, so the mean of 1e5 draws has std <= 0.0016 s.
    assert err.mean(dim=0).abs().max() < 0.01 * s
    assert abs(float(err.mean())) < 0.002 * s


def test_sr_seeded_reproducible():
    x = _activations()
    a = fake_quant(x, mode="SR", generator=torch.Generator().manual_seed(5))
    b = fake_quant(x, mode="SR", generator=torch.Generator().manual_seed(5))
    c = fake_quant(x, mode="SR", generator=torch.Generator().manual_seed(6))
    assert torch.equal(a, b) and not torch.equal(a, c)


def _qlin(mode, bits=4, storage="fp32", gammas=None, override=None):
    torch.manual_seed(0)
    lin = nn.Linear(200, 16)
    ctx = StepContext().configure(mode=mode, bits=bits, storage=storage)
    ctx.gamma_override = override
    ql = QuantLinear(lin, ctx, "layer")
    ctx.begin_image(seed=0, num_steps=6, gammas=gammas)
    return lin, ql, ctx


def _inputs(n=6):
    return [_activations(seed=i) for i in range(n)]


def test_sd_with_gamma_zero_equals_rtn_exactly():
    xs = _inputs()
    _, rtn, ctx_r = _qlin("RTN")
    _, sd, ctx_s = _qlin("SD", gammas=[0.0, 1.3, 0.7, 1.1, 0.9, 1.2], override=0.0)
    for i, x in enumerate(xs):
        ctx_r.set_step(i)
        ctx_s.set_step(i)
        assert torch.equal(rtn(x), sd(x))


def test_sd_carry_matches_definition():
    xs = _inputs(4)
    gammas = [0.0, 0.8, 1.25, 0.5]
    lin, sd, ctx = _qlin("SD", gammas=gammas)
    r = torch.zeros(64, 4, 64)
    for i, x in enumerate(xs):
        ctx.set_step(i)
        out = sd(x)
        xg, c = to_groups(x, 64)
        a_prime = xg + gammas[i] * r
        qg, s = rtn_grouped(a_prime, 4)
        expected = lin(from_groups(qg, c))
        assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)
        r = a_prime - qg
        stored = sd.state.store.get()
        assert torch.allclose(stored, r, atol=1e-6)
        assert torch.all(stored.abs() <= s / 2 * (1 + 1e-5) + 1e-12)


def test_sd_unscaled_uses_gamma_one():
    xs = _inputs(3)
    _, a, ca = _qlin("SD_UNSCALED", gammas=[0.0, 0.3, 0.6])
    _, b, cb = _qlin("SD", gammas=[0.0, 1.0, 1.0])
    for i, x in enumerate(xs):
        ca.set_step(i)
        cb.set_step(i)
        assert torch.equal(a(x), b(x))


def test_sd_step_zero_is_rtn_and_needs_gammas_later():
    x = _activations()
    _, rtn, cr = _qlin("RTN")
    _, sd, cs = _qlin("SD", gammas=None)
    assert torch.equal(rtn(x), sd(x))
    cs.set_step(1)
    with pytest.raises(RuntimeError, match="step-size ratios"):
        sd(_activations(seed=1))


def test_sd_twice_in_one_step_raises():
    _, sd, ctx = _qlin("SD", gammas=[0.0, 1.0])
    sd(_activations(0))
    with pytest.raises(RuntimeError, match="called twice"):
        sd(_activations(1))


@pytest.mark.parametrize("storage,code_bits", [("int8", 8), ("int4", 4)])
def test_integer_storage_error_within_half_code_step(storage, code_bits):
    g = torch.Generator().manual_seed(0)
    xg = torch.randn(8, 3, 64, generator=g)
    q, s = rtn_grouped(xg, 4)
    r = xg - q
    st = ResidualStore(storage, torch.float32)
    st.put(r, s)
    back = st.get()
    cmax = 2 ** (code_bits - 1) - 1
    step = s.to(torch.bfloat16).float() / 2 / cmax  # (s / 2) / (2^(b_r - 1) - 1)
    assert back.shape == r.shape
    assert torch.all((back - r).abs() <= step / 2 * (1 + 1e-4) + 1e-12)
    codes = torch.round(back / step)
    assert codes.abs().max() == cmax  # leftovers near s / 2 use the full code range


def test_storage_sizes_and_exact_fp32():
    xg = torch.randn(8, 3, 64)
    q, s = rtn_grouped(xg, 4)
    r = xg - q
    sizes = {}
    for storage in ("fp32", "bf16", "int8", "int4"):
        st = ResidualStore(storage, torch.float32)
        st.put(r, s)
        sizes[storage] = st.nbytes()
        if storage == "fp32":
            assert torch.equal(st.get(), r)
        if storage == "bf16":
            assert torch.allclose(st.get(), r, rtol=1e-2, atol=1e-6)
        st.clear()
        assert st.get() is None and st.nbytes() == 0
    n = r.numel()
    assert sizes["fp32"] == 4 * n and sizes["bf16"] == 2 * n
    assert sizes["int8"] == n + 2 * 8 * 3 and sizes["int4"] == n // 2 + 2 * 8 * 3


def test_carry_with_low_bit_storage_runs_and_stays_bounded():
    xs = _inputs(4)
    for storage in ("bf16", "int8", "int4"):
        _, sd, ctx = _qlin("SD", storage=storage, gammas=[0.0, 1.0, 1.0, 1.0])
        for i, x in enumerate(xs):
            ctx.set_step(i)
            y = sd(x)
            assert torch.isfinite(y).all()
        assert sd.state.store.storage == storage
