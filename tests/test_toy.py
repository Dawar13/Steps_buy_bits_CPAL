"""A7.4 and A7.5: telescoping and the three-regime law on the toy linear system."""
import numpy as np
import pytest
import torch

from sbb.metrics import fit_slope
from sbb.toy import ToySystem, shifted_sigmas, telescoped_error, toy_run


def test_shifted_schedule_is_uneven():
    sig = shifted_sigmas(20)
    h = sig[:-1] - sig[1:]
    assert sig[0] == pytest.approx(1.0) and sig[-1] == pytest.approx(0.0)
    assert h.max() / h.min() > 2.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sd_error_equals_boundary_plus_sensitivity_drift(seed):
    run = toy_run(ToySystem(seed=seed), "SD", 50)
    rhs = telescoped_error(run)
    assert torch.allclose(run["delta"], rhs, rtol=1e-9, atol=1e-12)


def test_sd_error_much_smaller_than_rtn():
    ratios = []
    for seed in range(4):
        system = ToySystem(seed=seed)
        ratios.append(toy_run(system, "SD", 50)["rel_err"] / toy_run(system, "RTN", 50)["rel_err"])
    assert max(ratios) < 0.3


def test_constant_sensitivity_leaves_only_the_boundary_term():
    class Flat(ToySystem):
        @staticmethod
        def c(sigma):
            return 1.0

    run = toy_run(Flat(seed=0), "SD", 40)
    h, r, W = run["h"], run["leftovers"], run["W"]
    boundary = h[-1] * (r[-1] @ W.T)  # x_q - x_fp = +h_(T-1) W r_(T-1) when G is constant
    assert torch.allclose(run["delta"], boundary, rtol=1e-9, atol=1e-12)


def test_toy_law_slopes():
    ts = [16, 32, 64, 128, 256]
    slopes = {m: [] for m in ("RTN", "SR", "SD")}
    for seed in range(8):
        system = ToySystem(seed=seed)
        for m in slopes:
            errs = [toy_run(system, m, t, seed=seed)["rel_err"] for t in ts]
            slopes[m].append(fit_slope(ts, errs, t_min=16)[0])
    med = {m: float(np.median(v)) for m, v in slopes.items()}
    assert abs(med["RTN"]) < 0.2, med
    assert -0.65 < med["SR"] < -0.35, med
    assert -1.3 < med["SD"] < -0.75, med
