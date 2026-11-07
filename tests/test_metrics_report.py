"""Metrics, the slope fit, and the decision table logic."""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from sbb.metrics import LPIPSMetric, bootstrap_median_ci, fit_slope, psnr, rel_err_sq, slope_table
from sbb.report import build_report, decide
from sbb.runtime import load_config


def test_rel_err_and_psnr():
    z = torch.ones(2, 8)
    assert rel_err_sq(z * 1.1, z) == pytest.approx(0.01)
    img = torch.zeros(1, 3, 4, 4)
    assert psnr(img + 0.1, img) == pytest.approx(20.0)
    assert math.isinf(psnr(img, img))


def test_fit_slope_recovers_power_law_and_ignores_small_T():
    ts = [8, 16, 28, 50, 100]
    errs = [3.0 * t ** -0.8 for t in ts]
    errs[0] = 100.0  # T = 8 is excluded by t_min = 16
    slope, intercept = fit_slope(ts, errs, t_min=16)
    assert slope == pytest.approx(-0.8) and intercept == pytest.approx(math.log(3.0))
    assert math.isnan(fit_slope([16], [0.1])[0])


def test_bootstrap_interval_contains_median():
    rng = np.random.default_rng(0)
    v = rng.normal(-1.0, 0.1, size=32)
    med, lo, hi = bootstrap_median_ci(v, n_boot=500)
    assert lo <= med <= hi and lo < -0.95 < hi + 0.1


def test_lpips_wrapper_with_mock(monkeypatch):
    import sbb.metrics as m

    class Mock(torch.nn.Module):
        def forward(self, a, b):
            assert a.min() >= -1 and a.max() <= 1
            return ((a - b) ** 2).mean().reshape(1)

    monkeypatch.setattr(m, "_load_lpips_model", lambda net="alex": Mock())
    lp = LPIPSMetric("cpu")
    a = torch.rand(1, 3, 8, 8)
    assert lp(a, a) == 0.0 and lp(a, 1 - a) > 0


# --------------------------------------------------------------------------- decision table
def _law(slopes: dict, base: dict, lpips: dict, n_prompts=8, ts=(8, 16, 28, 50, 100), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_prompts):
        for mode, k in slopes.items():
            for t in ts:
                e = base[mode] * (t / 16) ** k * math.exp(rng.normal(0, 0.02))
                rows.append({"prompt_id": pid, "T": t, "mode": mode, "bits": 4, "rel_err": e, "rel_err_sq": e * e,
                             "psnr": 30.0, "lpips": lpips[mode]})
    return pd.DataFrame(rows)


def _corr(v):
    return {"cos_by_lag": {"1": {"median": v, "q25": v, "q75": v, "n": 10}}, "settings": {"bits": 4, "T": 50}}


CFG = load_config()


def test_decision_go():
    law = _law({"RTN": 0.0, "SR": -0.5, "SD": -1.0}, {"RTN": 0.2, "SR": 0.2, "SD": 0.1},
               {"RTN": 0.3, "SR": 0.25, "SD": 0.1})
    res = decide(_corr(0.8), law, None, CFG)
    assert res["verdict"] == "GO", res
    assert res["cells"]["sd_slope"]["value"] == pytest.approx(-1.0, abs=0.05)


def test_decision_kill_on_low_correlation():
    law = _law({"RTN": 0.0, "SR": -0.5, "SD": -1.0}, {"RTN": 0.2, "SR": 0.2, "SD": 0.1},
               {"RTN": 0.3, "SR": 0.25, "SD": 0.1})
    assert decide(_corr(0.2), law, None, CFG)["verdict"] == "KILL"


def test_decision_kill_on_small_gain():
    law = _law({"RTN": 0.0, "SR": -0.5, "SD": -0.02}, {"RTN": 0.2, "SR": 0.2, "SD": 0.2},
               {"RTN": 0.3, "SR": 0.25, "SD": 0.29})
    res = decide(_corr(0.8), law, None, CFG)
    assert res["verdict"] == "KILL" and res["cells"]["sd_ratio"]["status"] == "KILL"


def test_decision_weak_when_sd_only_matches_sr():
    law = _law({"RTN": 0.0, "SR": -0.5, "SD": -0.5}, {"RTN": 0.2, "SR": 0.2, "SD": 0.1},
               {"RTN": 0.3, "SR": 0.25, "SD": 0.1})
    assert decide(_corr(0.8), law, None, CFG)["verdict"] == "WEAK"


def test_decision_incomplete_and_report_text():
    res = decide(None, None, None, CFG)
    assert res["verdict"] == "INCOMPLETE"
    law = _law({"RTN": 0.0, "SR": -0.5, "SD": -1.0}, {"RTN": 0.2, "SR": 0.2, "SD": 0.1},
               {"RTN": 0.3, "SR": 0.25, "SD": 0.1})
    meta = {"generated": "now", "git": {"commit": "abc", "dirty": False}, "inputs": {}}
    text = build_report(CFG, _corr(0.8) | {"n_images": 1, "n_layers": 1, "drift_ratio": {"median": 0.1, "q25": 0.1,
                        "q75": 0.1}, "settings": {"bits": 4, "T": 50, "mode": "RTN", "n_entries": 10, "seed": 0}},
                        law, None, None, meta, {})
    assert "## Verdict: **GO**" in text
    assert "—" not in text and "–" not in text
