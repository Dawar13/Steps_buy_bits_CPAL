"""A7.8: the correlation recorder."""
import numpy as np
import pytest
import torch

from sbb.context import StepContext
from sbb.corr import CorrRecorder, layer_kind, subset_indices, summarize
from sbb.flux import PromptItem, generate, tiny_embeddings
from sbb.quant import to_groups
from sbb.wrap import quant_states, wrap_transformer


@pytest.mark.parametrize("rho", [0.3, 0.8, 0.95])
def test_known_ar1_correlation_is_recovered(rho):
    g = torch.Generator().manual_seed(0)
    n, steps = 65536, 30
    rec = CorrRecorder(n_entries=n, lags=(1, 2, 5, 10))
    rec.begin_image(steps)
    r = torch.randn(n, generator=g)
    for i in range(steps):
        if i:
            r = rho * r + (1 - rho ** 2) ** 0.5 * torch.randn(n, generator=g)
        rec.observe_leftover("synthetic", r, i)
    im = rec.end_image()
    summ = summarize([im])
    for lag in (1, 2, 5, 10):
        assert summ["cos_by_lag"][str(lag)]["median"] == pytest.approx(rho ** lag, abs=0.02)
    assert np.isnan(im["cos"][0, 0, 0]) and np.isfinite(im["cos"][0, 1, 0])


def test_identical_inputs_give_cosine_one_and_zero_drift():
    x = torch.randn(16, 128)
    xg, c = to_groups(x, 64)
    rec = CorrRecorder(n_entries=1000, lags=(1, 2))
    rec.begin_image(5)
    for i in range(5):
        rec.observe("layer", xg, c, i, 4)
    im = rec.end_image()
    assert np.allclose(im["cos"][0, 1:, 0], 1.0)
    assert np.allclose(im["cos"][0, 2:, 1], 1.0)
    assert np.allclose(im["drift"][0, 1:], 0.0)


def test_independent_inputs_give_cosine_near_zero():
    g = torch.Generator().manual_seed(1)
    rec = CorrRecorder(n_entries=8192, lags=(1,))
    rec.begin_image(6)
    for i in range(6):
        xg, c = to_groups(torch.randn(64, 256, generator=g), 64)
        rec.observe("layer", xg, c, i, 4)
    summ = summarize([rec.end_image()])
    assert abs(summ["cos_by_lag"]["1"]["median"]) < 0.05


def test_subset_is_fixed_and_excludes_padding():
    a = subset_indices(10_000, 500, seed=3)
    b = subset_indices(10_000, 500, seed=3)
    assert np.array_equal(a, b) and len(np.unique(a)) == 500
    x = torch.arange(1, 3 * 100 + 1, dtype=torch.float32).reshape(3, 100)  # 100 channels -> padded to 128
    xg, c = to_groups(x, 64)
    rec = CorrRecorder(n_entries=300, lags=(1,))
    rec.begin_image(1)
    rec.observe("k", xg, c, 0, 4)
    vals = xg.reshape(-1)[rec.tracks["k"].idx]
    assert vals.numel() == 300 and torch.all(vals > 0)  # every original entry, no padding


def test_recorder_through_the_tiny_pipeline(tiny_pipe):
    ctx = StepContext().configure(mode="RTN", bits=4)
    wrap_transformer(tiny_pipe.transformer, ctx)
    rec = CorrRecorder(n_entries=512, lags=(1, 2, 5, 10))
    ctx.recorder = rec
    emb = tiny_embeddings(2)
    images = []
    for pid in range(2):
        item = PromptItem(pid, pid, "p", emb["prompt_embeds"][pid:pid + 1], emb["pooled_prompt_embeds"][pid:pid + 1])
        generate(tiny_pipe, ctx, item, 12, 32, 32, 3.5)
        images.append(rec.end_image())
    keys = images[0]["keys"]
    assert set(keys) == {st.key for st in quant_states(tiny_pipe.transformer)}
    summ = summarize(images)
    assert summ["n_layers"] == 10 and summ["n_steps"] == 12
    assert np.array(summ["lag1_matrix"], dtype=float).shape == (10, 12)
    assert summ["cos_by_lag"]["10"]["n"] > 0
    assert -1.0 <= summ["cos_by_lag"]["1"]["median"] <= 1.0
    assert layer_kind("single_transformer_blocks.12.proj_out") == "single_transformer_blocks.*.proj_out"
