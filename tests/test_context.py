"""A7.6: step ratios and the pipeline callback."""
import pytest
import torch

from sbb.context import StepCallback, StepContext, gammas_from_sigmas, step_sizes
from sbb.flux import PromptItem, generate, tiny_embeddings
from sbb.wrap import wrap_transformer


def test_gammas_hand_worked_example():
    # sigma = 1.0, 0.8, 0.5, 0.0 -> h = 0.2, 0.3, 0.5 -> gamma = 0, 0.2/0.3, 0.3/0.5
    sig = [1.0, 0.8, 0.5, 0.0]
    assert step_sizes(sig) == pytest.approx([0.2, 0.3, 0.5])
    assert gammas_from_sigmas(sig) == pytest.approx([0.0, 2.0 / 3.0, 0.6])
    assert gammas_from_sigmas(torch.tensor(sig)) == pytest.approx([0.0, 2.0 / 3.0, 0.6])


def test_gammas_uniform_schedule_is_one():
    assert gammas_from_sigmas([1.0, 0.75, 0.5, 0.25, 0.0]) == pytest.approx([0.0, 1.0, 1.0, 1.0])


def test_gammas_reject_non_decreasing():
    with pytest.raises(ValueError):
        gammas_from_sigmas([1.0, 1.0, 0.0])


def test_callback_advances_index_through_the_stock_pipeline(tiny_pipe):
    ctx = StepContext().configure(mode="SD", bits=4)
    wrap_transformer(tiny_pipe.transformer, ctx)
    seen = []
    tiny_pipe.transformer.register_forward_pre_hook(lambda m, a: seen.append((ctx.step, ctx.gamma)))
    emb = tiny_embeddings(1)
    item = PromptItem(0, 0, "p", emb["prompt_embeds"][:1], emb["pooled_prompt_embeds"][:1])
    T = 6
    res = generate(tiny_pipe, ctx, item, T, 32, 32, 3.5)
    expected = gammas_from_sigmas(tiny_pipe.scheduler.sigmas)
    assert len(tiny_pipe.scheduler.sigmas) == T + 1
    assert [s for s, _ in seen] == list(range(T))
    assert [g for _, g in seen] == pytest.approx(expected)
    assert res.gammas == pytest.approx(expected)
    assert ctx.step == T  # advanced after the last step


def test_callback_returns_kwargs_unchanged_and_checks_sync():
    ctx = StepContext()
    ctx.begin_image(num_steps=3)

    class Sched:
        sigmas = torch.tensor([1.0, 0.6, 0.3, 0.0])

    class Pipe:
        scheduler = Sched()

    cb = StepCallback(ctx)
    kw = {"latents": torch.ones(1)}
    assert cb(Pipe(), 0, None, kw) is kw
    assert ctx.step == 1 and ctx.gamma == pytest.approx(0.4 / 0.3)
    with pytest.raises(RuntimeError, match="out of sync"):
        cb(Pipe(), 0, None, kw)
