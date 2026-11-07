"""A7.7: wrapping the tiny Flux transformer."""
import pytest
import torch
from torch import nn

from conftest import transformer_inputs
from sbb.context import StepContext
from sbb.quant import QuantLinear
from sbb.wrap import (buffer_nbytes, quant_layers, quant_states, reset_quant_state, share_key, unwrap_transformer,
                      wrap_transformer)

EXPECTED = {
    "transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k", "transformer_blocks.0.attn.to_v",
    "transformer_blocks.0.attn.to_out.0", "transformer_blocks.0.attn.add_q_proj",
    "transformer_blocks.0.attn.add_k_proj", "transformer_blocks.0.attn.add_v_proj",
    "transformer_blocks.0.attn.to_add_out", "transformer_blocks.0.ff.net.0.proj", "transformer_blocks.0.ff.net.2",
    "transformer_blocks.0.ff_context.net.0.proj", "transformer_blocks.0.ff_context.net.2",
    "single_transformer_blocks.0.proj_mlp", "single_transformer_blocks.0.proj_out",
    "single_transformer_blocks.0.attn.to_q", "single_transformer_blocks.0.attn.to_k",
    "single_transformer_blocks.0.attn.to_v",
}


def _forward(tr, **kw):
    with torch.no_grad():
        return tr(**transformer_inputs(**kw))[0]


def test_fp_mode_is_identical_and_norm_layers_are_skipped(tiny_pipe):
    tr = tiny_pipe.transformer
    ref = _forward(tr)
    ctx = StepContext()
    names = wrap_transformer(tr, ctx)
    assert set(names) == EXPECTED
    assert not any("norm" in n for n in names)
    ctx.begin_image()
    assert torch.equal(_forward(tr), ref)
    # embedders, modulation layers and the output layer stay plain nn.Linear
    for name in ("x_embedder", "context_embedder", "proj_out", "norm_out.linear",
                 "transformer_blocks.0.norm1.linear", "transformer_blocks.0.norm1_context.linear",
                 "single_transformer_blocks.0.norm.linear"):
        mod = tr.get_submodule(name)
        assert isinstance(mod, nn.Linear) and not isinstance(mod, QuantLinear), name
    assert len(quant_layers(tr)) == 17


def test_rtn_changes_output_and_unwrap_restores(tiny_pipe):
    tr = tiny_pipe.transformer
    ref = _forward(tr)
    ctx = StepContext().configure(mode="RTN", bits=4)
    wrap_transformer(tr, ctx)
    ctx.begin_image()
    out = _forward(tr)
    assert not torch.equal(out, ref)
    assert torch.allclose(out, ref, atol=1.0)  # perturbed, not destroyed
    assert unwrap_transformer(tr) == 17
    assert not quant_layers(tr)
    assert torch.equal(_forward(tr), ref)


def test_share_groups():
    assert share_key("transformer_blocks.3.attn.to_k") == "transformer_blocks.3.attn.qkv_in"
    assert share_key("transformer_blocks.3.attn.add_v_proj") == "transformer_blocks.3.attn.add_qkv_in"
    assert share_key("single_transformer_blocks.2.proj_mlp") == "single_transformer_blocks.2.attn.qkv_in"
    assert share_key("single_transformer_blocks.2.attn.to_q") == "single_transformer_blocks.2.attn.qkv_in"
    assert share_key("single_transformer_blocks.2.proj_out") == "single_transformer_blocks.2.proj_out"


def _run_steps(tr, ctx, mode, steps=3):
    ctx.configure(mode=mode, bits=4)
    ctx.begin_image(num_steps=steps, gammas=[0.0] + [0.9] * (steps - 1))
    reset_quant_state(tr)
    outs = []
    for i in range(steps):
        ctx.set_step(i)
        outs.append(_forward(tr, seed=i, sigma=1.0 - 0.2 * i))
    return outs


@pytest.mark.parametrize("mode", ["RTN", "SD"])
def test_shared_inputs_give_the_same_result_as_separate_quantisers(mode):
    from sbb.flux import tiny_pipeline

    a, b = tiny_pipeline(seed=0).transformer, tiny_pipeline(seed=0).transformer
    ca, cb = StepContext(), StepContext()
    wrap_transformer(a, ca, share_inputs=True)
    wrap_transformer(b, cb, share_inputs=False)
    assert len(quant_states(a)) == 10 and len(quant_states(b)) == 17
    for x, y in zip(_run_steps(a, ca, mode), _run_steps(b, cb, mode)):
        assert torch.equal(x, y)


def test_reset_clears_buffers(tiny_pipe):
    tr = tiny_pipe.transformer
    ctx = StepContext()
    wrap_transformer(tr, ctx)
    _run_steps(tr, ctx, "SD")
    assert buffer_nbytes(tr) > 0
    assert all(st.store is not None and st.store.get() is not None for st in quant_states(tr))
    reset_quant_state(tr)
    assert buffer_nbytes(tr) == 0
    assert all(st.store is None and st.x is None and st.q is None for st in quant_states(tr))


def test_double_wrap_is_refused(tiny_pipe):
    ctx = StepContext()
    wrap_transformer(tiny_pipe.transformer, ctx)
    with pytest.raises(RuntimeError):
        wrap_transformer(tiny_pipe.transformer, ctx)
