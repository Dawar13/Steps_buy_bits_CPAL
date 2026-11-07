"""Replace the linears inside the FLUX transformer blocks by QuantLinear.

Wrapped: every nn.Linear inside `transformer_blocks` and `single_transformer_blocks` whose qualified
name does not contain "norm" (this excludes the adaptive-norm modulation linears norm1.linear,
norm1_context.linear and norm.linear). Embedders, time/text embeddings, norm_out and the final
proj_out stay in full precision because they sit outside the block lists.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable

from torch import nn

from .context import StepContext
from .quant import QuantLinear, QuantState

log = logging.getLogger(__name__)

BLOCK_LISTS = ("transformer_blocks", "single_transformer_blocks")
QKV = ("to_q", "to_k", "to_v")
ADD_QKV = ("add_q_proj", "add_k_proj", "add_v_proj")


def share_key(full_name: str) -> str:
    """Name of the input a linear reads; linears with the same key read the same tensor object.

    Double blocks: attn.to_q/k/v read norm_hidden_states; attn.add_q/k/v_proj read norm_encoder_hidden_states.
    Single blocks: attn.to_q/k/v and proj_mlp all read norm_hidden_states.
    The runtime check in QuantLinear.forward raises if this grouping is ever wrong.
    """
    parts = full_name.split(".")
    parent, child = ".".join(parts[:-1]), parts[-1]
    if child in QKV:
        return f"{parent}.qkv_in"
    if child in ADD_QKV:
        return f"{parent}.add_qkv_in"
    if parts[0] == "single_transformer_blocks" and len(parts) == 3 and child == "proj_mlp":
        return f"{parent}.attn.qkv_in"
    return full_name


def find_targets(transformer: nn.Module, skip_substring: str = "norm",
                 block_lists: Iterable[str] = BLOCK_LISTS) -> list[tuple[nn.Module, str, str, nn.Linear]]:
    targets = []
    for list_name in block_lists:
        blocks = getattr(transformer, list_name, None)
        if blocks is None:
            raise AttributeError(f"transformer has no attribute {list_name!r}")
        for name, mod in blocks.named_modules(prefix=list_name):
            if isinstance(mod, QuantLinear):
                continue
            for child_name, child in mod.named_children():
                if isinstance(child, nn.Linear) and not isinstance(child, QuantLinear):
                    full = f"{name}.{child_name}"
                    if skip_substring and skip_substring in full:
                        continue
                    targets.append((mod, child_name, full, child))
    return targets


def wrap_transformer(transformer: nn.Module, ctx: StepContext, share_inputs: bool = True,
                     skip_substring: str = "norm", logger: Callable[[str], None] | None = None) -> list[str]:
    """Wrap in place. Returns the qualified names of the wrapped linears (also logged)."""
    if quant_layers(transformer):
        raise RuntimeError("transformer is already wrapped")
    targets = find_targets(transformer, skip_substring)
    states: dict[str, QuantState] = {}
    for parent, child_name, full, lin in targets:
        key = share_key(full) if share_inputs else full
        state = states.setdefault(key, QuantState(key))
        setattr(parent, child_name, QuantLinear(lin, ctx, full, state))
    names = [t[2] for t in targets]
    msg = (f"wrapped {len(names)} linears in {len(states)} input groups "
           f"(share_inputs={share_inputs}): " + ", ".join(names))
    (logger or log.info)(msg)
    return names


def unwrap_transformer(transformer: nn.Module) -> int:
    n = 0
    for mod in list(transformer.modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, QuantLinear):
                setattr(mod, child_name, child.linear)
                n += 1
    return n


def quant_layers(transformer: nn.Module) -> list[QuantLinear]:
    return [m for m in transformer.modules() if isinstance(m, QuantLinear)]


def quant_states(transformer: nn.Module) -> list[QuantState]:
    seen: dict[int, QuantState] = {}
    for m in quant_layers(transformer):
        seen.setdefault(id(m.state), m.state)
    return list(seen.values())


def reset_quant_state(transformer: nn.Module) -> None:
    """Clear every leftover buffer and cached input. Call at the start of every image."""
    for st in quant_states(transformer):
        st.reset()


def buffer_nbytes(transformer: nn.Module) -> int:
    return sum(st.nbytes() for st in quant_states(transformer))
