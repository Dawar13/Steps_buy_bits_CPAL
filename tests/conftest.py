import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def tiny_pipe():
    from sbb.flux import tiny_pipeline

    return tiny_pipeline(seed=0)


def transformer_inputs(n_img: int = 16, n_txt: int = 8, seed: int = 0, sigma: float = 0.7):
    """Direct-call inputs for the tiny FluxTransformer2DModel."""
    from sbb.flux import TINY_TRANSFORMER

    g = torch.Generator().manual_seed(seed)
    d = TINY_TRANSFORMER["joint_attention_dim"]
    return dict(hidden_states=torch.randn(1, n_img, TINY_TRANSFORMER["in_channels"], generator=g),
                encoder_hidden_states=torch.randn(1, n_txt, d, generator=g),
                pooled_projections=torch.randn(1, TINY_TRANSFORMER["pooled_projection_dim"], generator=g),
                timestep=torch.tensor([sigma]), guidance=torch.tensor([3.5]),
                img_ids=torch.zeros(n_img, 3), txt_ids=torch.zeros(n_txt, 3), return_dict=False)


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
