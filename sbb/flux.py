"""Loading the FLUX pipeline (real or tiny), running one image with the step context, decoding latents.

The tiny pipeline is the stock diffusers FluxPipeline with a tiny random transformer, a tiny random
VAE and a FLUX-style scheduler, so `--tiny` exercises the same pipeline code and callback path as
the VM, on CPU and without any download.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .context import StepCallback, StepContext
from .wrap import reset_quant_state

FLUX_ID = "black-forest-labs/FLUX.1-dev"

# Checked against diffusers 0.42.1 (sum(axes_dims_rope) must equal attention_head_dim).
TINY_TRANSFORMER = dict(patch_size=1, in_channels=4, num_layers=1, num_single_layers=1, attention_head_dim=16,
                        num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=32,
                        axes_dims_rope=[4, 4, 8], guidance_embeds=True)
TINY_VAE = dict(in_channels=3, out_channels=3, down_block_types=["DownEncoderBlock2D"] * 2,
                up_block_types=["UpDecoderBlock2D"] * 2, block_out_channels=(8, 16), layers_per_block=1,
                latent_channels=1, norm_num_groups=4, scaling_factor=1.0, shift_factor=0.0,
                use_quant_conv=False, use_post_quant_conv=False)
# Values of the FLUX.1-dev scheduler config as recalled (verify against scheduler/scheduler_config.json on the VM).
TINY_SCHEDULER = dict(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=4096, max_shift=1.15,
                      num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True)
TINY_TEXT_LEN = 8


def tiny_pipeline(seed: int = 0, dtype: torch.dtype = torch.float32):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel

    torch.manual_seed(seed)
    transformer = FluxTransformer2DModel(**TINY_TRANSFORMER).to(dtype)
    torch.manual_seed(seed + 1)
    vae = AutoencoderKL(**TINY_VAE).to(dtype)
    scheduler = FlowMatchEulerDiscreteScheduler(**TINY_SCHEDULER)
    pipe = FluxPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, text_encoder_2=None,
                        tokenizer_2=None, transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_pipeline(model_id: str = FLUX_ID, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                  tiny: bool = False):
    """Real FLUX without the text encoders (prompt embeddings are precomputed), or the tiny pipeline."""
    if tiny:
        return tiny_pipeline(dtype=torch.float32).to(device)
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(model_id, text_encoder=None, text_encoder_2=None, tokenizer=None,
                                        tokenizer_2=None, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


@dataclass
class PromptItem:
    pid: int
    seed: int
    text: str
    prompt_embeds: torch.Tensor  # (1, L, D)
    pooled_prompt_embeds: torch.Tensor  # (1, D)


def tiny_embeddings(n: int, seed: int = 1234) -> dict:
    g = torch.Generator().manual_seed(seed)
    d = TINY_TRANSFORMER["joint_attention_dim"]
    return {"prompts": [f"tiny prompt {i}" for i in range(n)],
            "prompt_embeds": torch.randn(n, TINY_TEXT_LEN, d, generator=g),
            "pooled_prompt_embeds": torch.randn(n, TINY_TRANSFORMER["pooled_projection_dim"], generator=g),
            "model_id": "tiny", "max_sequence_length": TINY_TEXT_LEN}


def select_prompts(emb: dict, ids: list[int]) -> list[PromptItem]:
    n = emb["prompt_embeds"].shape[0]
    out = []
    for pid in ids:
        if not 0 <= pid < n:
            raise IndexError(f"prompt id {pid} out of range (cache holds {n} prompts)")
        out.append(PromptItem(pid=pid, seed=pid, text=emb["prompts"][pid],
                              prompt_embeds=emb["prompt_embeds"][pid: pid + 1],
                              pooled_prompt_embeds=emb["pooled_prompt_embeds"][pid: pid + 1]))
    return out


@dataclass
class GenResult:
    latent: torch.Tensor  # packed latent as returned by the pipeline, on CPU, float32
    seconds: float
    step_seconds: list[float]
    sigmas: list[float] | None
    gammas: list[float] | None


def generate(pipe, ctx: StepContext, item: PromptItem, num_steps: int, height: int, width: int,
             guidance: float, timing: bool = False) -> GenResult:
    """One image with the stock pipeline. Seed = prompt index; quantiser state is reset first."""
    ctx.begin_image(seed=item.seed, num_steps=num_steps)
    reset_quant_state(pipe.transformer)
    cb = StepCallback(ctx, timing=timing)
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    gen = torch.Generator("cpu").manual_seed(item.seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if timing:
        cb.times.append(t0)
    with torch.no_grad():
        out = pipe(prompt_embeds=item.prompt_embeds.to(device, dtype),
                   pooled_prompt_embeds=item.pooled_prompt_embeds.to(device, dtype),
                   height=height, width=width, num_inference_steps=num_steps, guidance_scale=guidance,
                   generator=gen, output_type="latent", callback_on_step_end=cb, return_dict=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    if cb.calls != num_steps:
        raise RuntimeError(f"callback ran {cb.calls} times for {num_steps} steps")
    step_seconds = [b - a for a, b in zip(cb.times[:-1], cb.times[1:])] if timing else []
    return GenResult(latent=out.images.detach().float().cpu(), seconds=seconds, step_seconds=step_seconds,
                     sigmas=cb.sigmas, gammas=list(ctx.gammas) if ctx.gammas is not None else None)


def generate_stock(pipe, item: PromptItem, num_steps: int, height: int, width: int, guidance: float) -> torch.Tensor:
    """The untouched pipeline (no callback), for the B1 sanity check."""
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    gen = torch.Generator("cpu").manual_seed(item.seed)
    with torch.no_grad():
        out = pipe(prompt_embeds=item.prompt_embeds.to(device, dtype),
                   pooled_prompt_embeds=item.pooled_prompt_embeds.to(device, dtype),
                   height=height, width=width, num_inference_steps=num_steps, guidance_scale=guidance,
                   generator=gen, output_type="latent", return_dict=True)
    return out.images.detach().float().cpu()


@torch.no_grad()
def decode(pipe, latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Packed latent -> image in [0, 1], shape (B, 3, H, W), float32 on the VAE device."""
    vae = pipe.vae
    device = next(vae.parameters()).device
    lat = pipe._unpack_latents(latent.to(device), height, width, pipe.vae_scale_factor)
    lat = lat / vae.config.scaling_factor + vae.config.shift_factor
    img = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
    return (img.float() / 2 + 0.5).clamp(0, 1)
