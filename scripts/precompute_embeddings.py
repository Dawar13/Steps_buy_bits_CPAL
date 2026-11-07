"""Encode the E1 prompts once with the FLUX text encoders and save cache/prompt_embeds.pt.

Loads the pipeline without the transformer and the VAE (falls back to a full load if that is not
supported), encodes every line of prompts.txt, saves prompt_embeds (N, L, 4096) and
pooled_prompt_embeds (N, 768) in BF16, then frees the memory. Afterwards the text_encoder_2 folder can
be removed from the Hugging Face cache (scripts/drop_text_encoder_2.py).
"""
import argparse
import gc
import logging
import sys

import _bootstrap  # noqa: F401
import torch

from sbb.flux import tiny_embeddings
from sbb.runtime import default_device, git_info, load_config, now, resolve_path, setup_torch

log = logging.getLogger("sbb.embeds")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--config", default=None)
    p.add_argument("--prompts", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--model_id", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max_sequence_length", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.tiny)
    prompts_path = resolve_path(args.prompts or cfg["prompts"])
    out = resolve_path(args.out or cfg["embeds"])
    model_id = args.model_id or cfg["model_id"]
    msl = int(args.max_sequence_length or cfg["max_sequence_length"])
    device = args.device or default_device(cfg["tiny"])
    setup_torch(device, cfg.get("mem_fraction"))

    if prompts_path.exists():
        prompts = [ln.strip() for ln in prompts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    elif cfg["tiny"]:
        prompts = [f"tiny prompt {i}" for i in range(4)]
    else:
        raise FileNotFoundError(f"{prompts_path} not found; run scripts/make_prompts.py first")

    if cfg["tiny"]:
        emb = tiny_embeddings(len(prompts))
        emb["prompts"] = prompts
    else:
        from diffusers import FluxPipeline

        try:
            pipe = FluxPipeline.from_pretrained(model_id, transformer=None, vae=None, torch_dtype=torch.bfloat16)
        except Exception as e:  # older diffusers may not accept None components
            log.warning("loading without transformer/VAE failed (%s); loading the full pipeline", e)
            pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            pipe.transformer = None
            pipe.vae = None
            gc.collect()
        pipe.text_encoder.to(device)
        pipe.text_encoder_2.to(device)
        pe_list, pp_list = [], []
        with torch.no_grad():
            for i, text in enumerate(prompts):
                pe, pp, _ = pipe.encode_prompt(prompt=text, prompt_2=None, device=torch.device(device),
                                               num_images_per_prompt=1, max_sequence_length=msl)
                pe_list.append(pe.to("cpu", torch.bfloat16))
                pp_list.append(pp.to("cpu", torch.bfloat16))
                log.info("encoded prompt %d/%d", i + 1, len(prompts))
        emb = {"prompts": prompts, "prompt_embeds": torch.cat(pe_list), "pooled_prompt_embeds": torch.cat(pp_list),
               "model_id": model_id, "max_sequence_length": msl}
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    emb.update({"created": now(), "git": git_info()})
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, out)
    print(f"wrote {out}: prompt_embeds {tuple(emb['prompt_embeds'].shape)}, "
          f"pooled {tuple(emb['pooled_prompt_embeds'].shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
