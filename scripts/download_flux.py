"""Selective FLUX.1-dev download into $HF_HOME (about 34 GB): only the diffusers folders E1 needs.

Skips the single-file checkpoints at the repository root (flux1-dev.safetensors, ae.safetensors).
Equivalent to `hf download black-forest-labs/FLUX.1-dev --include <pattern> --include <pattern> ...`
(one --include per pattern). Accept the licence on the model page and run `hf auth login` first.
"""
import argparse
import sys

PATTERNS = ["model_index.json", "scheduler/*", "transformer/*", "vae/*", "text_encoder/*", "text_encoder_2/*",
            "tokenizer/*", "tokenizer_2/*"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--skip_text_encoder_2", action="store_true",
                   help="only if cache/prompt_embeds.pt already exists")
    p.add_argument("--dry_run", action="store_true", help="list matching files and sizes without downloading")
    args = p.parse_args(argv)
    from huggingface_hub import HfApi, constants, snapshot_download

    patterns = [x for x in PATTERNS if not (args.skip_text_encoder_2 and x.startswith("text_encoder_2"))]
    print(f"HF cache: {constants.HF_HUB_CACHE}")
    if args.dry_run:
        import fnmatch

        info = HfApi().model_info(args.repo, files_metadata=True)
        total = 0
        for s in info.siblings:
            if any(fnmatch.fnmatch(s.rfilename, pat) for pat in patterns):
                total += s.size or 0
                print(f"{(s.size or 0) / 1e9:8.3f} GB  {s.rfilename}")
        print(f"total {total / 1e9:.2f} GB")
        return 0
    path = snapshot_download(args.repo, allow_patterns=patterns)
    print(f"downloaded to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
