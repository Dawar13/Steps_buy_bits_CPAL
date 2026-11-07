"""Sample the E1 prompts from PartiPrompts with a fixed seed and write prompts.txt (commit it).

The dataset id `nateraw/parti-prompts` (column `Prompt`) is the expected Hugging Face copy of
PartiPrompts (verify on the VM before use). Runs on the VM only; `--tiny` uses a built-in list.
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401
import numpy as np

from sbb.runtime import git_info, load_config, now, resolve_path

TINY_POOL = [f"a tiny test prompt number {i}" for i in range(40)]


def sample(pool: list[str], n: int, seed: int) -> tuple[list[int], list[str]]:
    if n > len(pool):
        raise ValueError(f"asked for {n} prompts from a pool of {len(pool)}")
    idx = np.random.default_rng(seed).choice(len(pool), size=n, replace=False)
    idx = [int(i) for i in idx]
    return idx, [" ".join(str(pool[i]).split()) for i in idx]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default="nateraw/parti-prompts", help="(verify on the VM)")
    p.add_argument("--split", default="train")
    p.add_argument("--column", default="Prompt")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default: prompts.txt (prompts_tiny.txt with --tiny)")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.tiny)
    out = resolve_path(args.out or cfg["prompts"])

    if args.tiny:
        pool, source = TINY_POOL, {"dataset": "built-in tiny list"}
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, split=args.split)
        if args.column not in ds.column_names:
            raise KeyError(f"column {args.column!r} not in {ds.column_names}")
        pool = list(ds[args.column])
        source = {"dataset": args.dataset, "split": args.split, "column": args.column, "pool_size": len(pool),
                  "fingerprint": getattr(ds, "_fingerprint", None)}
    idx, prompts = sample(pool, args.n, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    meta = {"source": source, "seed": args.seed, "n": args.n, "indices": idx, "created": now(), "git": git_info()}
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(prompts)} prompts to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
