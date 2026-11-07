"""E1.2: error versus steps. Modes RTN, SR, SD against FP at the same T, for each bit width.

Writes results/e1_law.csv (one row per prompt, T, mode, bits; appended as it goes, so a rerun resumes),
results/e1_law_meta.json and plots/law_b<bits>.png.
Example: python scripts/run_e1_law.py --n_prompts 16   (then rerun with --n_prompts 32 to add the rest)
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import int_list, str_list

from sbb.experiments import run_law
from sbb.runtime import add_common_args, build_env


def main(argv=None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--steps", type=int_list, default=None, help="e.g. 8,16,28,50,100")
    p.add_argument("--bits", type=int_list, default=None, help="e.g. 4,3")
    p.add_argument("--modes", type=str_list, default=None, help="e.g. RTN,SR,SD")
    p.add_argument("--lpips", choices=["on", "off"], default=None)
    args = p.parse_args(argv)
    env = build_env(args, "law", "run_e1_law")
    sec = env.cfg["law"]
    for k in ("steps", "bits", "modes"):
        if getattr(args, k) is not None:
            sec[k] = getattr(args, k)
    if args.lpips is not None:
        sec["lpips"] = args.lpips == "on"
    out = run_law(env)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
