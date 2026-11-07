"""E1.3: step-size-matched carry (SD) against the unscaled carry (SD_UNSCALED), b = 4, T in {28, 50}.

Writes results/e1_ablation.csv (appended, resumable) and results/e1_ablation_meta.json.
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import int_list

from sbb.experiments import run_ablation
from sbb.runtime import add_common_args, build_env


def main(argv=None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--steps", type=int_list, default=None, help="e.g. 28,50")
    p.add_argument("--bits", type=int_list, default=None)
    p.add_argument("--lpips", choices=["on", "off"], default=None)
    args = p.parse_args(argv)
    env = build_env(args, "ablation", "run_e1_ablation")
    sec = env.cfg["ablation"]
    for k in ("steps", "bits"):
        if getattr(args, k) is not None:
            sec[k] = getattr(args, k)
    if args.lpips is not None:
        sec["lpips"] = args.lpips == "on"
    out = run_ablation(env)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
