"""E1.4: leftover buffer memory (fp32, bf16, int8, int4 storage), its effect on error, and step time.

Writes results/e1_overhead.json (updated after each run, so a rerun skips finished entries).
Times are for fake quantisation in PyTorch and are for information only.
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import str_list

from sbb.experiments import run_overhead
from sbb.runtime import add_common_args, build_env


def main(argv=None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--storages", type=str_list, default=None, help="e.g. fp32,bf16,int8,int4")
    args = p.parse_args(argv)
    env = build_env(args, "overhead", "run_e1_overhead")
    sec = env.cfg["overhead"]
    if args.steps is not None:
        sec["steps"] = args.steps
    if args.storages is not None:
        sec["storages"] = args.storages
    out = run_overhead(env)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
