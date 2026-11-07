"""E1.1: temporal correlation of plain-rounding leftovers (RTN, b = 4, T = 50, 8 prompts by default).

Writes results/raw/corr/p<id>.npz per prompt (resume), results/e1_corr.json and plots/corr_heatmap.png.
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import int_list

from sbb.experiments import run_corr
from sbb.runtime import add_common_args, build_env


def main(argv=None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--bits", type=int, default=None)
    p.add_argument("--n_entries", type=int, default=None)
    p.add_argument("--lags", type=int_list, default=None, help="e.g. 1,2,5,10")
    args = p.parse_args(argv)
    env = build_env(args, "corr", "run_e1_corr")
    sec = env.cfg["corr"]
    for k in ("steps", "bits", "n_entries", "lags"):
        if getattr(args, k) is not None:
            sec[k] = getattr(args, k)
    out = run_corr(env)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
