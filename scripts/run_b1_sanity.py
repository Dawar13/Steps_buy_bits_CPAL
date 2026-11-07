"""B1 sanity checks on the real model (or the tiny one).

1. FP mode with the wrapper and the step callback reproduces the stock FluxPipeline latent for the same
   seed (relative error below 1e-6; normally exactly equal).
2. SD with gamma fixed to 0 equals RTN on one prompt.
3. SD with the step-ratio gamma differs from RTN (the carry is active).
Writes results/b1_sanity.json; exits with status 1 if a check fails.
"""
import argparse
import sys

import _bootstrap  # noqa: F401
import torch

from sbb.flux import generate, generate_stock
from sbb.metrics import rel_err_sq
from sbb.runtime import add_common_args, build_env, now, write_json
from sbb.wrap import unwrap_transformer, wrap_transformer


def main(argv=None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--steps", type=int, default=None, help="default 28 (4 with --tiny)")
    p.add_argument("--bits", type=int, default=4)
    args = p.parse_args(argv)
    if args.prompt_ids is None and args.n_prompts is None:
        args.prompt_ids = "0"
    env = build_env(args, "law", "run_b1_sanity")
    T = args.steps or (4 if env.tiny else 28)
    h, w = env.hw
    g = float(env.cfg["guidance"])
    item = env.prompts[0]

    unwrap_transformer(env.pipe.transformer)
    z_stock = generate_stock(env.pipe, item, T, h, w, g)
    wrap_transformer(env.pipe.transformer, env.ctx, share_inputs=bool(env.cfg["share_inputs"]))

    env.ctx.configure(mode="FP")
    z_fp = generate(env.pipe, env.ctx, item, T, h, w, g).latent
    env.ctx.configure(mode="RTN", bits=args.bits)
    z_rtn = generate(env.pipe, env.ctx, item, T, h, w, g).latent
    env.ctx.configure(mode="SD", bits=args.bits)
    env.ctx.gamma_override = 0.0
    z_sd0 = generate(env.pipe, env.ctx, item, T, h, w, g).latent
    env.ctx.gamma_override = None
    sd = generate(env.pipe, env.ctx, item, T, h, w, g)
    env.ctx.configure(mode="FP")

    e_fp = rel_err_sq(z_fp, z_stock)
    e_sd0 = rel_err_sq(z_sd0, z_rtn)
    e_sd = rel_err_sq(sd.latent, z_rtn)
    checks = {
        "fp_matches_stock": {"rel_err_sq": e_fp, "exactly_equal": bool(torch.equal(z_fp, z_stock)),
                             "pass": e_fp < 1e-6},
        "sd_gamma0_matches_rtn": {"rel_err_sq": e_sd0, "exactly_equal": bool(torch.equal(z_sd0, z_rtn)),
                                  "pass": e_sd0 < 1e-6},
        "sd_differs_from_rtn": {"rel_err_sq": e_sd, "pass": e_sd > 0.0},
        "rtn_vs_fp_rel_err_sq": rel_err_sq(z_rtn, z_fp),
        "sd_vs_fp_rel_err_sq": rel_err_sq(sd.latent, z_fp),
    }
    ok = all(v["pass"] for v in checks.values() if isinstance(v, dict))
    out = env.out_dir / "b1_sanity.json"
    write_json(out, {"checks": checks, "all_pass": ok, "prompt_id": item.pid, "T": T, "bits": args.bits,
                     "sigmas": sd.sigmas, "gammas": sd.gammas, "git": env.meta["git"], "env": env.meta["env"],
                     "n_wrapped": len(env.wrapped), "finished": now(), "tiny": env.tiny})
    for k, v in checks.items():
        print(k, v)
    print(f"wrote {out}: {'ALL PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
