"""Find where a run first produces a non-finite value (used for the two SD runs of E1.2 that ended in NaN).

Reruns one prompt with forward hooks on every quantised linear and records, per step, the largest |input|
and the first layer whose input or output is not finite. Writes results/diag_nan_p<id>_T<T>_<mode>_b<bits>.json.
Example: python scripts/diagnose_nan.py --prompt_ids 9 --steps 50 --mode SD --bits 4
"""
import argparse
import sys

import _bootstrap  # noqa: F401
import torch

from sbb.flux import generate
from sbb.runtime import add_common_args, build_env, now, write_json
from sbb.wrap import quant_layers


def main(argv=None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--mode", default="SD")
    p.add_argument("--bits", type=int, default=4)
    args = p.parse_args(argv)
    env = build_env(args, "law", "diagnose_nan")
    item = env.prompts[0]
    h, w = env.hw
    ctx = env.ctx
    first_bad: dict = {}
    max_abs: dict[int, float] = {}

    def hook(mod, inputs, output):
        x = inputs[0]
        step = ctx.step
        m = float(x.detach().abs().float().nan_to_num(posinf=float("inf")).max())
        max_abs[step] = max(max_abs.get(step, 0.0), m)
        if not first_bad:
            if not torch.isfinite(x).all():
                first_bad.update(step=step, layer=mod.name, where="input", max_abs_input=m)
            elif not torch.isfinite(output).all():
                first_bad.update(step=step, layer=mod.name, where="output", max_abs_input=m,
                                 max_abs_weight=float(mod.weight.abs().max()))

    handles = [m.register_forward_hook(hook) for m in quant_layers(env.pipe.transformer)]
    ctx.configure(mode=args.mode, bits=args.bits)
    res = generate(env.pipe, ctx, item, args.steps, h, w, float(env.cfg["guidance"]))
    for hd in handles:
        hd.remove()
    ctx.configure(mode="FP")
    out = {"prompt_id": item.pid, "T": args.steps, "mode": args.mode, "bits": args.bits,
           "latent_finite": bool(torch.isfinite(res.latent).all()), "first_non_finite": first_bad or None,
           "max_abs_input_per_step": [max_abs.get(i) for i in range(args.steps)], "git": env.meta["git"],
           "finished": now()}
    path = env.out_dir / f"diag_nan_p{item.pid:03d}_T{args.steps}_{args.mode}_b{args.bits}.json"
    write_json(path, out)
    print(out["latent_finite"], out["first_non_finite"])
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
