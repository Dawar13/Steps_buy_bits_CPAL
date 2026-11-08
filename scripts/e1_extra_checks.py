"""Extra E1 checks beyond the decision table; writes results/e1_checks.json.

1. Reproducibility: E1.4's SD fp32 runs must equal E1.2's SD rows for the same prompts (same computation,
   run alone versus next to a second worker).
2. Per-prompt view of SD against RTN and SR at the decision T and bit width.
3. How many prompts have an error that rises from T = 16 to the largest T, per mode and bit width.
4. Per-step lag-1 leftover cosine and drift ratio (E1.1), to locate steps where the activations jump.
"""
import argparse
import glob
import sys

import _bootstrap  # noqa: F401
import numpy as np

from sbb.runtime import git_info, load_config, now, read_json, resolve_path, write_json


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args(argv)
    import pandas as pd

    cfg = load_config(args.config, args.tiny)
    out_dir = resolve_path(args.out_dir or cfg["out_dir"])
    bits, T = int(cfg["decision"]["bits"]), int(cfg["decision"]["ratio_T"])
    law = pd.read_csv(out_dir / "e1_law.csv")
    out: dict = {"created": now(), "git": git_info(), "bits": bits, "T": T}

    ov_path = out_dir / "e1_overhead.json"
    if ov_path.exists():
        ov = read_json(ov_path)
        sd = law[(law["mode"] == "SD") & (law["bits"] == bits) & (law["T"] == ov["settings"]["T"])]
        sd = sd.set_index("prompt_id")["rel_err"]
        pairs = [{"prompt_id": q["prompt_id"], "e1_4": q["rel_err"], "e1_2": float(sd.get(q["prompt_id"], np.nan))}
                 for q in ov["runs"].get("SD_fp32", {}).get("per_prompt", [])]
        out["reproducibility_sd_fp32"] = {"pairs": pairs, "max_abs_diff": max(
            (abs(x["e1_4"] - x["e1_2"]) for x in pairs), default=None)}

    def at(mode):
        return law[(law["mode"] == mode) & (law["bits"] == bits) & (law["T"] == T)].set_index("prompt_id")["rel_err"]

    sd, rtn, sr = at("SD"), at("RTN"), at("SR")
    r = (sd / rtn).dropna()
    out["sd_vs_rtn"] = {"median": float(r.median()), "min": float(r.min()), "max": float(r.max()),
                        "n_sd_better": int((r < 1).sum()), "n": int(r.size)}
    r2 = (sd / sr).dropna()
    out["sd_vs_sr"] = {"median": float(r2.median()), "n_sd_better": int((r2 < 1).sum()), "n": int(r2.size)}

    rises = {}
    t_hi = int(law["T"].max())
    for (mode, b), g in law.groupby(["mode", "bits"]):
        piv = g.pivot(index="prompt_id", columns="T", values="rel_err")
        if 16 in piv.columns and t_hi in piv.columns:
            rises[f"{mode}_b{b}"] = {"n_rising": int((piv[t_hi] > piv[16]).sum()), "n": int(len(piv)),
                                     "median_ratio_Tmax_over_T16": float((piv[t_hi] / piv[16]).median())}
    out["error_rises_T16_to_Tmax"] = {"T_max": t_hi, "by_mode_bits": rises}

    files = sorted(glob.glob(str(out_dir / "raw" / "corr" / "p*.npz")))
    if files:
        cos = np.stack([np.load(f)["cos"][..., 0] for f in files])  # (N, L, T)
        drift = np.stack([np.load(f)["drift"] for f in files])
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                c_step = np.nanmedian(cos, axis=(0, 1))
                d_step = np.nanmedian(drift, axis=(0, 1))
                per_prompt_min = [int(np.nanargmin(np.nanmedian(cos[n], axis=0))) for n in range(cos.shape[0])]
        out["corr_per_step"] = {"lag1_median": [None if not np.isfinite(v) else float(v) for v in c_step],
                                "drift_median": [None if not np.isfinite(v) else float(v) for v in d_step],
                                "step_of_lowest_lag1_per_prompt": per_prompt_min}
    path = out_dir / "e1_checks.json"
    write_json(path, out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
