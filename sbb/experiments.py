"""E1 experiments. Every function writes its results incrementally so an interrupted job can resume."""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

import numpy as np
import torch

from .corr import CorrRecorder, summarize
from .flux import decode, generate
from .metrics import LPIPSMetric, psnr, rel_err_sq
from .runtime import Env, append_row, done_keys, now, read_json, write_json
from .wrap import buffer_nbytes, quant_states

log = logging.getLogger("sbb.exp")

GRID_FIELDS = ["experiment", "prompt_id", "seed", "T", "mode", "bits", "storage", "group_size", "share_inputs",
               "rel_err_sq", "rel_err", "psnr", "lpips", "seconds", "height", "width", "guidance", "model_id",
               "tiny", "git_commit", "git_dirty", "timestamp"]
GRID_KEY = ["prompt_id", "T", "mode", "bits", "storage"]


def storage_label(mode: str, storage: str) -> str:
    return storage if mode in ("SD", "SD_UNSCALED") else "none"


# --------------------------------------------------------------------------- FP references
def fp_reference(env: Env, item, num_steps: int) -> torch.Tensor:
    """Full-precision latent for (prompt, T), cached on disk and checked against the run settings."""
    h, w = env.hw
    g = float(env.cfg["guidance"])
    meta = {"model_id": env.cfg["model_id"], "seed": item.seed, "prompt_id": item.pid, "T": num_steps,
            "height": h, "width": w, "guidance": g, "tiny": env.tiny}
    path = env.cache_dir / "fp" / f"p{item.pid:03d}_T{num_steps}_{h}x{w}_g{g:g}.pt"
    if path.exists():
        d = torch.load(path, map_location="cpu", weights_only=False)
        if d.get("meta") == meta:
            return d["latent"]
        log.warning("FP cache %s does not match the run settings; recomputing", path.name)
    env.ctx.configure(mode="FP")
    res = generate(env.pipe, env.ctx, item, num_steps, h, w, g)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latent": res.latent, "meta": meta, "sigmas": res.sigmas, "git": env.meta["git"]}, path)
    return res.latent


def _lpips(env: Env, enabled: bool):
    if not enabled:
        return None
    return LPIPSMetric(env.device)


# --------------------------------------------------------------------------- E1.2 / E1.3 grid
def run_grid(env: Env, experiment: str, modes: list[str], bits_list: list[int], steps: list[int],
             out_csv: Path, lpips_on: bool):
    import pandas as pd

    done = done_keys(out_csv, GRID_KEY)
    lp = _lpips(env, lpips_on)
    h, w = env.hw
    g = float(env.cfg["guidance"])
    storage = env.cfg["storage"]
    total = len(env.prompts) * len(steps) * len(modes) * len(bits_list)
    count = 0
    for item in env.prompts:
        for T in steps:
            todo = [(b, m) for b in bits_list for m in modes
                    if (str(item.pid), str(T), m, str(b), storage_label(m, storage)) not in done]
            count += len(bits_list) * len(modes) - len(todo)
            if not todo:
                continue
            z_fp = fp_reference(env, item, T)
            img_fp = decode(env.pipe, z_fp, h, w)
            for bits, mode in todo:
                env.ctx.configure(mode=mode, bits=bits, storage=storage)
                res = generate(env.pipe, env.ctx, item, T, h, w, g)
                img = decode(env.pipe, res.latent, h, w)
                e2 = rel_err_sq(res.latent, z_fp)
                row = {"experiment": experiment, "prompt_id": item.pid, "seed": item.seed, "T": T, "mode": mode,
                       "bits": bits, "storage": storage_label(mode, storage), "group_size": env.ctx.group_size,
                       "share_inputs": env.cfg["share_inputs"], "rel_err_sq": e2, "rel_err": float(np.sqrt(e2)),
                       "psnr": psnr(img, img_fp), "lpips": lp(img, img_fp) if lp is not None else float("nan"),
                       "seconds": res.seconds, "height": h, "width": w, "guidance": g,
                       "model_id": env.cfg["model_id"], "tiny": env.tiny, "git_commit": env.meta["git"]["commit"],
                       "git_dirty": env.meta["git"]["dirty"], "timestamp": now()}
                append_row(out_csv, row, GRID_FIELDS)
                count += 1
                log.info("[%s %d/%d] prompt %d T=%d %s b=%d: rel_err=%.4g psnr=%.2f lpips=%.4g (%.1fs)",
                         experiment, count, total, item.pid, T, mode, bits, row["rel_err"], row["psnr"],
                         row["lpips"], res.seconds)
    env.ctx.configure(mode="FP")
    return pd.read_csv(out_csv)


def run_law(env: Env) -> Path:
    sec = env.cfg["law"]
    out_csv = env.out_dir / "e1_law.csv"
    write_json(env.out_dir / "e1_law_meta.json", env.meta)
    df = run_grid(env, "E1.2", list(sec["modes"]), [int(b) for b in sec["bits"]], [int(t) for t in sec["steps"]],
                  out_csv, bool(sec.get("lpips", True)))
    from .report import plot_law

    for bits in sorted({int(b) for b in sec["bits"]}):
        plot_law(df, bits, env.plot_dir / f"law_b{bits}.png", t_min=float(env.cfg["fit"]["t_min"]),
                 tiny=env.tiny)
    return out_csv


def run_ablation(env: Env) -> Path:
    sec = env.cfg["ablation"]
    out_csv = env.out_dir / "e1_ablation.csv"
    write_json(env.out_dir / "e1_ablation_meta.json", env.meta)
    run_grid(env, "E1.3", list(sec["modes"]), [int(b) for b in sec["bits"]], [int(t) for t in sec["steps"]],
             out_csv, bool(sec.get("lpips", True)))
    return out_csv


# --------------------------------------------------------------------------- E1.1 correlation
def run_corr(env: Env) -> Path:
    sec = env.cfg["corr"]
    h, w = env.hw
    g = float(env.cfg["guidance"])
    T, bits, mode = int(sec["steps"]), int(sec["bits"]), sec["mode"]
    rec = CorrRecorder(n_entries=int(sec["n_entries"]), lags=sec["lags"], seed=int(sec["seed"]))
    raw_dir = env.out_dir / "raw" / "corr"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {"T": T, "bits": bits, "mode": mode, "n_entries": int(sec["n_entries"]), "lags": list(sec["lags"]),
                "seed": int(sec["seed"]), "height": h, "width": w, "guidance": g, "model_id": env.cfg["model_id"],
                "group_size": env.ctx.group_size, "share_inputs": env.cfg["share_inputs"], "tiny": env.tiny}
    images = []
    env.ctx.recorder = rec
    try:
        for item in env.prompts:
            f = raw_dir / f"p{item.pid:03d}.npz"
            if f.exists():
                d = np.load(f, allow_pickle=False)
                if json.loads(str(d["meta"])) == run_meta:
                    images.append({"keys": [str(k) for k in d["keys"]], "cos": d["cos"], "drift": d["drift"],
                                   "lags": [int(l) for l in d["lags"]]})
                    log.info("[E1.1] prompt %d loaded from %s", item.pid, f.name)
                    continue
            env.ctx.configure(mode=mode, bits=bits)
            res = generate(env.pipe, env.ctx, item, T, h, w, g)
            im = rec.end_image()
            np.savez(f, keys=np.array(im["keys"]), cos=im["cos"], drift=im["drift"], lags=np.array(im["lags"]),
                     meta=np.array(json.dumps(run_meta)))
            images.append(im)
            l1 = im["cos"][..., 0]
            log.info("[E1.1] prompt %d: median lag-%d cosine %.3f (%.1fs)", item.pid, im["lags"][0],
                     float(np.nanmedian(l1)) if np.isfinite(l1).any() else float("nan"), res.seconds)
    finally:
        env.ctx.recorder = None
        env.ctx.configure(mode="FP")
    summary = summarize(images)
    summary.update({"settings": run_meta, "prompt_ids": [it.pid for it in env.prompts], "git": env.meta["git"],
                    "env": env.meta["env"], "finished": now()})
    if env.tiny:
        summary["note"] = env.meta["note"]
    out = env.out_dir / "e1_corr.json"
    write_json(out, summary)
    from .report import plot_corr_heatmap

    plot_corr_heatmap(summary, env.plot_dir / "corr_heatmap.png", tiny=env.tiny)
    return out


# --------------------------------------------------------------------------- E1.4 overhead
def _peak_reset(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak(device: str) -> int | None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    return None


BYTES_PER_ENTRY = {"fp32": 4.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}  # plus one bf16 scale per group for int


def run_overhead(env: Env) -> Path:
    sec = env.cfg["overhead"]
    h, w = env.hw
    g = float(env.cfg["guidance"])
    T, bits = int(sec["steps"]), int(sec["bits"])
    out = env.out_dir / "e1_overhead.json"
    settings = {"T": T, "bits": bits, "height": h, "width": w, "guidance": g, "model_id": env.cfg["model_id"],
                "group_size": env.ctx.group_size, "share_inputs": env.cfg["share_inputs"],
                "prompt_ids": [it.pid for it in env.prompts], "tiny": env.tiny}
    state = read_json(out) if out.exists() else {}
    if state.get("settings") != settings:
        state = {"settings": settings, "runs": {}}
    state.update({"git": env.meta["git"], "env": env.meta["env"]})
    if env.tiny:
        state["note"] = env.meta["note"]

    refs = {it.pid: fp_reference(env, it, T) for it in env.prompts}
    # Warm-up so that the first timed run does not pay for kernel selection and allocator growth.
    env.ctx.configure(mode="RTN", bits=bits)
    generate(env.pipe, env.ctx, env.prompts[0], 2, h, w, g)

    entries = [(m, None) for m in sec.get("timing_modes", ["FP", "RTN"]) if m not in ("SD", "SD_UNSCALED")]
    entries += [("SD", s) for s in sec["storages"]]
    for mode, storage in entries:
        name = mode if storage is None else f"{mode}_{storage}"
        if name in state["runs"]:
            log.info("[E1.4] %s already done", name)
            continue
        per = []
        for item in env.prompts:
            env.ctx.configure(mode=mode, bits=bits, storage=storage or env.cfg["storage"])
            _peak_reset(env.device)
            res = generate(env.pipe, env.ctx, item, T, h, w, g, timing=True)
            peak = _peak(env.device)
            e2 = rel_err_sq(res.latent, refs[item.pid])
            states = quant_states(env.pipe.transformer)
            per.append({"prompt_id": item.pid, "rel_err_sq": e2, "rel_err": float(np.sqrt(e2)), "peak_bytes": peak,
                        "buffer_bytes": buffer_nbytes(env.pipe.transformer),
                        "buffer_bytes_if_unshared": sum(st.nbytes() * st.n_members for st in states),
                        "buffer_entries": sum(st.store.data.numel() * (2 if st.store.storage == "int4" else 1)
                                              for st in states if st.store is not None and st.store.data is not None),
                        "step_seconds_median": statistics.median(res.step_seconds[1:] or res.step_seconds),
                        "seconds": res.seconds})
        entry = {"mode": mode, "storage": storage, "per_prompt": per,
                 "rel_err_sq_median": statistics.median(p["rel_err_sq"] for p in per),
                 "rel_err_median": statistics.median(p["rel_err"] for p in per),
                 "step_seconds_median": statistics.median(p["step_seconds_median"] for p in per),
                 "peak_bytes_max": max(p["peak_bytes"] for p in per) if per[0]["peak_bytes"] is not None else None,
                 "buffer_bytes": max(p["buffer_bytes"] for p in per),
                 "buffer_bytes_if_unshared": max(p["buffer_bytes_if_unshared"] for p in per),
                 "buffer_entries": max(p["buffer_entries"] for p in per)}
        state["runs"][name] = entry
        write_json(out, state)
        log.info("[E1.4] %s: rel_err %.4g, step %.4fs, buffers %.3f GB, peak %s", name, entry["rel_err_median"],
                 entry["step_seconds_median"], entry["buffer_bytes"] / 1e9,
                 f"{entry['peak_bytes_max'] / 1e9:.2f} GB" if entry["peak_bytes_max"] else "n/a (cpu)")
    env.ctx.configure(mode="FP")

    runs = state["runs"]
    rtn = runs.get("RTN")
    derived = {}
    for name, e in runs.items():
        d = {}
        if rtn is not None:
            if e.get("peak_bytes_max") is not None and rtn.get("peak_bytes_max") is not None:
                d["extra_peak_bytes_vs_RTN"] = e["peak_bytes_max"] - rtn["peak_bytes_max"]
            d["step_time_ratio_vs_RTN"] = e["step_seconds_median"] / rtn["step_seconds_median"]
            if e["mode"] == "SD":
                d["rel_err_ratio_vs_RTN"] = e["rel_err_median"] / rtn["rel_err_median"]
        derived[name] = d
    state["derived"] = derived
    state["finished"] = now()
    write_json(out, state)
    return out
