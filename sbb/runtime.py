"""Shared runner plumbing: config, command line, device set-up, logging, incremental result files."""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as _dt
import json
import logging
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "e1.yaml"
log = logging.getLogger("sbb")


# --------------------------------------------------------------------------- config
def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike | None = None, tiny: bool = False) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tiny_cfg = cfg.pop("tiny", {}) or {}
    if tiny:
        cfg = deep_merge(cfg, tiny_cfg)
    cfg["tiny"] = bool(tiny)
    return cfg


def resolve_path(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


# --------------------------------------------------------------------------- provenance
def git_info() -> dict:
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=REPO, capture_output=True, text=True, timeout=20)

    try:
        head = run("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"commit": "nogit", "dirty": None}
        dirty = run("status", "--porcelain", "--untracked-files=no")
        return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "nogit", "dirty": None}


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__}
    try:
        import diffusers

        info["diffusers"] = diffusers.__version__
    except ImportError:
        info["diffusers"] = None
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- files
def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "item") and getattr(x, "ndim", 1) == 0:
        return _jsonable(x.item())
    return x


def _lock(f, exclusive: bool = True) -> None:
    try:
        import fcntl  # POSIX only; the parallel workers run on Linux
    except ImportError:
        return
    fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)


def _unlock(f) -> None:
    try:
        import fcntl
    except ImportError:
        return
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_row(path: str | os.PathLike, row: dict, fields: list[str]) -> None:
    """Append one result row and flush, writing the header if the file is new.

    Takes an exclusive file lock, so several worker processes can append to the same CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        _lock(f)
        try:
            f.seek(0, os.SEEK_END)
            new = f.tell() == 0
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})
            f.flush()
            os.fsync(f.fileno())
        finally:
            _unlock(f)


def atomic_torch_save(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def done_keys(path: str | os.PathLike, key_fields: Iterable[str]) -> set[tuple]:
    path = Path(path)
    if not path.exists():
        return set()
    key_fields = list(key_fields)
    with open(path, "r", newline="", encoding="utf-8") as f:
        return {tuple(str(r[k]) for k in key_fields) for r in csv.DictReader(f)}


def parse_ids(spec: str) -> list[int]:
    """'0-15,20,22' -> [0, ..., 15, 20, 22]."""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# --------------------------------------------------------------------------- command line
def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--tiny", action="store_true", help="tiny random FLUX pipeline on CPU, no downloads")
    p.add_argument("--config", default=None, help="yaml config (default configs/e1.yaml)")
    p.add_argument("--device", default=None, help="default: cuda if available (cpu with --tiny)")
    p.add_argument("--model_id", default=None)
    p.add_argument("--embeds", default=None, help="precomputed prompt embeddings (.pt)")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--plot_dir", default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--prompt_ids", default=None, help="e.g. 0-15 or 0,3,7 (overrides --n_prompts)")
    p.add_argument("--n_prompts", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--storage", default=None, choices=["fp32", "bf16", "int8", "int4"])
    p.add_argument("--no_share", action="store_true", help="give to_q/k/v and proj_mlp separate quantisers")
    p.add_argument("--mem_fraction", type=float, default=None)
    return p


def apply_overrides(cfg: dict, args: argparse.Namespace, section: str | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for k in ("model_id", "embeds", "out_dir", "plot_dir", "cache_dir", "height", "width", "guidance", "storage",
              "mem_fraction"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if getattr(args, "no_share", False):
        cfg["share_inputs"] = False
    if section is not None and getattr(args, "n_prompts", None) is not None:
        cfg[section]["n_prompts"] = args.n_prompts
    return cfg


def setup_torch(device: str, mem_fraction: float | None) -> None:
    """Cap GPU memory first (the H100 is shared), then make kernels as repeatable as is cheap."""
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if mem_fraction:
            torch.cuda.set_per_process_memory_fraction(float(mem_fraction), dev.index or 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def setup_logging(out_dir: Path, name: str) -> None:
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for h in list(root.handlers):
        if getattr(h, "_sbb", False):
            root.removeHandler(h)
            h.close()
    sh = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(out_dir / "logs" / f"{name}.log", encoding="utf-8")
    for h in (sh, fh):
        h.setFormatter(fmt)
        h._sbb = True
        root.addHandler(h)


# --------------------------------------------------------------------------- environment
@dataclass
class Env:
    cfg: dict
    device: str
    pipe: Any
    ctx: Any
    wrapped: list[str]
    prompts: list
    out_dir: Path
    plot_dir: Path
    cache_dir: Path
    meta: dict

    @property
    def tiny(self) -> bool:
        return bool(self.cfg.get("tiny"))

    @property
    def hw(self) -> tuple[int, int]:
        return int(self.cfg["height"]), int(self.cfg["width"])


def default_device(tiny: bool) -> str:
    if tiny:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def prompt_ids_for(cfg: dict, args: argparse.Namespace, section: str) -> list[int]:
    if getattr(args, "prompt_ids", None):
        return parse_ids(args.prompt_ids)
    return list(range(int(cfg[section]["n_prompts"])))


def load_embeddings(cfg: dict, n_needed: int) -> dict:
    from .flux import tiny_embeddings

    path = resolve_path(cfg["embeds"])
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    if cfg.get("tiny"):
        return tiny_embeddings(max(n_needed, 1))
    raise FileNotFoundError(f"prompt embeddings not found at {path}; run scripts/precompute_embeddings.py first")


def build_env(args: argparse.Namespace, section: str, script_name: str) -> Env:
    from .context import StepContext
    from .flux import load_pipeline, select_prompts
    from .wrap import wrap_transformer

    cfg = apply_overrides(load_config(args.config, args.tiny), args, section)
    device = args.device or default_device(cfg["tiny"])
    setup_torch(device, cfg.get("mem_fraction"))  # before any CUDA allocation
    out_dir, plot_dir, cache_dir = (resolve_path(cfg[k]) for k in ("out_dir", "plot_dir", "cache_dir"))
    for d in (out_dir, plot_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, script_name)

    ids = prompt_ids_for(cfg, args, section)
    emb = load_embeddings(cfg, max(ids) + 1)
    prompts = select_prompts(emb, ids)
    if not cfg["tiny"] and emb.get("model_id") not in (None, cfg["model_id"]):
        raise ValueError(f"embeddings were made with {emb.get('model_id')}, config uses {cfg['model_id']}")

    pipe = load_pipeline(cfg["model_id"], device=device, tiny=cfg["tiny"])
    ctx = StepContext(group_size=int(cfg["group_size"]), storage=cfg["storage"])
    wrapped = wrap_transformer(pipe.transformer, ctx, share_inputs=bool(cfg["share_inputs"]), logger=log.info)
    (out_dir / "logs" / "wrapped_layers.txt").write_text("\n".join(wrapped) + "\n", encoding="utf-8")

    meta = {"script": script_name, "section": section, "started": now(), "git": git_info(), "env": env_info(),
            "device": device, "prompt_ids": ids, "config": cfg, "n_wrapped": len(wrapped),
            "argv": sys.argv[1:]}
    if cfg["tiny"]:
        meta["note"] = "TINY MODE: random tiny model, not a real result"
    log.info("git %s (dirty=%s), device %s, prompts %s", meta["git"]["commit"], meta["git"]["dirty"], device, ids)
    return Env(cfg=cfg, device=device, pipe=pipe, ctx=ctx, wrapped=wrapped, prompts=prompts, out_dir=out_dir,
               plot_dir=plot_dir, cache_dir=cache_dir, meta=meta)
