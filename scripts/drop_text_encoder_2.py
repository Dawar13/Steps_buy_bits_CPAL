"""Remove FLUX.1-dev text_encoder_2 (T5, about 9.5 GB) from the Hugging Face cache after embeddings are cached.

Dry run by default: lists the files and the blobs they point to. Pass --yes to delete them.
Uses $HF_HOME (or HF_HUB_CACHE) through huggingface_hub, so run it in the same shell as the download.
"""
import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--folder", default="text_encoder_2")
    p.add_argument("--yes", action="store_true", help="actually delete")
    args = p.parse_args(argv)
    from huggingface_hub import constants

    repo_dir = Path(constants.HF_HUB_CACHE) / ("models--" + args.repo.replace("/", "--"))
    snaps = repo_dir / "snapshots"
    if not snaps.exists():
        print(f"nothing cached at {repo_dir}")
        return 0
    plan, total = [], 0
    for snap in sorted(snaps.iterdir()):
        d = snap / args.folder
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if f.is_dir():
                continue
            target = f.resolve()
            size = target.stat().st_size if target.exists() else 0
            total += size
            plan.append((f, target if f.is_symlink() else None, size))
    if not plan:
        print(f"no {args.folder} files under {snaps}")
        return 0
    for link, blob, size in plan:
        print(f"{size / 1e9:7.2f} GB  {link}" + (f" -> {blob}" if blob else ""))
    print(f"total {total / 1e9:.2f} GB")
    if not args.yes:
        print("dry run; pass --yes to delete")
        return 0
    for link, blob, _ in plan:
        if blob is not None and blob.exists():
            blob.unlink()
        link.unlink(missing_ok=True)
    for snap in snaps.iterdir():
        d = snap / args.folder
        if d.exists():
            for sub in sorted(d.rglob("*"), reverse=True):
                if sub.is_dir():
                    sub.rmdir()
            d.rmdir()
    print("deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
