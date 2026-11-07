"""Build results/E1_REPORT.md from the E1 result files and apply the decision table (B3).

Reads results/e1_corr.json, results/e1_law.csv, results/e1_ablation.csv, results/e1_overhead.json
(whichever exist), redraws plots/law_b<bits>.png and plots/corr_heatmap.png, and writes the report.
"""
import argparse
import os
import sys

import _bootstrap  # noqa: F401

from sbb.report import build_report, plot_corr_heatmap, plot_law
from sbb.runtime import git_info, load_config, now, read_json, resolve_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--plot_dir", default=None)
    args = p.parse_args(argv)
    import pandas as pd

    cfg = load_config(args.config, args.tiny)
    out_dir = resolve_path(args.out_dir or cfg["out_dir"])
    plot_dir = resolve_path(args.plot_dir or cfg["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "E1_REPORT.md"

    def rel(p):
        return os.path.relpath(p, out_dir).replace(os.sep, "/")

    inputs, plots = {}, {}
    corr = law = abl = overhead = None
    f = out_dir / "e1_corr.json"
    if f.exists():
        corr = read_json(f)
        inputs["e1_corr.json"] = f"commit {corr.get('git', {}).get('commit')}, finished {corr.get('finished')}"
        hp = plot_corr_heatmap(corr, plot_dir / "corr_heatmap.png", tiny=cfg["tiny"])
        if hp:
            plots["corr"] = rel(hp)
    f = out_dir / "e1_law.csv"
    if f.exists():
        law = pd.read_csv(f)
        inputs["e1_law.csv"] = f"{len(law)} rows, commits {sorted(law['git_commit'].astype(str).unique())}"
        for b in sorted(law["bits"].unique()):
            lp = plot_law(law, int(b), plot_dir / f"law_b{int(b)}.png", t_min=float(cfg["fit"]["t_min"]),
                          tiny=cfg["tiny"])
            if lp:
                plots[f"law_b{int(b)}"] = rel(lp)
    f = out_dir / "e1_ablation.csv"
    if f.exists():
        abl = pd.read_csv(f)
        inputs["e1_ablation.csv"] = f"{len(abl)} rows, commits {sorted(abl['git_commit'].astype(str).unique())}"
    f = out_dir / "e1_overhead.json"
    if f.exists():
        overhead = read_json(f)
        inputs["e1_overhead.json"] = f"commit {overhead.get('git', {}).get('commit')}"

    meta = {"generated": now(), "git": git_info(), "inputs": inputs}
    text = build_report(cfg, corr, law, abl, overhead, meta, plots)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    verdict = next((ln for ln in text.splitlines() if ln.startswith("## Verdict")), "")
    print(f"wrote {report_path}: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
