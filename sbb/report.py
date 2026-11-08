"""Plots, the E1 decision table and the Markdown report (British spelling, no em-dashes)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

# Reference palette (dataviz skill): categorical slots in fixed order, chart chrome and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MODE_COLOURS = {"RTN": "#2a78d6", "SR": "#eb6834", "SD": "#1baf7a", "SD_UNSCALED": "#eda100", "FP": "#52514e"}
MODE_LABELS = {"RTN": "RTN (plain rounding)", "SR": "SR (stochastic rounding)", "SD": "SD (leftover carry)",
               "SD_UNSCALED": "SD unscaled (gamma = 1)", "FP": "FP"}
DIVERGING = ["#104281", "#3987e5", "#f0efec", "#e66767", "#a8292b"]  # blue <-> grey <-> red


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
    })
    return plt


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{float(x):.{nd}g}"
    return f"{float(x):.{nd}f}"


# --------------------------------------------------------------------------- plots
def plot_law(df, bits: int, path: Path, t_min: float = 16, tiny: bool = False, modes=("RTN", "SR", "SD")) -> Path | None:
    from .metrics import slope_table

    plt = _mpl()
    d = df[df["bits"] == bits]
    if d.empty:
        return None
    _, summ = slope_table(d, t_min=t_min)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ends = []
    for mode in modes:
        m = d[d["mode"] == mode]
        if m.empty:
            continue
        g = m.groupby("T")["rel_err"]
        ts = np.array(sorted(m["T"].unique()), dtype=float)
        med = g.median().reindex(ts).to_numpy()
        lo, hi = g.quantile(0.25).reindex(ts).to_numpy(), g.quantile(0.75).reindex(ts).to_numpy()
        c = MODE_COLOURS.get(mode, INK_2)
        ax.fill_between(ts, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.plot(ts, med, color=c, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=MODE_LABELS.get(mode, mode))
        row = summ[summ["mode"] == mode]
        slope = float(row["slope_median"].iloc[0]) if not row.empty else float("nan")
        ends.append((ts[-1], med[-1], f"{mode}  slope {_fmt(slope, 2)}"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(t_min, color=AXIS, linewidth=1, linestyle=":")
    ax.annotate(f"fit uses T >= {t_min:g}", (t_min, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                textcoords="offset points", fontsize=8, color=MUTED)
    ts_all = sorted(d["T"].unique())
    ax.set_xticks(ts_all)
    ax.set_xticklabels([str(t) for t in ts_all])
    ax.minorticks_off()
    ax.set_xlim(ts_all[0] * 0.85, ts_all[-1] * 2.2)
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ylo, yhi = ax.get_ylim()
    decades = math.log10(yhi) - math.log10(ylo)
    subs = (1.0, 2.0, 5.0) if decades > 1.0 else (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    # Direct labels at the line ends, in ink, nudged apart so they never overlap.
    gap = 0.06 * decades
    placed = []
    for x, y, text in sorted(ends, key=lambda e: e[1]):
        ly = math.log10(y)
        if placed and ly - placed[-1] < gap:
            ly = placed[-1] + gap
        placed.append(ly)
        ax.annotate(text, (x, y), xytext=(x * 1.08, 10 ** ly), textcoords="data", va="center", fontsize=9,
                    color=INK)
    ax.set_xlabel("sampling steps T")
    ax.set_ylabel("relative latent error ||z_q - z_FP|| / ||z_FP||")
    title = (("TINY TEST MODEL. " if tiny else "") + f"Error versus steps, {bits}-bit activations\n"
             f"median over {d['prompt_id'].nunique()} prompts, band = quartiles")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _short_key(k: str) -> str:
    return k.replace("single_transformer_blocks", "single").replace("transformer_blocks", "double")


def plot_corr_heatmap(summary: dict, path: Path, tiny: bool = False) -> Path | None:
    from matplotlib.colors import LinearSegmentedColormap

    plt = _mpl()
    mat = summary.get("lag1_matrix")
    if not mat:
        return None
    a = np.array([[np.nan if v is None else v for v in row] for row in mat], dtype=float)
    keys = summary["keys"]
    cmap = LinearSegmentedColormap.from_list("sbb_div", DIVERGING)
    cmap.set_bad(SURFACE)
    h = max(3.5, min(9.0, 0.03 * len(keys) + 2.5))
    fig, ax = plt.subplots(figsize=(7.2, h))
    im = ax.imshow(a, aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.grid(False)
    n = len(keys)
    step = max(1, n // 12)
    ticks = list(range(0, n, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([_short_key(keys[i]) for i in ticks], fontsize=7)
    ax.set_xlabel("sampling step i (cosine of leftover at step i with step i - 1)")
    ax.set_ylabel("quantised input (execution order)")
    s = summary.get("settings", {})
    title = (("TINY TEST MODEL. " if tiny else "") + "Lag-1 cosine of plain-rounding leftovers\n"
             f"{s.get('bits', '?')}-bit, T = {s.get('T', '?')}, median over {summary.get('n_images', '?')} prompts")
    ax.set_title(title, fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("cosine similarity", color=INK_2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- decision
def _slope_ci(summ, mode: str, bits: int):
    if summ is None or summ.empty:
        return None
    r = summ[(summ["mode"] == mode) & (summ["bits"] == bits)]
    if r.empty:
        return None
    r = r.iloc[0]
    return {"median": float(r["slope_median"]), "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
            "n_prompts": int(r["n_prompts"])}


def _paired_ratio(df, mode_a: str, mode_b: str, bits: int, T: int, metric: str):
    a = df[(df["mode"] == mode_a) & (df["bits"] == bits) & (df["T"] == T)].set_index("prompt_id")[metric]
    b = df[(df["mode"] == mode_b) & (df["bits"] == bits) & (df["T"] == T)].set_index("prompt_id")[metric]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return None
    ratios = (a.loc[common] / b.loc[common]).replace([np.inf, -np.inf], np.nan).dropna()
    return {"median": float(ratios.median()) if len(ratios) else float("nan"), "n_prompts": int(len(common))}


def _median(df, mode: str, bits: int, T: int, col: str):
    v = df[(df["mode"] == mode) & (df["bits"] == bits) & (df["T"] == T)][col].astype(float)
    v = v[np.isfinite(v)]
    return float(v.median()) if len(v) else float("nan")


def decide(corr: dict | None, law_df, abl_df, cfg: dict) -> dict:
    """Apply the E1 decision table. Thresholds come from cfg['decision'] and are not altered here."""
    from .metrics import slope_table

    dc, fit = cfg["decision"], cfg["fit"]
    bits, T = int(dc["bits"]), int(dc["ratio_T"])
    cells: dict[str, dict] = {}
    missing = []

    # E1.1
    if corr is not None and corr.get("cos_by_lag", {}).get("1", {}).get("median") is not None:
        v = float(corr["cos_by_lag"]["1"]["median"])
        status = "GO" if v >= dc["corr_go"] else ("KILL" if v < dc["corr_kill"] else "BETWEEN")
        cells["corr"] = {"value": v, "status": status}
    else:
        missing.append("E1.1 (results/e1_corr.json)")

    summ = None
    if law_df is not None and not law_df.empty:
        _, summ = slope_table(law_df, t_min=float(fit["t_min"]), n_boot=int(fit["n_boot"]), seed=int(fit["seed"]))
        rtn = _slope_ci(summ, "RTN", bits)
        sd = _slope_ci(summ, "SD", bits)
        sr = _slope_ci(summ, "SR", bits)
        if rtn is not None:
            v = rtn["median"]
            lo, hi = dc["rtn_slope_go"]
            if lo <= v <= hi:
                status = "GO"
            elif v <= dc["rtn_slope_premise_wrong"]:
                status = "KILL"
            else:
                status = "BETWEEN"
            cells["rtn_slope"] = {"value": v, "ci": rtn, "status": status}
        if sd is not None:
            v = sd["median"]
            status = "GO" if v <= dc["sd_slope_go"] else ("KILL" if v > dc["sd_slope_kill"] else "BETWEEN")
            cells["sd_slope"] = {"value": v, "ci": sd, "status": status}
        if sr is not None:
            cells["sr_slope"] = {"value": sr["median"], "ci": sr, "status": "INFO"}
        metric = dc.get("ratio_metric", "rel_err")
        ratio = _paired_ratio(law_df, "SD", "RTN", bits, T, metric)
        if ratio is not None:
            lp_sd, lp_rtn = _median(law_df, "SD", bits, T, "lpips"), _median(law_df, "RTN", bits, T, "lpips")
            lpips_ok = math.isfinite(lp_sd) and math.isfinite(lp_rtn) and lp_sd < lp_rtn
            v = ratio["median"]
            if v >= dc["sd_ratio_kill"]:
                status = "KILL"
            elif v <= dc["sd_ratio_go"] and lpips_ok:
                status = "GO"
            else:
                status = "BETWEEN"
            cells["sd_ratio"] = {"value": v, "metric": metric, "lpips_sd": lp_sd, "lpips_rtn": lp_rtn,
                                 "lpips_lower": lpips_ok, "n_prompts": ratio["n_prompts"], "status": status}
        sd_sr = _paired_ratio(law_df, "SD", "SR", bits, T, metric)
        if sd_sr is not None and "sd_slope" in cells and "sr_slope" in cells:
            sd_ci, sr_ci = cells["sd_slope"]["ci"], cells["sr_slope"]["ci"]
            overlap = sd_ci["ci_low"] <= sr_ci["ci_high"] and sr_ci["ci_low"] <= sd_ci["ci_high"]
            not_lower = not (sd_sr["median"] < 1.0)  # SD error at T not lower than SR's (NaN counts as not lower)
            cells["sd_vs_sr"] = {"overlap": overlap, "sd_over_sr": sd_sr["median"], "not_lower": not_lower,
                                 "status": "INFO"}
    else:
        missing.append("E1.2 (results/e1_law.csv)")

    if abl_df is not None and not abl_df.empty:
        per_t = []
        for t in sorted(abl_df["T"].unique()):
            a = _median(abl_df, "SD", bits, int(t), "rel_err")
            b = _median(abl_df, "SD_UNSCALED", bits, int(t), "rel_err")
            per_t.append({"T": int(t), "sd": a, "sd_unscaled": b})
        ok = all(math.isfinite(p["sd"]) and math.isfinite(p["sd_unscaled"]) and p["sd"] <= p["sd_unscaled"]
                 for p in per_t)
        cells["ablation"] = {"per_T": per_t, "status": "OK" if ok else "NOT OK", "informative": True}

    # Verdict, in order KILL > WEAK > GO > UNDECIDED.
    # Overall KILL: cosine below corr_kill, SD at or above sd_ratio_kill x RTN, or RTN slope at or below
    # rtn_slope_premise_wrong (premise wrong). An SD slope KILL cell alone gives UNDECIDED.
    reasons = []
    kill_keys = ("corr", "sd_ratio", "rtn_slope")
    kill = any(cells.get(k, {}).get("status") == "KILL" for k in kill_keys)
    weak = False
    vs = cells.get("sd_vs_sr")
    if "sd_ratio" in cells and vs is not None:
        weak = cells["sd_ratio"]["value"] < 1.0 and (vs["overlap"] or vs["not_lower"])
    go_cells = ("corr", "rtn_slope", "sd_slope", "sd_ratio")
    all_go = all(cells.get(k, {}).get("status") == "GO" for k in go_cells)
    if missing:
        verdict = "INCOMPLETE"
        reasons.append("missing inputs: " + ", ".join(missing))
    if kill:
        verdict = "KILL"
        if cells.get("corr", {}).get("status") == "KILL":
            reasons.append(f"median lag-1 leftover cosine {_fmt(cells['corr']['value'])} is below {dc['corr_kill']}")
        if cells.get("sd_ratio", {}).get("status") == "KILL":
            reasons.append(f"SD error at T = {T} is {_fmt(cells['sd_ratio']['value'])} x RTN, at or above "
                           f"{dc['sd_ratio_kill']}")
        if cells.get("rtn_slope", {}).get("status") == "KILL":
            reasons.append(f"RTN slope {_fmt(cells['rtn_slope']['value'], 2)} is at or below "
                           f"{dc['rtn_slope_premise_wrong']}: plain rounding already improves with steps, so the "
                           "premise is wrong")
    elif missing:
        pass
    elif weak:
        verdict = "WEAK"
        why = []
        if vs["overlap"]:
            why.append("its slope interval overlaps the SR slope interval")
        if vs["not_lower"]:
            why.append(f"its error at T = {T} is not lower than SR's ({_fmt(vs['sd_over_sr'])} x SR)")
        reasons.append("SD beats RTN but " + " and ".join(why) + ", so the gain may come from decorrelation "
                       "rather than cancellation")
    elif all_go:
        verdict = "GO"
        reasons.append("every GO cell is met")
    else:
        verdict = "UNDECIDED"
        for k in go_cells:
            st = cells.get(k, {}).get("status")
            if st != "GO":
                reasons.append(f"{k}: {st}")
        reasons.append("no KILL rule fired and not every GO cell is met; discuss")
    return {"verdict": verdict, "reasons": reasons, "cells": cells, "bits": bits, "ratio_T": T,
            "slope_summary": summ}


# --------------------------------------------------------------------------- report
def _status_word(s: str | None) -> str:
    return {"GO": "GO", "KILL": "KILL", "BETWEEN": "neither", "INFO": "info", "OK": "OK",
            "NOT OK": "not OK", None: "not run"}.get(s, str(s))


def build_report(cfg: dict, corr: dict | None, law_df, abl_df, overhead: dict | None, meta: dict,
                 plots: dict[str, str]) -> str:
    dc = cfg["decision"]
    res = decide(corr, law_df, abl_df, cfg)
    cells, bits, T = res["cells"], res["bits"], res["ratio_T"]
    L: list[str] = []
    L.append("# E1 report: Steps Buy Bits go/kill experiment")
    L.append("")
    L.append(f"Generated {meta['generated']} by `scripts/make_report.py` from commit `{meta['git']['commit']}` "
             f"(uncommitted changes: {meta['git']['dirty']}).")
    if cfg.get("tiny"):
        L.append("")
        L.append("> **TINY MODE.** These numbers come from a tiny random model on CPU. They test the code path "
                 "only and say nothing about FLUX.")
    L.append("")
    L.append(f"Model `{cfg['model_id']}`, {cfg['height']} x {cfg['width']}, guidance {cfg['guidance']}, default "
             f"scheduler, seed = prompt index. Activations only are quantised (symmetric, per token, groups of "
             f"{cfg['group_size']} channels); weights stay BF16. Full precision at the same T is the reference.")
    L.append("")
    L.append(f"## Verdict: **{res['verdict']}**")
    L.append("")
    for r in res["reasons"]:
        L.append(f"- {r}")
    nxt = {"GO": "Next: E2 (W4A4 with baselines and quality metrics on FLUX and SD3.5).",
           "WEAK": "Discuss before continuing.",
           "KILL": "Stop here, as agreed in the decision table.",
           "UNDECIDED": "Discuss before continuing.",
           "INCOMPLETE": "Run the missing parts and regenerate this report."}[res["verdict"]]
    L.append("")
    L.append(nxt)
    L.append("")

    # Decision table
    L.append("## Decision table")
    L.append("")
    L.append("| Check | Measured | GO if | KILL if | Status |")
    L.append("|---|---|---|---|---|")
    c = cells.get("corr")
    L.append(f"| E1.1 median lag-1 leftover cosine (b = {corr['settings']['bits'] if corr else 4}, "
             f"T = {corr['settings']['T'] if corr else 50}) | {_fmt(c['value']) if c else 'not run'} | "
             f">= {dc['corr_go']} | < {dc['corr_kill']} | {_status_word(c['status'] if c else None)} |")
    c = cells.get("rtn_slope")
    lo, hi = dc["rtn_slope_go"]
    L.append(f"| E1.2 RTN slope (b = {bits}) | {_ci_text(c)} | between {lo} and {hi} | <= "
             f"{dc['rtn_slope_premise_wrong']} (premise wrong) | {_status_word(c['status'] if c else None)} |")
    c = cells.get("sd_slope")
    L.append(f"| E1.2 SD slope (b = {bits}) | {_ci_text(c)} | <= {dc['sd_slope_go']} | > {dc['sd_slope_kill']} | "
             f"{_status_word(c['status'] if c else None)} |")
    c = cells.get("sd_ratio")
    if c:
        meas = (f"{_fmt(c['value'])} x RTN ({c['metric']}); LPIPS SD {_fmt(c['lpips_sd'])} vs RTN "
                f"{_fmt(c['lpips_rtn'])}")
    else:
        meas = "not run"
    L.append(f"| E1.2 SD latent error at T = {T}, b = {bits} | {meas} | <= {dc['sd_ratio_go']} x RTN and lower "
             f"LPIPS | >= {dc['sd_ratio_kill']} x RTN | {_status_word(c['status'] if c else None)} |")
    c = cells.get("ablation")
    if c:
        meas = "; ".join(f"T = {p['T']}: SD {_fmt(p['sd'])}, unscaled {_fmt(p['sd_unscaled'])}" for p in c["per_T"])
    else:
        meas = "not run"
    L.append(f"| E1.3 step-ratio gamma vs unscaled (median rel_err) | {meas} | step-ratio at least as good | "
             f"informative | {_status_word(c['status'] if c else None)} |")
    c = cells.get("sr_slope")
    if c:
        vs = cells.get("sd_vs_sr", {})
        L.append(f"| SR slope (b = {bits}), for the WEAK rule | {_ci_text(c)}; SD error at T = {T} is "
                 f"{_fmt(vs.get('sd_over_sr'))} x SR | | | info; intervals overlap: {vs.get('overlap')}; SD not "
                 f"lower than SR: {vs.get('not_lower')} |")
    L.append("")

    lo_go, hi_go = dc["rtn_slope_go"]
    L.append("### How the checks are computed (rules confirmed on 7 Nov 2026)")
    L.append("")
    L.append("- Error: rel_err = ||z_q - z_FP|| / ||z_FP|| (unsquared) for every check, against FP at the same T. "
             "The CSV also keeps the squared form rel_err_sq.")
    L.append(f"- Slopes: least squares of log(rel_err) on log(T) for T >= {cfg['fit']['t_min']}, per prompt; the "
             f"median over prompts with a {cfg['fit']['n_boot']}-sample bootstrap 95% interval.")
    L.append(f"- SD versus RTN at T = {T}: median over prompts of the per-prompt ratio of {dc['ratio_metric']}. "
             "LPIPS check: median LPIPS of SD below that of RTN.")
    L.append(f"- Slope and ratio checks use b = {bits}; 3-bit results are reported below but do not decide.")
    L.append(f"- RTN slope: GO between {lo_go} and {hi_go}; at or below {dc['rtn_slope_premise_wrong']} the premise "
             f"is wrong (KILL); between {dc['rtn_slope_premise_wrong']} and {lo_go} gives UNDECIDED.")
    L.append(f"- WEAK: SD beats RTN at T = {T} and either the bootstrap intervals of the SD and SR slopes overlap or "
             f"SD's error at T = {T} is not lower than SR's (median per-prompt ratio at or above 1). WEAK overrides "
             "GO.")
    L.append(f"- Overall KILL: cosine below {dc['corr_kill']}, SD at or above {dc['sd_ratio_kill']} x RTN, or the "
             f"RTN premise rule. An SD slope above {dc['sd_slope_kill']} shows as a KILL cell but on its own gives "
             "UNDECIDED. UNDECIDED means neither GO nor KILL fired; read the plots.")
    L.append("")

    # E1.1
    L.append("## E1.1 Leftover correlation")
    L.append("")
    if corr:
        s = corr["settings"]
        L.append(f"Mode {s['mode']}, b = {s['bits']}, T = {s['T']}, {corr['n_images']} prompts "
                 f"(ids {_ids(corr.get('prompt_ids'))}), {corr['n_layers']} quantised inputs, "
                 f"{s['n_entries']} sampled entries per input (seed {s['seed']}). Statistics pool layers, steps and "
                 "prompts.")
        L.append("")
        L.append("| Lag | Median cosine | Quartiles | Values |")
        L.append("|---|---|---|---|")
        for lag, st in corr["cos_by_lag"].items():
            L.append(f"| {lag} | {_fmt(st['median'])} | {_fmt(st['q25'])} to {_fmt(st['q75'])} | {st['n']} |")
        dr = corr["drift_ratio"]
        L.append("")
        L.append(f"Drift ratio ||a_i - a_(i-1)|| / ||s_i||: median {_fmt(dr['median'])} (quartiles "
                 f"{_fmt(dr['q25'])} to {_fmt(dr['q75'])}). Values well below 1 mean neighbouring steps move by less "
                 "than one quantisation step.")
        if corr.get("lag1_by_kind"):
            L.append("")
            L.append("| Input kind | Median lag-1 cosine | Quartiles |")
            L.append("|---|---|---|")
            for k, st in corr["lag1_by_kind"].items():
                L.append(f"| `{_short_key(k)}` | {_fmt(st['median'])} | {_fmt(st['q25'])} to {_fmt(st['q75'])} |")
        if plots.get("corr"):
            L.append("")
            L.append(f"![lag-1 cosine heat map]({plots['corr']})")
    else:
        L.append("Not run yet.")
    L.append("")

    # E1.2
    L.append("## E1.2 Error versus steps")
    L.append("")
    if law_df is not None and not law_df.empty:
        summ = res["slope_summary"]
        n_p = law_df["prompt_id"].nunique()
        L.append(f"{n_p} prompts. Median relative latent error rel_err over prompts (FP at the same T is the "
                 "reference):")
        bad = law_df[~np.isfinite(law_df["rel_err"].astype(float))]
        if len(bad):
            runs = "; ".join(f"prompt {int(r.prompt_id)}, T = {int(r.T)}, {r.mode}, b = {int(r.bits)}"
                             for r in bad.itertuples())
            L.append("")
            L.append(f"{len(bad)} of {len(law_df)} runs ended with a non-finite latent and are left out of the "
                     f"medians, slopes and ratios: {runs}.")
        for b in sorted(law_df["bits"].unique(), reverse=True):
            d = law_df[law_df["bits"] == b]
            ts = sorted(d["T"].unique())
            modes = [m for m in ("RTN", "SR", "SD") if m in set(d["mode"])]
            L.append("")
            L.append(f"**b = {b}**")
            L.append("")
            L.append("| Mode | " + " | ".join(f"T = {t}" for t in ts) + " | Slope (95% interval) |")
            L.append("|---|" + "---|" * len(ts) + "---|")
            for m in modes:
                vals = [_fmt(_median(d, m, int(b), int(t), "rel_err"), 4) for t in ts]
                L.append(f"| {m} | " + " | ".join(vals) + f" | {_ci_text({'ci': _slope_ci(summ, m, int(b))})} |")
            if T in ts:
                L.append("")
                L.append(f"At T = {T}: " + "; ".join(
                    f"{m} PSNR {_fmt(_median(d, m, int(b), T, 'psnr'), 2)} dB, LPIPS {_fmt(_median(d, m, int(b), T, 'lpips'))}"
                    for m in modes) + " (medians over prompts, against FP at the same T).")
            key = f"law_b{b}"
            if plots.get(key):
                L.append("")
                L.append(f"![law b = {b}]({plots[key]})")
    else:
        L.append("Not run yet.")
    L.append("")

    # E1.3
    L.append("## E1.3 Step-size matching ablation")
    L.append("")
    if abl_df is not None and not abl_df.empty:
        L.append("| T | SD median rel_err | SD unscaled median rel_err | Unscaled / SD |")
        L.append("|---|---|---|---|")
        for p in cells["ablation"]["per_T"]:
            ratio = p["sd_unscaled"] / p["sd"] if p["sd"] and math.isfinite(p["sd"]) else float("nan")
            L.append(f"| {p['T']} | {_fmt(p['sd'], 4)} | {_fmt(p['sd_unscaled'], 4)} | {_fmt(ratio)} |")
        L.append("")
        L.append("On a smooth schedule neighbouring step sizes differ by a small amount, so the unscaled carry is "
                 "expected to be close to the step-ratio carry; the difference grows with how uneven the steps are.")
    else:
        L.append("Not run yet.")
    L.append("")

    # E1.4
    L.append("## E1.4 Overhead")
    L.append("")
    if overhead and overhead.get("runs"):
        st = overhead["settings"]
        L.append(f"b = {st['bits']}, T = {st['T']}, prompts {_ids(st['prompt_ids'])}. Times are for fake "
                 "quantisation in PyTorch, for information only. Buffer sizes are exact sums over the leftover "
                 "buffers; peak memory is torch.cuda.max_memory_allocated.")
        L.append("")
        L.append("| Run | Leftover buffers (GB) | Without input sharing (GB) | Extra peak vs RTN (GB) | Step time (s) | "
                 "Time vs RTN | Median rel_err | Error vs RTN |")
        L.append("|---|---|---|---|---|---|---|---|")
        der = overhead.get("derived", {})
        for name, e in overhead["runs"].items():
            dv = der.get(name, {})
            extra = dv.get("extra_peak_bytes_vs_RTN")
            L.append(f"| {name} | {_fmt(e['buffer_bytes'] / 1e9)} | {_fmt(e['buffer_bytes_if_unshared'] / 1e9)} | "
                     f"{_fmt(extra / 1e9) if extra is not None else 'n/a'} | {_fmt(e['step_seconds_median'], 4)} | "
                     f"{_fmt(dv.get('step_time_ratio_vs_RTN'))} | {_fmt(e['rel_err_median'], 4)} | "
                     f"{_fmt(dv.get('rel_err_ratio_vs_RTN'))} |")
    else:
        L.append("Not run yet.")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append(f"- Report commit: `{meta['git']['commit']}`; result files record the commit of the run that wrote them.")
    for name, info in meta.get("inputs", {}).items():
        L.append(f"- {name}: {info}")
    text = "\n".join(L) + "\n"
    return text.replace("—", "-").replace("–", "-")


def _ci_text(c: dict | None) -> str:
    if not c or not c.get("ci"):
        return "not run"
    ci = c["ci"]
    return f"{_fmt(ci['median'], 2)} [{_fmt(ci['ci_low'], 2)}, {_fmt(ci['ci_high'], 2)}], {ci['n_prompts']} prompts"


def _ids(ids) -> str:
    if not ids:
        return "n/a"
    ids = sorted(int(i) for i in ids)
    if ids == list(range(ids[0], ids[-1] + 1)) and len(ids) > 2:
        return f"{ids[0]} to {ids[-1]}"
    return ", ".join(str(i) for i in ids)
