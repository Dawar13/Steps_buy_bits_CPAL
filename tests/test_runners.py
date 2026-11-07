"""A7.9: every runner with --tiny writes its output files (in-process, LPIPS mocked)."""
import csv
import json

import pytest
import torch

from conftest import load_script


class MockLPIPS(torch.nn.Module):
    def forward(self, a, b):
        return ((a - b) ** 2).mean().reshape(1)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    import sbb.metrics

    monkeypatch.setattr(sbb.metrics, "_load_lpips_model", lambda net="alex": MockLPIPS())
    d = {k: tmp_path / k for k in ("results", "plots", "cache")}
    common = ["--tiny", "--out_dir", str(d["results"]), "--plot_dir", str(d["plots"]), "--cache_dir", str(d["cache"]),
              "--embeds", str(tmp_path / "emb.pt")]
    return tmp_path, d, common


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_all_runners_end_to_end(dirs):
    tmp, d, common = dirs
    prompts = tmp / "prompts.txt"
    assert load_script("make_prompts").main(["--tiny", "--n", "3", "--out", str(prompts)]) == 0
    assert len(prompts.read_text(encoding="utf-8").splitlines()) == 3
    assert (tmp / "prompts_meta.json").exists()

    assert load_script("precompute_embeddings").main(
        ["--tiny", "--prompts", str(prompts), "--out", str(tmp / "emb.pt")]) == 0
    emb = torch.load(tmp / "emb.pt", weights_only=False)
    assert emb["prompt_embeds"].shape[0] == 3 and emb["prompts"][0] == prompts.read_text().splitlines()[0]

    assert load_script("run_b1_sanity").main(common) == 0
    b1 = json.loads((d["results"] / "b1_sanity.json").read_text())
    assert b1["all_pass"] and b1["checks"]["fp_matches_stock"]["exactly_equal"]
    assert b1["checks"]["sd_gamma0_matches_rtn"]["exactly_equal"]

    assert load_script("run_e1_corr").main(common) == 0
    corr = json.loads((d["results"] / "e1_corr.json").read_text())
    assert corr["n_images"] == 2 and set(corr["cos_by_lag"]) == {"1", "2", "5", "10"}
    assert (d["plots"] / "corr_heatmap.png").exists()
    assert len(list((d["results"] / "raw" / "corr").glob("p*.npz"))) == 2

    assert load_script("run_e1_law").main(common + ["--lpips", "on"]) == 0
    rows = _rows(d["results"] / "e1_law.csv")
    assert len(rows) == 2 * 3 * 3 * 2  # prompts x T x modes x bits
    assert all(r["lpips"] not in ("", "nan") for r in rows)
    assert {r["mode"] for r in rows} == {"RTN", "SR", "SD"}
    assert (d["plots"] / "law_b4.png").exists() and (d["plots"] / "law_b3.png").exists()
    assert len(list((d["cache"] / "fp").glob("*.pt"))) == 2 * 3

    # resume: a second run adds nothing
    assert load_script("run_e1_law").main(common + ["--lpips", "on"]) == 0
    assert len(_rows(d["results"] / "e1_law.csv")) == len(rows)

    assert load_script("run_e1_ablation").main(common) == 0
    abl = _rows(d["results"] / "e1_ablation.csv")
    assert {r["mode"] for r in abl} == {"SD", "SD_UNSCALED"} and len(abl) == 2 * 2 * 2

    assert load_script("run_e1_overhead").main(common) == 0
    ov = json.loads((d["results"] / "e1_overhead.json").read_text())
    assert set(ov["runs"]) == {"FP", "RTN", "SD_fp32", "SD_bf16", "SD_int8", "SD_int4"}
    b = {k: ov["runs"][k]["buffer_bytes"] for k in ("SD_fp32", "SD_bf16", "SD_int8", "SD_int4")}
    assert b["SD_fp32"] > b["SD_bf16"] > b["SD_int8"] > b["SD_int4"] > 0
    assert ov["runs"]["SD_fp32"]["buffer_bytes_if_unshared"] > b["SD_fp32"]

    assert load_script("make_report").main(["--tiny", "--out_dir", str(d["results"]), "--plot_dir",
                                            str(d["plots"])]) == 0
    text = (d["results"] / "E1_REPORT.md").read_text(encoding="utf-8")
    assert "## Verdict:" in text and "TINY MODE" in text
    assert "—" not in text
