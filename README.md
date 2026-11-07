# Steps Buy Bits: E1 go/kill experiment

E1 asks whether, in FLUX.1-dev with 4-bit activations, rounding leftovers are strongly correlated between
neighbouring sampling steps, and whether carrying the leftover into the next step (scaled by the ratio of step
sizes) makes the final error fall with the number of steps (slope about -1) while plain rounding stays flat
(slope about 0). Weights stay BF16, so only the activation effect is tested. See `PROJECT_CONTEXT.md` and
`E1_INSTRUCTIONS.md`.

## Layout

```
sbb/quant.py        QuantLinear: RTN, SR, SD (leftover carry), SD_UNSCALED; leftover storage fp32/bf16/int8/int4
sbb/context.py      StepContext, gammas_from_sigmas, StepCallback (callback_on_step_end for the stock pipeline)
sbb/wrap.py         wraps every nn.Linear in transformer_blocks and single_transformer_blocks except "norm" ones
sbb/corr.py         E1.1 recorder: lag cosines of plain-rounding leftovers, drift ratio
sbb/metrics.py      relative latent error, PSNR, LPIPS (lazy), slope fit with bootstrap interval
sbb/flux.py         real or tiny FluxPipeline, generate (seed = prompt index), decode
sbb/experiments.py  E1.1 to E1.4, resumable
sbb/report.py       plots, decision table, E1_REPORT.md
sbb/toy.py          toy linear system for the tests
configs/e1.yaml     all settings and the decision thresholds (do not change the thresholds without asking)
scripts/            entry points; every runner accepts --tiny (tiny random model on CPU, no downloads)
tests/              pytest, CPU only
```

## Design notes

- **Quantiser.** Symmetric, b bits, per token, per group of 64 input channels, scale = max|a'| / (2^(b-1) - 1),
  computed in float32 and cast back. A channel count that is not a multiple of 64 gets a zero-padded last group.
- **Carry.** `a' = a + gamma_i r_prev`, `q = RTN(a')`, `r = a' - q` with `gamma_0 = 0` and
  `gamma_i = h_(i-1) / h_i` (0-based in code). The leftover is measured against the BF16 value the matmul sees.
- **Shared inputs.** `to_q/to_k/to_v` (and in single blocks also `proj_mlp`) read the same tensor, so they share one
  quantiser: the input is quantised once per step, as a fused QKV kernel would do. This keeps SR noise faithful and
  stores one leftover buffer instead of three or four. `--no_share` gives every linear its own quantiser; a test
  checks that RTN and SD give identical results either way.
- **Step index.** The context is set to step 0 before `pipe(...)`. The callback runs after step i, reads
  `pipe.scheduler.sigmas` (T + 1 values, last one 0), and moves the context to step i + 1.
- **Metric.** Every check and slope uses the unsquared `rel_err = ||z_q - z_FP|| / ||z_FP||` (the predicted
  0, -1/2, -1 are for the size of the error). Rows also keep `rel_err_sq`.
- **Leftover storage default** is fp32 for E1.1 to E1.3, so storage precision does not mix into the go/kill
  question; E1.4 measures bf16, int8 and int4. Integer storage uses the step (s / 2) / (2^(b_r - 1) - 1), so the
  leftover range |r| <= s / 2 fills the whole code range.

## Phase A (laptop, CPU)

```bash
python -m venv --system-site-packages .venv      # reuses an existing CPU torch
.venv/Scripts/python -m pip install diffusers==0.42.1 "huggingface_hub>=1.32,<2.0"
.venv/Scripts/python -m pytest -q
```

## Phase B (VM with H100)

`huggingface-cli` no longer works with huggingface_hub 1.x, which diffusers 0.42 needs; use `hf` instead.

```bash
git clone <private repo> "Steps buy bits_CPAL" && cd "Steps buy bits_CPAL"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME=/path/on/vm/big_disk/hf          # every download goes here
hf auth login                                   # accept the FLUX.1-dev licence on Hugging Face first
python scripts/download_flux.py --dry_run       # lists the files and the total (about 34 GB)
python scripts/download_flux.py                 # selective download, no root-level single-file checkpoints
nvidia-smi                                      # check how much memory the CVPR job is using
pytest -q                                       # CPU tests, about 2 minutes

python scripts/make_prompts.py                  # 32 PartiPrompts, seed 0; check the dataset id first (verify)
python scripts/precompute_embeddings.py         # writes cache/prompt_embeds.pt
python scripts/drop_text_encoder_2.py           # dry run: lists what would be deleted
python scripts/drop_text_encoder_2.py --yes     # frees about 9.5 GB
git add prompts.txt prompts_meta.json && git commit -m "E1 prompts" && git push
export HF_HUB_OFFLINE=1                         # runs use the local cache only
```

Every GPU script calls `torch.cuda.set_per_process_memory_fraction(0.5)` before loading anything
(`mem_fraction` in `configs/e1.yaml`). Run long jobs inside `tmux` or with `nohup`; every runner appends results as
it goes, so rerunning the same command resumes where it stopped.

```bash
# B1 sanity: FP equals the stock pipeline; SD with gamma = 0 equals RTN (exit status 1 on failure)
python scripts/run_b1_sanity.py

# B2
python scripts/run_e1_corr.py                   # E1.1, about 20 min
python scripts/run_e1_law.py --n_prompts 16     # E1.2 first half if the GPU is busy
python scripts/run_e1_law.py                    # E1.2 all 32 prompts (skips finished rows), about 2 to 2.5 h
python scripts/run_e1_ablation.py               # E1.3, about 15 min
python scripts/run_e1_overhead.py               # E1.4, about 10 min

# B3
python scripts/make_report.py                   # results/E1_REPORT.md with the decision table applied
```

Outputs: `results/e1_corr.json`, `plots/corr_heatmap.png`, `results/e1_law.csv`, `plots/law_b4.png`,
`plots/law_b3.png`, `results/e1_ablation.csv`, `results/e1_overhead.json`, `results/E1_REPORT.md`, plus
`results/b1_sanity.json`. Raw per-prompt data goes to `results/raw/` (not committed); full-precision reference
latents are cached in `cache/fp/`. Every row and JSON file records the git commit, the config and the seed.

## Decision rules as implemented (confirmed 7 Nov 2026)

The thresholds live in `configs/e1.yaml` under `decision:`.

- Every check uses the unsquared rel_err; slope and ratio checks use b = 4 (3-bit is reported, does not decide).
- SD at T = 50: GO at or below 0.5 x RTN with lower LPIPS; KILL at or above 0.8 x RTN.
- RTN slope: GO between -0.3 and +0.1; at or below -0.4 the premise is wrong (KILL); between -0.4 and -0.3 gives
  UNDECIDED.
- WEAK: SD beats RTN and either the SD and SR slope intervals overlap or SD's error at T = 50 is not lower than
  SR's. WEAK overrides GO.
- KILL: cosine below 0.3, SD at or above 0.8 x RTN, or the RTN premise rule. An SD slope above -0.3 alone gives
  UNDECIDED. UNDECIDED means neither GO nor KILL fired; read the plots.
