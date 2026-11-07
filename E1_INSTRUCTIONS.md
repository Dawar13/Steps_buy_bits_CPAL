# E1 INSTRUCTIONS: Steps Buy Bits go/kill experiment

This replaces the earlier E1_SPEC.md.

**Question:** in FLUX.1-dev, are 4-bit activation rounding leftovers strongly correlated between
neighbouring sampling steps, and does a step-size-matched leftover carry make the final error fall
with the number of steps (slope about -1) while plain rounding stays flat (slope about 0)?
Weights stay in BF16 in E1, so only the activation effect is tested.

Work happens in two phases.
- **Phase A (now, laptop, no GPU, no downloads):** write all code and tests; run tests on CPU with a tiny random Flux transformer and toy systems.
- **Phase B (VM with H100):** set up, download selectively, run sanity checks, run E1, write the report.

---

## Phase A: code everything offline

### A0. Environment (laptop, small)
- If local disk allows (about 2 GB): venv with CPU-only PyTorch
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), plus
  `diffusers numpy pandas matplotlib scipy pyyaml pytest tqdm`.
- If disk is too tight: write code and tests without running them; Phase B starts by running the tests.
- Do not download FLUX, LPIPS weights, prompts datasets or anything else on the laptop.

### A1. Quantiser (`sbb/quant.py`)
- `QuantLinear(nn.Module)`: wraps an `nn.Linear`, quantises its input, then calls the original linear with the original BF16 weight.
- Activation quantiser: symmetric uniform, b bits, per token, per group of 64 input channels;
  scale s = max|a'| over the group / (2^(b-1) - 1);
  q = clamp(round(a'/s), -(2^(b-1)-1), 2^(b-1)-1) * s. Compute in float32, cast back.
- Modes:
  - `FP`: pass-through.
  - `RTN`: a' = a.
  - `SR`: stochastic rounding floor(a/s + U[0,1)) * s, fresh U each call, seeded generator.
  - `SD` (ours): a' = a + gamma_i * r_prev; q = RTN(a'); r = a' - q stored in the module's buffer for the next step.
  - `SD_UNSCALED` (ablation): as SD with gamma_i = 1 (except step 1).
- Leftover buffer: allocated lazily with the input's shape; optional storage precision (bf16, 8-bit, 4-bit with scale s/2^(b_r - 1)).
- `reset()` clears buffers (called at the start of every image).

### A2. Step context (`sbb/context.py`)
- Global `StepContext` with fields: step index i, gamma_i, mode, bits, flags for recording.
- `gammas_from_sigmas(sigmas)`: h_i = sigma_i - sigma_(i+1); gamma_1 = 0; gamma_i = h_(i-1)/h_i for i >= 2.
- Driving it from the stock pipeline: set i = 0 before `pipe(...)`; in `callback_on_step_end`, read
  `pipe.scheduler.sigmas` (available after `set_timesteps`), advance i and set gamma_i for the next step.
  Return `callback_kwargs` unchanged.

### A3. Wrapping (`sbb/wrap.py`)
- Replace every `nn.Linear` inside `transformer.transformer_blocks` and
  `transformer.single_transformer_blocks` whose qualified name does not contain `norm`
  (excludes adaptive-norm modulation layers). Embedders and the final output layer stay full precision.
- Return the list of wrapped names; log it.

### A4. Correlation recorder (`sbb/corr.py`)
- For each wrapped layer, a fixed random subset of 65,536 input entries (seeded). At every step, compute
  the plain-rounding leftover r_i = a_i - RTN(a_i) on that subset (no carry), keep a ring buffer of the last 11.
- Cosine similarity of r_i with r_(i-lag) for lag in {1, 2, 5, 10}; drift ratio ||a_i - a_(i-1)|| / ||s_i|| on the subset.
- Aggregate: median and quartiles over layers and steps; a layer x step matrix for lag 1.

### A5. Metrics (`sbb/metrics.py`)
- Relative latent error ||z_q - z_FP||^2 / ||z_FP||^2 (primary, before VAE decoding).
- PSNR on decoded images in [0, 1]. LPIPS (AlexNet) through the `lpips` package, imported lazily (VM only); tests use a mock.
- Slope fit: least squares of log error on log T for T >= 16, per mode, bits and prompt, then the median and a bootstrap 95% interval over prompts.

### A6. Runners (`scripts/`)
- `precompute_embeddings.py`: load the pipeline with transformer and VAE set to None if supported
  (otherwise load fully), encode the 32 prompts, save `cache/prompt_embeds.pt` (prompt_embeds and pooled), free memory.
- `make_prompts.py`: sample 32 prompts from PartiPrompts with seed 0 (dataset id to be checked on the VM, verify), save `prompts.txt` (commit it).
- `run_e1_corr.py` (E1.1), `run_e1_law.py` (E1.2), `run_e1_ablation.py` (E1.3), `run_e1_overhead.py` (E1.4), `make_report.py`.
- All runners: `--tiny` flag runs a toy Euler loop with a tiny random `FluxTransformer2DModel` on CPU
  (for example patch_size=1, in_channels=4, num_layers=1, num_single_layers=1, attention_head_dim=16,
  num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=32, axes_dims_rope=[4, 4, 8];
  check against the installed diffusers version), so every script is exercised end to end without downloads.
- Write results incrementally (append per run) so an interrupted job can resume.

### A7. Tests (`tests/`, all CPU)
1. RTN error at most s/2 everywhere; never clamps; at most 2^b - 1 distinct levels per group.
2. SR is unbiased: mean error near 0 over 100,000 draws.
3. SD with gamma fixed to 0 equals RTN exactly.
4. Toy telescoping: linear "network" v_i = W a_i with smoothly varying a_i; with step sizes from a shifted schedule, the sum of h_i W q_i under SD equals the full-precision sum up to the boundary and sensitivity-drift terms; the SD error is much smaller than RTN.
5. Toy law: on the same toy system, fitted slopes are about 0 (RTN), -0.5 (SR), -1 (SD).
6. `gammas_from_sigmas` correct on a hand-worked example; callback advances the index correctly.
7. Wrapper on the tiny Flux transformer: FP mode output identical to the unwrapped model; no `norm` layers wrapped; RTN changes the output; `reset()` clears buffers.
8. Correlation recorder: on synthetic leftovers with known correlation, the estimate is close.
9. End-to-end: every runner with `--tiny` writes its output files.

### A8. Finish Phase A
- `requirements.txt` for the VM (CUDA PyTorch, `diffusers transformers accelerate sentencepiece protobuf huggingface_hub datasets lpips numpy pandas matplotlib scipy pyyaml pytest tqdm pillow`), `README.md` with the Phase B commands, `.gitignore`.
- Commit and push to a private git repository.

---

## Phase B: run on the VM

### B0. Setup and selective download
```bash
git clone <private repo> && cd "Steps buy bits_CPAL"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME=/path/on/vm/big_disk/hf
huggingface-cli login            # accept the FLUX.1-dev licence on Hugging Face first
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --include "model_index.json" "scheduler/*" "transformer/*" "vae/*" \
            "text_encoder/*" "text_encoder_2/*" "tokenizer/*" "tokenizer_2/*"
nvidia-smi                       # check the CVPR job's memory use
pytest -q
```
Set `torch.cuda.set_per_process_memory_fraction(0.5)` at the top of every GPU script.
Then: `python scripts/make_prompts.py`, `python scripts/precompute_embeddings.py`, and delete the
`text_encoder_2` folder from the Hugging Face cache (about 9.5 GB).

### B1. Sanity on the real model
- FP mode reproduces the stock `FluxPipeline` latent for the same seed (relative error below 1e-6).
- SD with gamma = 0 equals RTN on one prompt.

### B2. Run E1 (512 x 512, guidance 3.5, default scheduler, seed = prompt index)
- **E1.1 Correlation (about 20 min):** RTN, b = 4, T = 50, 8 prompts. Leftover cosine for lags 1, 2, 5, 10; drift ratio; heat map.
- **E1.2 Law (about 2 to 2.5 h):** T in {8, 16, 28, 50, 100}; modes FP, RTN, SR, SD; b in {4, 3}; 32 prompts. Relative latent error, PSNR, LPIPS vs FP at the same T; slopes per mode. If the GPU is busy, run 16 prompts first, then the rest.
- **E1.3 Ablation (about 15 min):** SD vs SD_UNSCALED, b = 4, T in {28, 50}.
- **E1.4 Overhead (about 10 min):** peak extra VRAM from buffers (bf16, 8-bit, 4-bit storage) and the effect of storage bits on error; per-step time vs RTN (fake quantisation, for information).

Expected total about 3 GPU-hours, about 35 GB VRAM.

### B3. Decision table (applied automatically in `results/E1_REPORT.md`)
| Check | GO | KILL |
|---|---|---|
| E1.1 median lag-1 leftover cosine (b = 4, T = 50) | at least 0.5 | below 0.3 |
| E1.2 RTN slope | between -0.3 and +0.1 | if RTN already falls like -0.5, the premise is wrong |
| E1.2 SD slope | at most -0.6 | above -0.3 |
| E1.2 SD latent error at T = 50, b = 4 | at most 0.5 x RTN and lower LPIPS | at or above 0.8 x RTN |
| E1.3 | step-ratio gamma at least as good as unscaled | informative |

- **GO:** all GO cells. Next: E2 (W4A4 with baselines and quality metrics on FLUX and SD3.5).
- **WEAK:** SD beats RTN but only matches SR (gain comes from decorrelation, not cancellation). Discuss before continuing.
- **KILL:** correlation below 0.3, or SD error at or above 0.8 x RTN at T = 50.

Outputs: `results/e1_corr.json`, `plots/corr_heatmap.png`, `results/e1_law.csv`,
`plots/law_b4.png`, `plots/law_b3.png`, `results/e1_ablation.csv`, `results/e1_overhead.json`, `results/E1_REPORT.md`.
