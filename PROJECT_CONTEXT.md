# PROJECT CONTEXT: Steps Buy Bits (CPAL 2027)

## 1. Summary
**Title:** Steps Buy Bits: Carrying Rounding Leftovers Across Sampling Steps Makes Low-Bit Image and Video Generators Improve With More Steps

When a diffusion or flow model runs with low-bit activations, its rounding errors repeat almost
identically from one sampling step to the next, because neighbouring steps see nearly the same
numbers. Repeating errors pile up instead of cancelling, so a 4-bit generator does not get closer to
its full-precision self when given more steps. We (1) measure this temporal correlation in real
generators (FLUX, SD3.5, Wan), overturning the independence assumption used by recent methods,
(2) show an error-versus-steps law with three regimes (plain rounding flat, random rounding slope
-1/2, leftover carry slope -1), and (3) give a training-free, sampler-matched leftover carry
(sigma-delta noise shaping across steps) that makes low-bit errors cancel.

## 2. Venue and dates
- CPAL 2027, Tokyo, 23 to 26 March 2027. Proceedings Track: 5 December 2026 AoE, PMLR, 9 pages. Primary target.
- Recent Spotlight Track: 18 January 2027. Fallback, or the extended version (more model families).
- Internal milestones: E1 go/kill by 9 Nov; E2 and E3 by 22 Nov; writing 22 Nov to 3 Dec; submit 5 Dec.

## 3. Background in plain words
- **Sampling.** A flow model starts from noise and takes T small steps; each step the network
  predicts a direction and the sampler moves by step size h_i. The final image is the start plus the
  sum of all moves.
- **Quantisation.** Running layers on 4-bit or 3-bit numbers: faster and smaller, but every number is
  rounded to the nearest allowed level. The rounding leftover is the error.
- **Correlated errors.** Neighbouring steps see almost the same activations, so their leftovers are
  almost the same. Like a shop rounding every bill the same way, the errors add up.
- **Sigma-delta (noise shaping).** Carry the leftover into the next rounding. Over many steps the
  leftovers cancel, as in audio converters.
- **Step-size matching.** Steps are uneven, so the carried leftover is scaled by the ratio of step sizes.

## 4. The mathematics (load-bearing)
Execution order i = 1..T, noise levels sigma_1 > ... > sigma_(T+1) = 0, step size h_i = sigma_i - sigma_(i+1).
Euler flow sampler: x_(i+1) = x_i - h_i v_theta(x_i, sigma_i).

**4.1 Error propagation.** For one quantised layer input a_i with error e_i, to first order the
final error is delta = sum_i h_i G_i e_i, where G_i combines the network's sensitivity to that
input at step i and the propagation of the sampler from step i to the end (the same linear
recurrence TCEC writes as delta_(t-1) = A_t delta_t + B_t eps_t).

**4.2 Three regimes.**
- Plain rounding with coherent errors (e_i nearly equal across steps): delta ~ (sum h_i G_i) e = O(1). Flat in T.
- Independent zero-mean errors (stochastic rounding): delta = O(T^(-1/2)).
- Leftover carry: a'_i = a_i + gamma_i r_(i-1), q_i = round(a'_i), r_i = a'_i - q_i, so
  e_i = gamma_i r_(i-1) - r_i. With **gamma_i = h_(i-1) / h_i** (previous step size over current), summation by parts gives
  sum_i h_i G_i e_i = sum_(i=1..T-1) h_i (G_(i+1) - G_i) r_i - h_T G_T r_T.
  The leftovers enter only through the change in sensitivity between neighbouring steps plus one
  boundary term. With |r_i| <= s/2 and smooth G (the same smoothness that makes feature caching
  work), delta = O(1/T): slope -1.
- Equivalently: doubling the steps halves the error, worth about one extra bit (hypothesis to be measured).

**4.3 Relation to general theory.** "When Can Depth Replace Precision?" (arXiv 2607.23390, July 2026)
proves the general principle that carrying increments lets iterations replace precision in quantised
iterative systems. Our contribution is the sampler-specific version (step-size matching, the
sensitivity-drift term), the first measurement in real generators, and the method at scale.
Cite it plainly; do not present the general telescoping idea as ours.

## 5. Hypotheses
- **H1 (coherence):** adjacent-step rounding leftovers are strongly correlated (median cosine at least 0.5 at 4-bit, 50 steps).
- **H2 (law):** error vs steps is flat for plain rounding, slope about -1/2 for stochastic rounding, about -1 for the carry.
- **H3 (useful):** at 28 to 50 steps the carry at least halves the error of plain rounding at 4-bit activations, and makes 3-bit activations usable.
- **H4 (step matching):** gamma = step-size ratio beats an unscaled carry on uneven schedules.
- **H5 (cheap):** the leftover can be stored in about 4 to 8 bits with little loss; overhead is one buffer per quantised layer input.

## 6. Prior work, verified facts and our delta
| Work | Verified fact | Our delta |
|---|---|---|
| TCEC, "Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion Models", ICML 2026, arXiv 2508.12094 | Section 3.1: the per-step error "only relies on the module at iteration t and is independent of the others". Error recurrence delta_(t-1) = A_t delta_t + B_t eps_t. Built on SVDQuant: SDXL W4A4 PSNR 20.7 -> 21.9, FID 20.6 -> 18.1; OpenSora W4A8 VBench imaging quality 64.97 vs 61.07 (ViDiT-Q); under 0.5% latency | We measure that the independence assumption fails and cancel errors at the quantiser itself, with no calibration of correction coefficients |
| MoDiff, ICML 2025, arXiv 2506.22463 | Quantises the change of each layer input between steps (closed loop), caching previous quantised input and output. Tested on DDIM CIFAR-10, LDM bedrooms/churches, SD v1.4, DiT-XL/2, SDXL-Turbo; W4/W8, A2 to A8; about 3.8 MB extra at batch 1 on small models | MoDiff makes each step's error smaller; its errors do not cancel across steps. We carry leftovers so they cancel; one buffer instead of two; tested on FLUX, SD3.5, Wan. MoDiff is a baseline and can be combined |
| When Can Depth Replace Precision?, arXiv 2607.23390 | General theory of trading iterations for precision with increment error feedback; no diffusion experiments (verify details) | Sampler-specific theorem, measurement, method and results |
| QuAKE, arXiv 2609.21407 | Assumes temporally independent quantisation noise (verify wording) | Our measurement tests this assumption |
| Q-Drift, arXiv 2603.18095 | Corrects the sampler for quantisation drift (verify mechanism) | Baseline |
| TASQ, arXiv 2608.03057 | Learned per-stage LSB masks on a shared 8-bit buffer, distillation; PixArt-Sigma, SANA, SDXL-Turbo | Training-free; different mechanism |
| Guidance-Aware Mixed Precision, arXiv 2607.08241; "Why Classifier-Free Guidance Is Fragile Under Quantization" (NeurIPS 2026) | CFG-specific quantisation | Out of scope: we do not claim CFG contributions |
| SVDQuant (verify ID 2411.05007), ViDiT-Q (verify ID 2406.02540), PTQD (verify ID 2305.10657) | Standard diffusion PTQ baselines | Our carry is orthogonal and can sit on top of their quantisers |
| Sigma-delta modulation; noise feedback coding (Atal and Schroeder, 1979, verify) | Classical noise shaping | Brought into the sampling loop of modern generators |

**Open statement (re-check on OpenReview ICLR 2027 before submission):** as of 7 November 2026,
under the logged queries, no diffusion or flow quantisation method carries the exact activation
rounding leftover into the next sampling step's quantiser with step-size matching, and no paper has
measured the temporal correlation of rounding leftovers or the error-versus-steps slopes in FLUX,
SD3.5 or Wan.

## 7. Experiment plan (total about 8 to 10 H100-hours)
- **E1 (go/kill, about 3 h):** FLUX.1-dev, 512 x 512, activation-only quantisation. Correlation, law, step-matching ablation, overhead. See E1_INSTRUCTIONS.md.
- **E2 (about 3 h):** W4A4 and W4A3 on FLUX.1-dev and SD3.5-Medium at 1024 x 1024 with baselines (RTN, SVDQuant-style low-rank plus 4-bit, MoDiff, TCEC, Q-Drift where code exists) and quality metrics (FID against full-precision images, ImageReward, GenEval).
- **E3 (about 2 h):** Wan2.1-1.3B at 480p (VBench subset). Memory is the limit; carry only in the most sensitive layers if needed.
- **E4 (about 1 h):** overhead: leftover storage at 8 and 4 bits; layer subsets; measured latency.
- **E5 (optional):** alternate weight rounding across steps (floor/ceil schedule) to cancel static weight error.
- **E6 (spotlight version):** other iterative models (a diffusion language model such as LLaDA, or RAFT-Stereo) to show generality.

## 8. Metrics
- Law: relative final-latent error vs full precision at the same steps; PSNR and LPIPS of decoded images; fitted log-log slopes.
- Quality: FID, ImageReward, GenEval (images); VBench (video).
- Cost: extra VRAM from leftover buffers, latency.

## 9. Risks and mitigations
- Leftovers not correlated (H1 fails): kill in E1.
- Nonlinear effects swamp the cancellation at 3-bit: report 4-bit as the main result.
- Memory at high resolution and video: store leftovers in low bits; carry only sensitive layers; Wan 1.3B at 480p.
- Theory overlap with 2607.23390: cite clearly; our novelty is the measurement, the sampler-specific version and the results.
- Collision: re-run the novelty search on OpenReview before submission.

## 10. Repository layout
```
sbb/               library (quant, context, wrap, corr, metrics, runners, report)
scripts/           entry points per experiment
tests/             pytest (CPU, tiny random Flux transformer, toy linear systems)
configs/           yaml configs
results/ plots/    outputs (git-ignore raw data)
```

## 11. Environment and resources
- Laptop: about 6 GB free. Never download models or datasets locally. Local work is code and CPU tests only.
- VM: one H100 80 GB, shared with a CVPR project. E1 needs about 35 GB VRAM at batch 1; cap with
  `torch.cuda.set_per_process_memory_fraction(0.5)` and check `nvidia-smi` first.
- All downloads under `$HF_HOME` on the VM disk; selective FLUX download (about 34 GB); delete `text_encoder_2` after caching prompt embeddings.
- Code moves laptop -> VM through a private git repository.

## 12. Rules for Claude Code
- Never invent numbers, citations or results. Mark anything not verified as (verify).
- Every reported number comes from a script in this repo with a fixed seed; log config, seed and git hash.
- Same seed per prompt across all modes and step counts; full-precision results are the reference.
- Do not change the E1 decision thresholds without asking.
- Reports: British spelling, plain language, no em-dashes.
