# E1 notes (7 to 8 Nov 2026)

Observations that the decision table in `E1_REPORT.md` does not show. Numbers come from `E1_REPORT.md` and
`e1_checks.json` (written by `scripts/e1_extra_checks.py`); interpretations are marked (verify).

## Outcome

KILL on two independent rules: the median lag-1 leftover cosine is 0.270 (KILL below 0.3), and SD's error at
T = 50, b = 4 is 0.905 x RTN (KILL at or above 0.8). The SD slope cell is also KILL (0.12, needs at most -0.6).

## What the data show

- **Errors grow with steps instead of staying flat.** At b = 4 the median rel_err of RTN rises from 0.218 (T = 8)
  to 0.341 (T = 100); its slope is +0.17. SR (+0.10) and SD (+0.12) rise too. From T = 16 to T = 100 the error
  rises for 21 of 32 prompts under RTN, 21 under SR and 20 under SD.
- **The errors are large.** At T = 50, b = 4, every mode is near rel_err 0.32 to 0.34 (PSNR about 19 dB, LPIPS about
  0.19 to 0.22); at b = 3 it is about 0.52 to 0.63. With errors this size the first-order theory (small errors,
  smooth sensitivity) is not expected to hold (verify).
- **Leftovers are only weakly correlated between steps.** Neighbouring steps move the activations by a median
  0.39 quantisation steps (drift ratio), enough to re-randomise the rounding. Double-block inputs keep high
  correlation (text stream up to 0.78); single-block inputs are near 0.12 to 0.14.
- **The carry helps a little, on most prompts, by about as much as random rounding.** SD is below RTN at T = 50
  on 23 of 31 prompts (median ratio 0.905, range 0.30 to 1.67) and below SR on 21 of 31 (median 0.972).
- **Step-size matching makes no difference** (E1.3): unscaled / SD = 1.009 at T = 28 and 0.997 at T = 50.
- **Overhead** (E1.4): leftover buffers take 6.8 GB in fp32, 3.4 GB in bf16, 1.8 GB in int8 and 0.9 GB in int4
  (input sharing saves about 30%); fake-quantised steps take 1.23 to 1.68 x the RTN step time.

## Anomalies

- **Two SD runs ended in NaN**: prompt 4 at T = 100 and prompt 9 at T = 50 (both b = 4). They are left out of
  medians, slopes and ratios. The unscaled carry on the same prompt and T finished normally, and so did RTN and
  SR. Not diagnosed yet: the GPU was full with another project. To diagnose:
  `python scripts/diagnose_nan.py --prompt_ids 9 --steps 50 --mode SD --bits 4`.
  Including them as failures would not change the verdict.
- **Step 13 of 50**: in every prompt the lag-1 cosine drops to about 0.17 and the drift ratio doubles (0.61
  against about 0.33 at steps 12 and 14). The activations genuinely jump at that step. BF16 rounding of the
  timestep was checked and does not explain it. Cause unknown.

## Checks that passed

- B1: FP mode equals the stock FluxPipeline latent bit for bit; SD with gamma = 0 equals RTN bit for bit.
- E1.4 reran SD (fp32 leftovers) for prompts 0 to 3 alone on the GPU; the errors match E1.2's parallel runs to
  6e-17, so results do not depend on the second worker.
- All 960 E1.2 rows and 128 E1.3 rows come from commit e8398df with a clean tree.

## Where everything is

- Committed: `results/` summaries (CSV, JSON, report, notes, per-run metadata), `plots/`, `prompts.txt`.
- Not committed (git-ignored raw data), kept on the laptop and on the A100 under `~/sbb/repo`:
  `results/raw/corr/` (per-prompt cosine and drift arrays), `results/logs/`, and `cache/fp/` (the 160
  full-precision reference latents, 32 prompts x 5 step counts, 41 MB).
