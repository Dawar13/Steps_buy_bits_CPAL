# CLAUDE.md (Steps buy bits_CPAL)

Read `PROJECT_CONTEXT.md` and `E1_INSTRUCTIONS.md` before doing anything.

**Current phase: A (code only).** The laptop has about 6 GB free and no GPU. Do not download any
model, dataset, prompt set or weights here. Write all code and tests described in Phase A, run the
tests on CPU if the environment exists, then commit and push. Phase B runs on the H100 VM.

Rules:
- Never invent numbers, citations or results; mark anything unverified as (verify).
- Every reported number comes from a script in this repo with a fixed seed; log config, seed and git hash.
- Same seed per prompt across all modes and step counts; full precision is the reference.
- Do not change E1 decision thresholds without asking.
- On the VM: all downloads under `$HF_HOME`; selective FLUX download only; cap GPU memory with `torch.cuda.set_per_process_memory_fraction(0.5)` (the GPU is shared with another project).
- Reports: British spelling, plain language, no em-dashes.
