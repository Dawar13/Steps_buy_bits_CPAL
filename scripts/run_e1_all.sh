#!/usr/bin/env bash
# All of E1 after the prompts and embeddings exist, ordered for the earliest go/kill signal and run in parallel
# where the GPU allows:
#   B1 sanity (also measures one worker's peak memory)
#   E1.1 correlation, interim report
#   E1.2 law: two workers at once (prompts 0-15 and 16-31) if two peaks fit in free GPU memory, else in sequence
#   interim report, E1.3 (reuses the identical SD rows of E1.2), E1.4 alone (clean timings), final report
# Every runner resumes from its result files, so rerunning this script after an interruption continues.
# Usage (inside tmux):  bash scripts/run_e1_all.sh        PARALLEL=0 forces one worker.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1
mkdir -p results/logs
LOG=results/logs/run_all.txt
step() {
  echo "$(date -u '+%F %T') START $*" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  local s=$?
  echo "$(date -u '+%F %T') END status=$s $*" | tee -a "$LOG"
  return $s
}
step python scripts/run_b1_sanity.py || { echo "B1 sanity failed; stopping" | tee -a "$LOG"; exit 1; }
step python scripts/run_e1_corr.py
step python scripts/make_report.py

PEAK=$(python -c "import json; print(json.load(open('results/b1_sanity.json')).get('peak_gb') or 99)")
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
PAR=$(python -c "print(1 if 2 * ($PEAK + 3) < $FREE / 1024 - 4 else 0)")
PAR=${PARALLEL:-$PAR}
echo "$(date -u '+%F %T') worker peak ${PEAK} GB, free ${FREE} MiB -> parallel=${PAR}" | tee -a "$LOG"
if [ "$PAR" = "1" ]; then
  step python scripts/run_e1_law.py --prompt_ids 0-15 &
  sleep 90  # stagger model loading
  step python scripts/run_e1_law.py --prompt_ids 16-31 &
  wait
else
  step python scripts/run_e1_law.py --n_prompts 16
  step python scripts/make_report.py
fi
step python scripts/run_e1_law.py   # fills anything missing (all 32 prompts)
step python scripts/make_report.py
step python scripts/run_e1_ablation.py
step python scripts/run_e1_overhead.py
step python scripts/make_report.py
echo "$(date -u '+%F %T') ALL_DONE" | tee -a "$LOG"
