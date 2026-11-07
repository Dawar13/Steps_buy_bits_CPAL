#!/usr/bin/env bash
# All of E1 after the prompts and embeddings exist, ordered for the earliest go/kill signal:
# B1 sanity, E1.1, E1.2 on 16 prompts, interim report, E1.3, E1.4, E1.2 on all 32, final report.
# Every runner resumes from its result files, so rerunning this script after an interruption continues.
# Usage (inside tmux):  bash scripts/run_e1_all.sh
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
step python scripts/run_e1_law.py --n_prompts 16
step python scripts/make_report.py
step python scripts/run_e1_ablation.py
step python scripts/run_e1_overhead.py
step python scripts/run_e1_law.py
step python scripts/make_report.py
echo "$(date -u '+%F %T') ALL_DONE" | tee -a "$LOG"
