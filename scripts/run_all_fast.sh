#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_all_fast.sh [--dataset local|mbpp] [--no-split] [--mbpp-max-samples N]

Runs a faster pipeline with reduced PPO settings.
Logs are saved to results/ as a single txt file.
USAGE
}

DATASET="local"
NO_SPLIT="false"
MBPP_MAX_SAMPLES="200"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --no-split)
      NO_SPLIT="true"
      shift
      ;;
    --mbpp-max-samples)
      MBPP_MAX_SAMPLES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p results
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="results/run_fast_${DATASET}_${STAMP}.txt"

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing .venv. Please create venv and install requirements first."
  exit 1
fi

COMMON_ARGS=""
if [[ "$DATASET" == "mbpp" ]]; then
  COMMON_ARGS="--dataset mbpp --mbpp-train-split train --mbpp-eval-split validation --mbpp-max-samples ${MBPP_MAX_SAMPLES}"
else
  if [[ "$NO_SPLIT" == "true" ]]; then
    COMMON_ARGS="--no-split"
  fi
fi

{
  echo "[info] dataset=${DATASET} no_split=${NO_SPLIT} mbpp_max_samples=${MBPP_MAX_SAMPLES}"
  echo "[info] started_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
} | tee "${LOG_FILE}"

echo -e "\n[step] random baseline" | tee -a "${LOG_FILE}"
$PYTHON train.py --mode random --episodes 3 --max-steps 80 $COMMON_ARGS | tee -a "${LOG_FILE}"

echo -e "\n[step] template baseline" | tee -a "${LOG_FILE}"
$PYTHON train.py --mode template --episodes 3 --max-steps 80 $COMMON_ARGS | tee -a "${LOG_FILE}"

echo -e "\n[step] pretrain" | tee -a "${LOG_FILE}"
$PYTHON train.py --mode pretrain --epochs 20 --batch-size 4 --save-path checkpoints/pretrain_fast.pt $COMMON_ARGS | tee -a "${LOG_FILE}"

echo -e "\n[step] eval pretrain" | tee -a "${LOG_FILE}"
$PYTHON train.py --mode eval --load-path checkpoints/pretrain_fast.pt --max-gen-len 120 --max-steps 120 $COMMON_ARGS | tee -a "${LOG_FILE}"

echo -e "\n[step] ppo train (fast)" | tee -a "${LOG_FILE}"
if [[ "$DATASET" == "mbpp" ]]; then
  $PYTHON ppo_train.py --dataset mbpp --mbpp-train-split train --mbpp-eval-split validation --mbpp-max-samples "${MBPP_MAX_SAMPLES}" --updates 3 --episodes-per-update 2 --teacher-episodes 1 --max-steps 80 --max-seq-len 80 --save-path checkpoints/ppo_fast.pt | tee -a "${LOG_FILE}"
else
  $PYTHON ppo_train.py --updates 3 --episodes-per-update 2 --teacher-episodes 1 --max-steps 80 --max-seq-len 80 --save-path checkpoints/ppo_fast.pt $COMMON_ARGS | tee -a "${LOG_FILE}"
fi

echo -e "\n[step] eval ppo" | tee -a "${LOG_FILE}"
if [[ "$DATASET" == "mbpp" ]]; then
  $PYTHON mbpp_eval.py --ckpt checkpoints/ppo_fast.pt --split validation --max-samples "${MBPP_MAX_SAMPLES}" | tee -a "${LOG_FILE}"
else
  $PYTHON benchmark.py --mode model --ckpt checkpoints/ppo_fast.pt --max-gen-len 120 --max-steps 120 $COMMON_ARGS | tee -a "${LOG_FILE}"
fi

echo -e "\n[done] logs saved to ${LOG_FILE}"
