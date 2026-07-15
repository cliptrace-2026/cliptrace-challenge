#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# CLIPTrace 2026 - DECREE-style detection and target-recovery baseline
# ============================================================================

PHASE=${PHASE:-development}
MODELS_DIR=${MODELS_DIR:-"../resources/model-repository/models/${PHASE}"}
DATA_DIR=${DATA_DIR:-"../resources/data/imagenet"}
SUBMISSION_DIR=${SUBMISSION_DIR:-"../submission"}

DEVICE=${DEVICE:-auto}
EPOCHS=${EPOCHS:-100}
MAX_SAMPLES=${MAX_SAMPLES:-785}
BATCH_SIZE=${BATCH_SIZE:-12}
LEARNING_RATE=${LEARNING_RATE:-0.1}
SEED=${SEED:-42}
MODEL_ID=${MODEL_ID:-}

echo "============================================================"
echo "CLIPTrace 2026 baseline"
echo "Models:     ${MODELS_DIR}"
echo "Data:       ${DATA_DIR}"
echo "Submission: ${SUBMISSION_DIR}"
echo "Device:     ${DEVICE}"
echo "============================================================"

EXTRA_ARGS=()
if [[ -n "${MODEL_ID}" ]]; then
  EXTRA_ARGS+=(--model-id "${MODEL_ID}")
fi

python scripts/baseline_decree.py \
  --models-dir "${MODELS_DIR}" \
  --data-dir "${DATA_DIR}" \
  --submission-dir "${SUBMISSION_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LEARNING_RATE}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

echo "Baseline completed. Results are in ${SUBMISSION_DIR}."

