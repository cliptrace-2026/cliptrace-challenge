#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LR="${LR:-0.1}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.2}"
THRES="${THRES:-0.98}"
PL1_THRESHOLD="${PL1_THRESHOLD:-0.010}"
MASK_INIT="${MASK_INIT:-rand}"
ENCODER_USAGE_INFO="${ENCODER_USAGE_INFO:-HF_CLIP}"
STOP_ON_BACKDOOR_PL1="${STOP_ON_BACKDOOR_PL1:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
RESET_SUBMISSION="${RESET_SUBMISSION:-1}"
SKIP_MODEL_ID_CHECK="${SKIP_MODEL_ID_CHECK:-0}"

MODELS_ROOT="${MODELS_ROOT:-$PROJECT_ROOT/cliptrace-2026-models/models}"
IMAGENET_ROOT="${IMAGENET_ROOT:-$PROJECT_ROOT/Baseline/data/imagenet}"
TRIGGER_FILE="${TRIGGER_FILE:-$PROJECT_ROOT/Baseline/trigger/trigger_clip_l.npz}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/outputs/baseline-artifacts}"
METRICS_FILE="${METRICS_FILE:-$PROJECT_ROOT/outputs/baseline_metrics.jsonl}"
SUBMISSION_DIR="${SUBMISSION_DIR:-$PROJECT_ROOT/submission}"
SUBMISSION_ZIP="${SUBMISSION_ZIP:-$PROJECT_ROOT/submission.zip}"

if [[ ! -d "$MODELS_ROOT" ]]; then
    echo "models directory not found: $MODELS_ROOT" >&2
    exit 1
fi

if [[ ! -d "$IMAGENET_ROOT" ]]; then
    echo "ImageNet directory not found: $IMAGENET_ROOT" >&2
    exit 1
fi

if [[ ! -f "$TRIGGER_FILE" ]]; then
    echo "trigger file not found: $TRIGGER_FILE" >&2
    exit 1
fi

mapfile -t MODEL_CONFIGS < <(find "$MODELS_ROOT" -type f -name config.json | sort)
if [[ "${#MODEL_CONFIGS[@]}" -eq 0 ]]; then
    echo "no model config.json found under: $MODELS_ROOT" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR" "$(dirname "$METRICS_FILE")" "$SUBMISSION_DIR/embeddings"
if [[ "$RESET_SUBMISSION" == "1" ]]; then
    printf '{\n  "version": "1.0",\n  "predictions": []\n}\n' > "$SUBMISSION_DIR/predictions.json"
fi

echo "models: ${#MODEL_CONFIGS[@]}"
echo "models root: $MODELS_ROOT"
echo "submission dir: $SUBMISSION_DIR"
echo "submission zip: $SUBMISSION_ZIP"
echo "metrics file: $METRICS_FILE"

if [[ "$STOP_ON_BACKDOOR_PL1" == "1" ]]; then
    stop_on_backdoor_pl1_arg="--stop_on_backdoor_pl1"
else
    stop_on_backdoor_pl1_arg="--no-stop_on_backdoor_pl1"
fi

failed=0
total="${#MODEL_CONFIGS[@]}"
for index in "${!MODEL_CONFIGS[@]}"; do
    model_dir="$(dirname "${MODEL_CONFIGS[$index]}")"
    model_id="$(basename "$model_dir")"
    current=$((index + 1))
    echo "[$current/$total] running $model_id"

    if "$PYTHON_BIN" "$PROJECT_ROOT/Baseline/main.py" \
        --model_id "$model_id" \
        --encoder_path "$model_dir" \
        --gpu "$GPU" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --sample_ratio "$SAMPLE_RATIO" \
        --thres "$THRES" \
        --pl1_threshold "$PL1_THRESHOLD" \
        --mask_init "$MASK_INIT" \
        "$stop_on_backdoor_pl1_arg" \
        --encoder_usage_info "$ENCODER_USAGE_INFO" \
        --imagenet_root "$IMAGENET_ROOT" \
        --trigger_file "$TRIGGER_FILE" \
        --result_dir "$RESULT_DIR" \
        --submission_dir "$SUBMISSION_DIR" \
        --metrics_file "$METRICS_FILE"; then
        echo "[$current/$total] done $model_id"
    else
        failed=$((failed + 1))
        echo "[$current/$total] failed $model_id" >&2
        if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
            exit 1
        fi
    fi
done

if [[ "$failed" -gt 0 ]]; then
    echo "$failed model(s) failed" >&2
    exit 1
fi

pack_args=(
    "$PROJECT_ROOT/create_submission.py"
    --submission-dir "$SUBMISSION_DIR"
    --output "$SUBMISSION_ZIP"
    --models-dir "$MODELS_ROOT"
)
if [[ "$SKIP_MODEL_ID_CHECK" == "1" ]]; then
    pack_args+=(--skip-model-id-check)
fi

if ! "$PYTHON_BIN" "${pack_args[@]}"; then
    echo "failed to create a valid submission zip" >&2
    exit 1
fi
echo "all models finished; submit this file: $SUBMISSION_ZIP"
