#!/usr/bin/env bash
# Memory-safe fast hybrid training on Ascend (bf16 + gradient checkpointing + DDP).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CSV_PATH="${CSV_PATH:-datasets/predictors/output_length.csv}"
MODEL_DIR="${MODEL_DIR:-models/output_length_deberta_hybrid}"
EPOCHS="${EPOCHS:-6}"
MAX_LENGTH="${MAX_LENGTH:-512}"
FREEZE_EPOCH="${FREEZE_EPOCH:-3}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-2}"

: "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export ASCEND_RT_VISIBLE_DEVICES
export ASCEND_DEVICES="${ASCEND_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
IFS=',' read -ra _IDS <<< "${ASCEND_RT_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#_IDS[@]}}"

EXTRA_ARGS=("$@")

_common=(
  -m predictors.output_length.train_output_length_deberta
  --bucket
  --csv-path "$CSV_PATH"
  --model-dir "$MODEL_DIR"
  --head-tail-truncation
  --epochs "$EPOCHS"
  --max-length "$MAX_LENGTH"
  --freeze-backbone-after-epoch "$FREEZE_EPOCH"
  --batch-size "$BATCH"
  --gradient-accumulation-steps "$ACCUM"
  --bf16
  --gradient-checkpointing
)

if [[ "$NPROC" -gt 1 ]]; then
  echo "Hybrid DDP: nproc=$NPROC batch=$BATCH accum=$ACCUM effective=$((BATCH * NPROC * ACCUM))"
  exec torchrun --nproc_per_node="$NPROC" "${_common[@]}" "${EXTRA_ARGS[@]}"
fi

echo "Hybrid single-NPU: batch=$BATCH accum=$ACCUM"
exec python "${_common[@]}" --ascend-devices "${_IDS[0]}" "${EXTRA_ARGS[@]}"
