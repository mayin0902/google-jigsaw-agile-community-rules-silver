#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

accelerate launch --config_file configs/accelerate_zero2.yaml scripts/train_qwen_lora.py
python scripts/infer_qwen_lora.py --num-gpus "${JIGSAW_NUM_GPUS:-2}"
python scripts/train_infer_gte.py
python scripts/train_infer_deberta.py
python scripts/blend_submissions.py
