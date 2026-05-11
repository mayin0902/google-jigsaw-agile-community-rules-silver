#!/usr/bin/env bash
set -euo pipefail

WHEEL_DIR="${1:-/kaggle/input/jigsaw-packages2/whls/}"

uv pip install --system --no-index --find-links="${WHEEL_DIR}" \
  'trl==0.21.0' 'optimum==1.27.0' 'auto-gptq==0.7.1' \
  'bitsandbytes==0.46.1' 'logits-processor-zoo==0.2.1' 'vllm==0.10.0'
uv pip install --system --no-index --find-links="${WHEEL_DIR}" 'deepspeed==0.17.4' -q
uv pip install --system --no-index --find-links="${WHEEL_DIR}" 'triton==3.2.0'
uv pip install --system --no-index --find-links="${WHEEL_DIR}" 'clean-text'
uv pip install --system --no-index -U --no-deps --find-links="${WHEEL_DIR}" \
  'peft' 'accelerate' 'datasets'
