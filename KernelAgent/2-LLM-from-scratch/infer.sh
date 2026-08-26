#!/bin/bash

# 설정 변수
CHECKPOINT="checkpoint_final.pt"
PROMPT="O Romeo, Romeo! wherefore art thou Romeo?"
MAX_TOKENS=200
TEMPERATURE=0.8

echo "🚀 Starting LLM Inference Process..."
echo "Checkpoint: $CHECKPOINT"
echo "Prompt: '$PROMPT'"

# PYTHONPATH 설정
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 추론 실행
python src/generate.py \
    $CHECKPOINT \
    --prompt "$PROMPT" \
    --max_new_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE

echo "✅ Inference Completed."
