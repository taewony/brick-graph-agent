#!/bin/bash

# 설정 변수
DATA_PATH="data/shakespeare.txt"
MAX_STEPS=5000

echo "🚀 Starting LLM Training Process..."
echo "Data: $DATA_PATH"
echo "Max Steps: $MAX_STEPS"

# PYTHONPATH 설정 (src 디렉토리의 모듈을 찾을 수 있도록)
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 훈련 실행 (batch_size 인자 제거, train.py 기본값 사용)
python src/train.py \
    $DATA_PATH \
    --max_steps $MAX_STEPS

echo "✅ Training Completed. Checkpoints saved in current directory."
