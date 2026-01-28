#!/usr/bin/env bash
set -euo pipefail
############################################
# USER SETTINGS
############################################
DATASET="znhoughton/babylm-100m-v3"
TOKENIZER_NAME="opt-babylm-100m-bpe"
BLOCK_SIZE=1024
VOCAB_SIZE=8192
# ~20M tokens per checkpoint with 262,144 tokens/step
SAVE_STEPS=80
SAVE_TOTAL_LIMIT=1  # Only keep the most recent checkpoint (which will be the final one)
WARMUP_STEPS=2000
SEED=42
############################################
# STEP 0: Train tokenizer ONCE
############################################
TOKENIZER_PATH="models/${TOKENIZER_NAME}"
if [ -d "${TOKENIZER_PATH}" ]; then
  echo "=== Tokenizer already exists at ${TOKENIZER_PATH}, skipping tokenizer training ==="
else
  echo "=== Training tokenizer ==="
  python tokenizer_and_config.py \
    --base_model facebook/opt-125m \
    --model_name ${TOKENIZER_NAME} \
    --train_file ${DATASET} \
    --from_iterator \
    --bpe \
    --vocab ${VOCAB_SIZE} \
    --hidden_size 768 \
    --attention_heads 12 \
    --layers 12 \
    --intermediate_size 3072 \
    --max_len ${BLOCK_SIZE}
fi
############################################
# FUNCTION: train one OPT model
############################################
train_opt () {
    MODEL_SIZE=$1
    BASE_MODEL=$2
    HIDDEN=$3
    HEADS=$4
    LAYERS=$5
    FFN=$6
    BATCH=$7
    GRAD_ACCUM=$8
    LR=$9
    MODEL_NAME="opt-babylm-${MODEL_SIZE}"
    MODEL_PATH="models/${MODEL_NAME}"
    RUN_DIR="runs/${MODEL_NAME}"
    echo "=== Training ${MODEL_NAME} ==="
    python tokenizer_and_config.py \
        --base_model ${BASE_MODEL} \
        --model_name ${MODEL_NAME} \
        --train_file ${DATASET} \
        --from_iterator \
        --bpe \
        --vocab ${VOCAB_SIZE} \
        --hidden_size ${HIDDEN} \
        --attention_heads ${HEADS} \
        --layers ${LAYERS} \
        --intermediate_size ${FFN} \
        --max_len ${BLOCK_SIZE}
        
    # ✅ Use torchrun for distributed training
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_autoreg.py \
        --model_type opt \
        --config_name ${MODEL_PATH} \
        --tokenizer_name ${TOKENIZER_PATH} \
        --dataset_name ${DATASET} \
        --do_train \
        --do_eval \
        --bf16 \
        --gradient_checkpointing \
        --block_size ${BLOCK_SIZE} \
        --per_device_train_batch_size ${BATCH} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LR} \
        --warmup_steps ${WARMUP_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --save_total_limit ${SAVE_TOTAL_LIMIT} \
        --save_only_model \
        --logging_steps 10 \
        --num_train_epochs 20 \
        --seed ${SEED} \
        --output_dir ${RUN_DIR} \
        --push_to_hub \
        --hub_model_id znhoughton/${MODEL_NAME}-seed${SEED} \
        --hub_strategy checkpoint \
        --ddp_find_unused_parameters False
}
############################################
# OPT-125M
# tokens/step = 1024 × 128 × 1 × 2 GPUs = 262,144
############################################
train_opt \
  125m \
  facebook/opt-125m \
  768 \
  12 \
  12 \
  3072 \
  128 \
  1 \
  3e-4
############################################
# OPT-350M  
# tokens/step = 1024 × 32 × 4 × 2 GPUs = 262,144
############################################
train_opt \
  350m \
  facebook/opt-350m \
  1024 \
  16 \
  24 \
  4096 \
  32 \
  4 \
  2e-4
############################################
# OPT-1.3B
# tokens/step = 1024 × 16 × 8 × 2 GPUs = 262,144
############################################
train_opt \
  1.3b \
  facebook/opt-1.3b \
  2048 \
  32 \
  24 \
  8192 \
  16 \
  8 \
  1e-4