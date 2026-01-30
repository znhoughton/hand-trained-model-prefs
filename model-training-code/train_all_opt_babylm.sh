#!/usr/bin/env bash
set -euo pipefail
############################################
# USER SETTINGS
############################################
DATASET="znhoughton/babylm-100m-v3"
TOKENIZER_NAME="opt-babylm-100m-bpe"
BLOCK_SIZE=1024
VOCAB_SIZE=8192
# TARGET: 10M tokens per checkpoint
TOKENS_PER_CHECKPOINT=10000000
SAVE_TOTAL_LIMIT=1
WARMUP_STEPS=2000
SEED=42
############################################
# STEP 0: Train tokenizer ONCE
############################################
TOKENIZER_PATH="models/${TOKENIZER_NAME}"
if [ -d "${TOKENIZER_PATH}" ]; then
  echo "=== Tokenizer already exists at ${TOKENIZER_PATH}, skipping ==="
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
    
    # Calculate tokens per step and save_steps
    TOKENS_PER_STEP=$((BLOCK_SIZE * BATCH * GRAD_ACCUM * 2))
    SAVE_STEPS=$((TOKENS_PER_CHECKPOINT / TOKENS_PER_STEP))
    
    MODEL_NAME="opt-babylm-${MODEL_SIZE}"
    MODEL_PATH="models/${MODEL_NAME}"
    RUN_DIR="runs/${MODEL_NAME}"
    
    echo "============================================================"
    echo "=== Training ${MODEL_NAME} ==="
    echo "=== Tokens/step: ${TOKENS_PER_STEP} ==="
    echo "=== Save every ${SAVE_STEPS} steps (${TOKENS_PER_CHECKPOINT} tokens) ==="
    echo "============================================================"
    
    # Check if checkpoint exists
    RESUME_ARG=""
    if [ -d "${RUN_DIR}" ]; then
        LATEST_CHECKPOINT=$(ls -d ${RUN_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || echo "")
        if [ -n "${LATEST_CHECKPOINT}" ]; then
            CHECKPOINT_NUM=$(basename ${LATEST_CHECKPOINT} | sed 's/checkpoint-//')
            echo "=== FOUND CHECKPOINT: ${LATEST_CHECKPOINT} ==="
            echo "=== Checkpoint step: ${CHECKPOINT_NUM} ==="
            echo "=== RESUMING FROM CHECKPOINT ==="
            RESUME_ARG="--resume_from_checkpoint ${LATEST_CHECKPOINT}"
        else
            echo "=== No checkpoint found, starting from scratch ==="
        fi
    else
        echo "=== No run directory found, starting from scratch ==="
    fi
    
    # Build config only if not resuming (avoid overwriting)
    if [ -z "${RESUME_ARG}" ]; then
        echo "=== Creating model config ==="
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
    else
        echo "=== Skipping config creation (resuming from checkpoint) ==="
    fi
    
    # Train
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
        ${RESUME_ARG}
        #--push_to_hub \
        #--hub_model_id znhoughton/${MODEL_NAME}-seed${SEED} \
        #--hub_strategy checkpoint \
        #--ddp_find_unused_parameters False
    
    echo "=== Finished training ${MODEL_NAME} ==="
    # IMPORTANT: free disk before next model
    echo "=== Deleting local run directory ${RUN_DIR} ==="
    rm -rf "${RUN_DIR}"
}
############################################
# OPT-125M
# OLD: tokens/step = 1024 × 256 × 1 × 2 = 524,288
# NEW: tokens/step = 1024 × 320 × 1 × 2 = 655,360 (+25% throughput)
# save_steps = 10M / 655,360 ≈ 15 steps
############################################
train_opt \
  125m \
  facebook/opt-125m \
  768 \
  12 \
  12 \
  3072 \
  320 \
  1 \
  3e-4
############################################
# OPT-350M
# OLD: tokens/step = 1024 × 64 × 2 × 2 = 262,144
# NEW: tokens/step = 1024 × 80 × 2 × 2 = 327,680 (+25% throughput)
# save_steps = 10M / 327,680 ≈ 30 steps
############################################
train_opt \
  350m \
  facebook/opt-350m \
  1024 \
  16 \
  24 \
  4096 \
  80 \
  2 \
  2e-4
############################################
# OPT-1.3B
# OLD: tokens/step = 1024 × 32 × 4 × 2 = 262,144
# NEW: tokens/step = 1024 × 40 × 4 × 2 = 327,680 (+25% throughput)
# save_steps = 10M / 327,680 ≈ 30 steps
############################################
train_opt \
  1.3b \
  facebook/opt-1.3b \
  2048 \
  32 \
  24 \
  8192 \
  40 \
  4 \
  1e-4
