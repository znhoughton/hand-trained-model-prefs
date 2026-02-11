#!/usr/bin/env bash
set -euo pipefail
source /opt/modeling/zhoughton/envs/opt-model-training/bin/activate
############################################
# USER SETTINGS
############################################

DATASET_NAME="znhoughton/c4-subset-10B-tokens"
TOKENIZER_NAME="opt-babylm-100m-bpe"  # Your existing tokenizer
BLOCK_SIZE=1024
VOCAB_SIZE=8192

# TARGET: 10M tokens per checkpoint
TOKENS_PER_CHECKPOINT=10000000


SAVE_TOTAL_LIMIT=1
WARMUP_STEPS=4000
SEED=964
TOKENIZER_PATH="models/${TOKENIZER_NAME}"

# Verify tokenizer exists
if [ ! -d "${TOKENIZER_PATH}" ]; then
  echo "ERROR: Tokenizer not found at ${TOKENIZER_PATH}"
  echo "Please train the tokenizer first or check the path."
  exit 1
fi

echo "=== Using existing tokenizer at ${TOKENIZER_PATH} ==="

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
    
    # Calculate tokens per step, max steps, and save frequency
    TOKENS_PER_STEP=$((BLOCK_SIZE * BATCH * GRAD_ACCUM * 2))
    SAVE_STEPS=$((TOKENS_PER_CHECKPOINT / TOKENS_PER_STEP))
    TOTAL_TOKENS=10000000000
    MAX_STEPS=$((TOTAL_TOKENS / TOKENS_PER_STEP))
    MODEL_NAME="opt-c4-${MODEL_SIZE}"
    MODEL_PATH="models/${MODEL_NAME}"
    RUN_DIR="runs/${MODEL_NAME}_${SEED}"
    
    echo "============================================================"
    echo "=== Training ${MODEL_NAME} on C4 subset (10B tokens)"
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
            echo "=== RESUMING FROM CHECKPOINT ==="
            RESUME_ARG="--resume_from_checkpoint ${LATEST_CHECKPOINT}"
        else
            echo "=== No checkpoint found, starting from scratch ==="
        fi
    else
        echo "=== No run directory found, starting from scratch ==="
    fi
    
    # Create config only if not resuming (avoid overwriting)
    if [ -z "${RESUME_ARG}" ]; then
        echo "=== Creating model config ==="
        python create_config_only.py \
            --base_model ${BASE_MODEL} \
            --model_name ${MODEL_NAME} \
            --source_tokenizer ${TOKENIZER_PATH} \
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
        --dataset_name ${DATASET_NAME} \
        --max_steps ${MAX_STEPS} \
        --do_train \
        --streaming \
        --preprocessing_num_workers 8 \
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
        --report_to tensorboard \
        --seed ${SEED} \
        --output_dir ${RUN_DIR} \
        ${RESUME_ARG} \
        --push_to_hub \
        --hub_model_id znhoughton/${MODEL_NAME}-seed${SEED} \
        --hub_strategy checkpoint \
        --ddp_find_unused_parameters False
    
    echo "=== Finished training ${MODEL_NAME} ==="
    
    # Free disk before next model
    echo "=== Deleting local run directory ${RUN_DIR} ==="
    rm -rf "${RUN_DIR}"
}

############################################
# OPT-125M - OPTIMIZED FOR 2x H100
# Target: Max batch size while maintaining speed
# tokens/step = 1024 × 512 × 1 × 2 = 1,048,576
# save_steps = 10M / 1,048,576 ≈ 10 steps
############################################
train_opt \
  125m \
  facebook/opt-125m \
  768 \
  12 \
  12 \
  3072 \
  400 \
  1 \
  3e-4

############################################
# OPT-350M - OPTIMIZED FOR 2x H100
# tokens/step = 1024 × 384 × 2 × 2 = 1,572,864
# save_steps = 10M / 1,572,864 ≈ 6 steps
############################################
train_opt \
  350m \
  facebook/opt-350m \
  1024 \
  16 \
  24 \
  4096 \
  300 \
  2 \
  1e-4

############################################
# OPT-1.3B - OPTIMIZED FOR 2x H100
# tokens/step = 1024 × 256 × 4 × 2 = 2,097,152
# save_steps = 10M / 2,097,152 ≈ 5 steps
############################################
train_opt \
  1.3b \
  facebook/opt-1.3b \
  2048 \
  32 \
  24 \
  8192 \
  100 \
  4 \
  1e-4