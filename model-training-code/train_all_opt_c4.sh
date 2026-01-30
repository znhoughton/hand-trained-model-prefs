#!/usr/bin/env bash
set -euo pipefail

############################################
# USER SETTINGS
############################################

DATASET_NAME="allenai/c4"
DATASET_CONFIG="en"

TOKENIZER_NAME="opt-babylm-100m-bpe"  # Your existing tokenizer
BLOCK_SIZE=1024
VOCAB_SIZE=8192

# TARGET: 10M tokens per checkpoint
TOKENS_PER_CHECKPOINT=10000000

# Total training budget: 10B tokens
TOTAL_TOKENS=10000000000

SAVE_TOTAL_LIMIT=1
WARMUP_STEPS=2000
SEED=42

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
    MAX_STEPS=$((TOTAL_TOKENS / TOKENS_PER_STEP))
    SAVE_STEPS=$((TOKENS_PER_CHECKPOINT / TOKENS_PER_STEP))
    
    MODEL_NAME="opt-c4-${MODEL_SIZE}"
    MODEL_PATH="models/${MODEL_NAME}"
    RUN_DIR="runs/${MODEL_NAME}"
    
    echo "============================================================"
    echo "=== Training ${MODEL_NAME} on C4 with STREAMING ==="
    echo "=== Tokens/step: ${TOKENS_PER_STEP} ==="
    echo "=== Max steps: ${MAX_STEPS} (${TOTAL_TOKENS} tokens) ==="
    echo "=== Save every ${SAVE_STEPS} steps (${TOKENS_PER_CHECKPOINT} tokens) ==="
    echo "============================================================"
    
    # Check if checkpoint exists
    RESUME_ARG=""
    if [ -d "${RUN_DIR}" ]; then
        LATEST_CHECKPOINT=$(ls -d ${RUN_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || echo "")
        if [ -n "${LATEST_CHECKPOINT}" ]; then
            CHECKPOINT_NUM=$(basename ${LATEST_CHECKPOINT} | sed 's/checkpoint-//')
            echo "=== FOUND CHECKPOINT: ${LATEST_CHECKPOINT} ==="
            echo "=== Checkpoint step: ${CHECKPOINT_NUM} / ${MAX_STEPS} ==="
            echo "=== Steps remaining: $((MAX_STEPS - CHECKPOINT_NUM)) ==="
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
    
    # Train with STREAMING enabled
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_autoreg.py \
        --model_type opt \
        --config_name ${MODEL_PATH} \
        --tokenizer_name ${TOKENIZER_PATH} \
        --dataset_name ${DATASET_NAME} \
        --dataset_config_name ${DATASET_CONFIG} \
        --streaming \
        --do_train \
        --bf16 \
        --gradient_checkpointing \
        --block_size ${BLOCK_SIZE} \
        --per_device_train_batch_size ${BATCH} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LR} \
        --warmup_steps ${WARMUP_STEPS} \
        --max_steps ${MAX_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --save_total_limit ${SAVE_TOTAL_LIMIT} \
        --save_only_model \
        --logging_steps 10 \
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
# OPT-125M
# tokens/step = 1024 × 320 × 1 × 2 = 655,360
# max_steps = 10B / 655,360 ≈ 15,259 steps
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
# tokens/step = 1024 × 80 × 2 × 2 = 327,680
# max_steps = 10B / 327,680 ≈ 30,518 steps
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
# tokens/step = 1024 × 40 × 4 × 2 = 327,680
# max_steps = 10B / 327,680 ≈ 30,518 steps
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
