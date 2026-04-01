#!/usr/bin/env bash
# run_ns_surprisal.sh
# ====================
# Run Oh & Schuler's get_llm_surprisal.py for all models on the
# Natural Stories Corpus, then parse outputs into item/zone CSVs.
#
# Uses their code unchanged — no modifications needed.
# Our custom OPT-based models (opt-babylm-*, opt-c4-*) are handled
# automatically because "opt" appears in the model name.
#
# Usage:
#   bash run_ns_surprisal.sh               # run all models
#   bash run_ns_surprisal.sh --recompute   # redo even if output exists
#
# Prerequisites:
#   - get_llm_surprisal.py from byungdoh/llm_surprisal in this directory
#     (or set LLM_SURP_SCRIPT below)
#   - natural_stories/ns.sentitems prepared by prepare_ns_input.py
#   - pip install torch transformers accelerate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_SURP_SCRIPT="${SCRIPT_DIR}/get_llm_surprisal.py"
SENTITEMS="${SCRIPT_DIR}/natural_stories/ns.sentitems"
SURP_DIR="${SCRIPT_DIR}/ns_surprisal"
RAW_DIR="${SURP_DIR}/raw"
PARSE_SCRIPT="${SCRIPT_DIR}/parse_surprisal_output.py"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_ns_input.py"

RECOMPUTE=0
if [[ "${1:-}" == "--recompute" ]]; then
    RECOMPUTE=1
fi

mkdir -p "${RAW_DIR}"

# ── Check prerequisites ───────────────────────────────────────────────────────
if [[ ! -f "${LLM_SURP_SCRIPT}" ]]; then
    echo "ERROR: get_llm_surprisal.py not found at ${LLM_SURP_SCRIPT}"
    echo "Copy it from: https://github.com/byungdoh/llm_surprisal/blob/eacl24/get_llm_surprisal.py"
    exit 1
fi

# Prepare .sentitems input if needed
if [[ ! -f "${SENTITEMS}" ]]; then
    echo "Preparing sentitems input..."
    python "${PREPARE_SCRIPT}"
fi

# ── Model list: model_id | family | params [| revision] ──────────────────────
# Format: "model_id|family|params"           (final checkpoint, no revision)
#         "model_id|family|params|revision"  (specific training checkpoint)
MODELS=(
    # BabyLM — final checkpoint
    "znhoughton/opt-babylm-125m-64eps-seed964|BabyLM|125M"
    "znhoughton/opt-babylm-350m-64eps-seed964|BabyLM|350M"
    "znhoughton/opt-babylm-1.3b-64eps-seed964|BabyLM|1300M"
    # BabyLM — early checkpoint (~1/3 through training, step index 6/19)
    "znhoughton/opt-babylm-125m-64eps-seed964|BabyLM (early)|125M|step-144"
    "znhoughton/opt-babylm-350m-64eps-seed964|BabyLM (early)|350M|step-96"
    "znhoughton/opt-babylm-1.3b-64eps-seed964|BabyLM (early)|1300M|step-144"
    # BabyLM — mid checkpoint (~1/2 through training, step index 9/19)
    "znhoughton/opt-babylm-125m-64eps-seed964|BabyLM (mid)|125M|step-432"
    "znhoughton/opt-babylm-350m-64eps-seed964|BabyLM (mid)|350M|step-288"
    "znhoughton/opt-babylm-1.3b-64eps-seed964|BabyLM (mid)|1300M|step-420"
    # C4 — final checkpoint
    "znhoughton/opt-c4-125m-seed964|C4|125M"
    "znhoughton/opt-c4-350m-seed964|C4|350M"
    "znhoughton/opt-c4-1.3b-seed964|C4|1300M"
    # C4 — early checkpoint (~1/3 through training, step index 6/19)
    "znhoughton/opt-c4-125m-seed964|C4 (early)|125M|step-180"
    "znhoughton/opt-c4-350m-seed964|C4 (early)|350M|step-112"
    "znhoughton/opt-c4-1.3b-seed964|C4 (early)|1300M|step-204"
    # C4 — mid checkpoint (~1/2 through training, step index 9/19)
    "znhoughton/opt-c4-125m-seed964|C4 (mid)|125M|step-480"
    "znhoughton/opt-c4-350m-seed964|C4 (mid)|350M|step-312"
    "znhoughton/opt-c4-1.3b-seed964|C4 (mid)|1300M|step-492"
    # GPT-2
    "gpt2|GPT-2|124M"
    "gpt2-medium|GPT-2|355M"
    "gpt2-large|GPT-2|774M"
    "gpt2-xl|GPT-2|1542M"
    # GPT-Neo
    "EleutherAI/gpt-neo-125m|GPT-Neo|125M"
    "EleutherAI/gpt-neo-1.3B|GPT-Neo|1300M"
    "EleutherAI/gpt-neo-2.7B|GPT-Neo|2700M"
    # OPT
    "facebook/opt-125m|OPT|125M"
    "facebook/opt-350m|OPT|350M"
    "facebook/opt-1.3b|OPT|1300M"
    "facebook/opt-2.7b|OPT|2700M"
    "facebook/opt-6.7b|OPT|6700M"
    "facebook/opt-13b|OPT|13000M"
    "facebook/opt-30b|OPT|30000M"
)

# ── Run each model ────────────────────────────────────────────────────────────
TOTAL=${#MODELS[@]}
IDX=0

for entry in "${MODELS[@]}"; do
    IDX=$((IDX + 1))
    MODEL_ID="${entry%%|*}"
    REST="${entry#*|}"
    FAMILY="${REST%%|*}"
    REST2="${REST#*|}"
    # Split PARAMS and optional REVISION from the remaining fields
    if [[ "$REST2" == *"|"* ]]; then
        PARAMS="${REST2%%|*}"
        REVISION="${REST2#*|}"
    else
        PARAMS="$REST2"
        REVISION=""
    fi

    # Include revision in safe name so checkpoint CSVs don't collide with final
    if [[ -n "$REVISION" ]]; then
        SAFE_NAME="${MODEL_ID//\//_}_${REVISION}"
    else
        SAFE_NAME="${MODEL_ID//\//_}"
    fi
    OUT_CSV="${SURP_DIR}/${SAFE_NAME}.csv"
    RAW_FILE="${RAW_DIR}/${SAFE_NAME}.surprisal"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    if [[ -n "$REVISION" ]]; then
        echo "[${IDX}/${TOTAL}] ${MODEL_ID}@${REVISION}  (${FAMILY}, ${PARAMS})"
    else
        echo "[${IDX}/${TOTAL}] ${MODEL_ID}  (${FAMILY}, ${PARAMS})"
    fi
    echo "════════════════════════════════════════════════════════════"

    # Skip if already done (unless --recompute)
    if [[ $RECOMPUTE -eq 0 && -f "${OUT_CSV}" ]]; then
        echo "  Skipping — output already exists: ${OUT_CSV}"
        continue
    fi

    # Pre-download model shards to HF cache before loading into GPU memory.
    echo "  Downloading ${MODEL_ID}${REVISION:+@${REVISION}}..."
    if [[ -n "$REVISION" ]]; then
        DL_CMD=(huggingface-cli download "${MODEL_ID}" --revision "${REVISION}" --repo-type model --quiet)
    else
        DL_CMD=(huggingface-cli download "${MODEL_ID}" --repo-type model --quiet)
    fi
    if ! "${DL_CMD[@]}"; then
        echo "  ERROR: download failed for ${MODEL_ID}${REVISION:+@${REVISION}}" >&2
        continue
    fi

    # Run Oh & Schuler's script — outputs to stdout
    echo "  Running get_llm_surprisal.py..."
    if [[ -n "$REVISION" ]]; then
        SURP_CMD=(python "${LLM_SURP_SCRIPT}" "${SENTITEMS}" "${MODEL_ID}" "${REVISION}" word)
    else
        SURP_CMD=(python "${LLM_SURP_SCRIPT}" "${SENTITEMS}" "${MODEL_ID}" word)
    fi
    if "${SURP_CMD[@]}" > "${RAW_FILE}" 2>"${RAW_DIR}/${SAFE_NAME}.log"; then
        echo "  Surprisal written to ${RAW_FILE}"
    else
        echo "  ERROR: get_llm_surprisal.py failed for ${MODEL_ID}${REVISION:+@${REVISION}}" >&2
        echo "  Check log: ${RAW_DIR}/${SAFE_NAME}.log" >&2
        huggingface-cli delete-cache --include "${MODEL_ID}" --yes 2>/dev/null || true
        continue
    fi

    # Parse raw output into item/zone CSV + update perplexity table
    echo "  Parsing output..."
    if [[ -n "$REVISION" ]]; then
        python "${PARSE_SCRIPT}" "${RAW_FILE}" "${MODEL_ID}" "${FAMILY}" "${PARAMS}" "${REVISION}"
    else
        python "${PARSE_SCRIPT}" "${RAW_FILE}" "${MODEL_ID}" "${FAMILY}" "${PARAMS}"
    fi

    # Delete HF cache to free disk space before the next model downloads
    echo "  Deleting cache for ${MODEL_ID}..."
    huggingface-cli delete-cache --include "${MODEL_ID}" --yes 2>/dev/null || true

    echo "  Done."
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "All models complete."
echo "  Surprisal CSVs : ${SURP_DIR}"
echo "  Perplexity     : ${SURP_DIR}/ns_perplexity.csv"
echo "  Raw outputs    : ${RAW_DIR}"
echo ""
echo "Next: source reading_time_delta_ll.R"
echo "════════════════════════════════════════════════════════════"
