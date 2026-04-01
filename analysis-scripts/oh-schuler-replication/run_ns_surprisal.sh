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
# Format: "model_id|family|params"          ← final checkpoints
#         "model_id|family|params|revision" ← intermediate checkpoints
MODELS=(
    # BabyLM — final checkpoint
    "znhoughton/opt-babylm-125m-64eps-seed964|BabyLM|125M"
    "znhoughton/opt-babylm-350m-64eps-seed964|BabyLM|350M"
    "znhoughton/opt-babylm-1.3b-64eps-seed964|BabyLM|1300M"
    # BabyLM — early checkpoint (~1/3 through training by token count)
    "znhoughton/opt-babylm-125m-64eps-seed964|BabyLM (early)|125M|step-3732"
    "znhoughton/opt-babylm-350m-64eps-seed964|BabyLM (early)|350M|step-2488"
    "znhoughton/opt-babylm-1.3b-64eps-seed964|BabyLM (early)|1300M|step-4020"
    # BabyLM — mid checkpoint (~1/2 through training by token count)
    "znhoughton/opt-babylm-125m-64eps-seed964|BabyLM (mid)|125M|step-5340"
    "znhoughton/opt-babylm-350m-64eps-seed964|BabyLM (mid)|350M|step-3560"
    "znhoughton/opt-babylm-1.3b-64eps-seed964|BabyLM (mid)|1300M|step-5532"
    # C4 — final checkpoint
    "znhoughton/opt-c4-125m-seed964|C4|125M"
    "znhoughton/opt-c4-350m-seed964|C4|350M"
    "znhoughton/opt-c4-1.3b-seed964|C4|1300M"
    # C4 — early checkpoint (~1/3 through training by token count)
    "znhoughton/opt-c4-125m-seed964|C4 (early)|125M|step-4104"
    "znhoughton/opt-c4-350m-seed964|C4 (early)|350M|step-2720"
    "znhoughton/opt-c4-1.3b-seed964|C4 (early)|1300M|step-3948"
    # C4 — mid checkpoint (~1/2 through training by token count)
    "znhoughton/opt-c4-125m-seed964|C4 (mid)|125M|step-5892"
    "znhoughton/opt-c4-350m-seed964|C4 (mid)|350M|step-3920"
    "znhoughton/opt-c4-1.3b-seed964|C4 (mid)|1300M|step-5640"
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
    IFS='|' read -ra FIELDS <<< "$entry"
    MODEL_ID="${FIELDS[0]}"
    FAMILY="${FIELDS[1]}"
    PARAMS="${FIELDS[2]}"
    REVISION="${FIELDS[3]:-}"

    # PPL_KEY: include @revision for checkpoint entries so the CSV key matches
    # what get_model_prefs.py writes to oh_schuler_perplexity.csv
    if [[ -n "${REVISION}" ]]; then
        PPL_KEY="${MODEL_ID}@${REVISION}"
    else
        PPL_KEY="${MODEL_ID}"
    fi
    SAFE_NAME="${PPL_KEY//\//_}"
    OUT_CSV="${SURP_DIR}/${SAFE_NAME}.csv"
    RAW_FILE="${RAW_DIR}/${SAFE_NAME}.surprisal"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "[${IDX}/${TOTAL}] ${MODEL_ID}  (${FAMILY}, ${PARAMS})"
    echo "════════════════════════════════════════════════════════════"

    # Skip if already done (unless --recompute)
    if [[ $RECOMPUTE -eq 0 && -f "${OUT_CSV}" ]]; then
        echo "  Skipping — output already exists: ${OUT_CSV}"
        continue
    fi

    # Pre-download model shards to HF cache before loading into GPU memory.
    echo "  Downloading ${MODEL_ID}${REVISION:+ @ ${REVISION}}..."
    if [[ -n "${REVISION}" ]]; then
        DL_ARGS="--revision ${REVISION}"
    else
        DL_ARGS=""
    fi
    # shellcheck disable=SC2086
    if ! huggingface-cli download "${MODEL_ID}" ${DL_ARGS} --repo-type model --quiet; then
        echo "  ERROR: download failed for ${MODEL_ID}" >&2
        continue
    fi

    # Run Oh & Schuler's script — outputs to stdout
    echo "  Running get_llm_surprisal.py..."
    if [[ -n "${REVISION}" ]]; then
        python "${LLM_SURP_SCRIPT}" "${SENTITEMS}" "${MODEL_ID}" "${REVISION}" word \
            > "${RAW_FILE}" 2>"${RAW_DIR}/${SAFE_NAME}.log"
    else
        python "${LLM_SURP_SCRIPT}" "${SENTITEMS}" "${MODEL_ID}" word \
            > "${RAW_FILE}" 2>"${RAW_DIR}/${SAFE_NAME}.log"
    fi
    if [[ $? -eq 0 ]]; then
        echo "  Surprisal written to ${RAW_FILE}"
    else
        echo "  ERROR: get_llm_surprisal.py failed for ${MODEL_ID}" >&2
        echo "  Check log: ${RAW_DIR}/${SAFE_NAME}.log" >&2
        huggingface-cli delete-cache --include "${MODEL_ID}" --yes 2>/dev/null || true
        continue
    fi

    # Parse raw output into item/zone CSV + update perplexity table.
    # PPL_KEY (model@revision or just model) is used as the model identifier in
    # the output CSV so it matches the keys written by get_model_prefs.py.
    echo "  Parsing output..."
    python "${PARSE_SCRIPT}" "${RAW_FILE}" "${PPL_KEY}" "${FAMILY}" "${PARAMS}"

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
