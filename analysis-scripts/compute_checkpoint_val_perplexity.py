#!/usr/bin/env python3
"""
compute_checkpoint_val_perplexity.py

Computes validation perplexity at every sampled training checkpoint for the
six OPT models (BabyLM × {125M, 350M, 1.3B} and C4 × {125M, 350M, 1.3B}).

Reads ck_meta.csv (written by the ck-index chunk in abspref_casestudy.Rmd)
to evaluate exactly the same checkpoints used in the ΔLL trajectory plot.

Held-out validation sets (same as evaluate_training_quality.py):
  BabyLM  →  znhoughton/babylm-150m-v3          split="train[:5%]"
  C4      →  znhoughton/c4-subset-10B-tokens     split="validation"

Output: ../Data/training_quality/val_perplexity_curves.csv
  columns: corpus, params, step, val_perplexity

Resume-safe: skips (corpus, params, step) rows already in the output file.

Requirements:
  pip install torch transformers datasets huggingface_hub pandas tqdm

Run from analysis-scripts/:
  python compute_checkpoint_val_perplexity.py
"""

import math
import os
import re
import tempfile
import shutil
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BASE_DIR   = SCRIPT_DIR.parent
DATA_DIR   = BASE_DIR / "Data"
TQ_DIR     = DATA_DIR / "training_quality"
CK_META_CSV = TQ_DIR / "ck_meta.csv"
OUT_CSV     = TQ_DIR / "val_perplexity_curves.csv"

TOKENIZER_ID    = "znhoughton/opt-babylm-125m-64eps-seed964"  # same tokenizer for all 6
MAX_EVAL_TOKENS = 500_000   # 500K tokens is fast and representative
BLOCK_SIZE      = 1024
BATCH_SIZE      = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Map the short model name (from ck_meta) to HuggingFace repo + metadata
MODEL_META = {
    "znhoughton/opt-babylm-125m-64eps-seed964": {"corpus": "BabyLM", "params": "125M"},
    "znhoughton/opt-babylm-350m-64eps-seed964": {"corpus": "BabyLM", "params": "350M"},
    "znhoughton/opt-babylm-1.3b-64eps-seed964": {"corpus": "BabyLM", "params": "1.3B"},
    "znhoughton/opt-c4-125m-seed964":            {"corpus": "C4",     "params": "125M"},
    "znhoughton/opt-c4-350m-seed964":            {"corpus": "C4",     "params": "350M"},
    "znhoughton/opt-c4-1.3b-seed964":            {"corpus": "C4",     "params": "1.3B"},
}

# ── Validation data loader ────────────────────────────────────────────────────

def load_val_tokens(corpus: str, tokenizer) -> torch.Tensor:
    """Tokenise the held-out validation split and return chunked LongTensor."""
    print(f"  Loading {corpus} validation data ...")
    if corpus == "BabyLM":
        dataset = load_dataset("znhoughton/babylm-150m-v3",
                               split="train[:5%]", trust_remote_code=True)
    else:
        dataset = load_dataset("znhoughton/c4-subset-10B-tokens",
                               split="validation", trust_remote_code=True)

    all_ids = []
    for example in tqdm(dataset, desc=f"  Tokenising {corpus}", leave=False):
        ids = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        if len(all_ids) >= MAX_EVAL_TOKENS:
            break

    all_ids = all_ids[:MAX_EVAL_TOKENS]
    n_full  = (len(all_ids) // BLOCK_SIZE) * BLOCK_SIZE
    chunks  = [all_ids[i : i + BLOCK_SIZE] for i in range(0, n_full, BLOCK_SIZE)]
    print(f"    → {len(chunks):,} chunks × {BLOCK_SIZE} tokens")
    return torch.tensor(chunks, dtype=torch.long)


# ── Perplexity computation ─────────────────────────────────────────────────────

@torch.inference_mode()
def compute_perplexity(model, chunks: torch.Tensor) -> float:
    model.eval()
    total_loss   = 0.0
    total_tokens = 0
    for i in tqdm(range(0, len(chunks), BATCH_SIZE),
                  desc="  Evaluating", leave=False):
        batch = chunks[i : i + BATCH_SIZE].to(DEVICE)
        loss  = model(batch, labels=batch).loss.item()
        n     = batch.numel()
        total_loss   += loss * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Load ck_meta ──────────────────────────────────────────────────────────
    if not CK_META_CSV.exists():
        raise FileNotFoundError(
            f"{CK_META_CSV} not found.\n"
            "Knit/run abspref_casestudy.Rmd through the ck-index chunk first "
            "to export ck_meta.csv."
        )

    ck_meta = pd.read_csv(CK_META_CSV)
    # Keep only the six OPT models
    ck_meta = ck_meta[ck_meta["model"].isin(MODEL_META)].copy()
    print(f"ck_meta: {len(ck_meta)} rows, "
          f"{ck_meta['model'].nunique()} models, "
          f"{ck_meta['step'].nunique()} unique steps")

    # ── Load already-computed results (resume safety) ─────────────────────────
    # Validate each row has a real perplexity value — a crash mid-write could
    # have left a corrupted file or rows with NaN.
    if OUT_CSV.exists():
        try:
            raw = pd.read_csv(OUT_CSV)
            done = raw.dropna(subset=["val_perplexity"])
            n_corrupt = len(raw) - len(done)
            if n_corrupt:
                print(f"⚠️  Dropped {n_corrupt} corrupt/incomplete rows from {OUT_CSV}")
        except Exception as e:
            print(f"⚠️  Could not read {OUT_CSV} ({e}) — starting fresh.")
            done = pd.DataFrame()
        done_keys = set(zip(done["corpus"], done["params"], done["step"]))
        print(f"Resuming: {len(done)} valid rows already computed, skipping those.")
    else:
        done      = pd.DataFrame()
        done_keys = set()

    # ── Tokeniser (shared across all models) ─────────────────────────────────
    print("Loading tokeniser ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Cache validation tokens per corpus (expensive to re-tokenise) ─────────
    val_cache: dict[str, torch.Tensor] = {}

    # ── Iterate: group by model so we load each model once per block ──────────
    new_rows = []

    for model_id, group in ck_meta.groupby("model"):
        meta   = MODEL_META[model_id]
        corpus = meta["corpus"]
        params = meta["params"]

        # Pre-load validation tokens for this corpus if not cached
        if corpus not in val_cache:
            val_cache[corpus] = load_val_tokens(corpus, tokenizer)
        chunks = val_cache[corpus]

        # Only evaluate steps not yet done
        todo = group[
            ~group["step"].apply(lambda s: (corpus, params, s) in done_keys)
        ]
        if todo.empty:
            print(f"\n✅ {model_id}: all checkpoints already computed, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_id}  ({params}, {corpus})")
        print(f"  {len(todo)} checkpoints to evaluate")
        print(f"{'='*60}")

        for _, row in todo.iterrows():
            step     = int(row["step"])
            revision = f"step-{step}"

            print(f"\n  → Checkpoint step={step}  (revision={revision})")

            tmp_cache = tempfile.mkdtemp(prefix="hf_ckpt_val_")
            model = None
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    revision=revision,
                    torch_dtype=torch.bfloat16,
                    device_map=DEVICE,
                    cache_dir=tmp_cache,
                ).eval()

                ppl = compute_perplexity(model, chunks)
                print(f"     val_perplexity = {ppl:.2f}")

                new_rows.append({
                    "corpus":         corpus,
                    "params":         params,
                    "step":           step,
                    "val_perplexity": ppl,
                })

                # Atomic incremental save after each checkpoint
                all_rows = (
                    pd.concat([done, pd.DataFrame(new_rows)], ignore_index=True)
                    if not done.empty
                    else pd.DataFrame(new_rows)
                )
                tmp_out = str(OUT_CSV) + ".tmp"
                all_rows.to_csv(tmp_out, index=False)
                os.replace(tmp_out, OUT_CSV)

            except Exception as e:
                print(f"  ⚠️  Failed step={step}: {e}")
            finally:
                if model is not None:
                    del model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                shutil.rmtree(tmp_cache, ignore_errors=True)
                print(f"     🗑️  Checkpoint cache deleted.")

    print(f"\n✅ Done. Results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
