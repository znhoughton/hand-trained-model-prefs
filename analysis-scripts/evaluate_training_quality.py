#!/usr/bin/env python3
"""
evaluate_training_quality.py

Two complementary checks that a model trained well:

  1. Perplexity table  — final-checkpoint perplexity for all six models on
     their respective held-out validation sets, showing expected scaling
     behaviour (125M > 350M > 1.3B within each corpus).

  2. Loss curves       — training loss trajectory from each model's
     trainer_state.json on HuggingFace Hub, showing convergence.

Held-out sets used (matching actual training splits):
  BabyLM → znhoughton/babylm-150m-v3       split="train[:5%]"
  C4     → znhoughton/c4-subset-10B-tokens  split="validation"

Outputs written to ../Data/training_quality/:
  perplexity_results.csv
  loss_curves.csv
  perplexity_table.png
  loss_curves.png

Requirements:
  pip install torch transformers datasets huggingface_hub pandas matplotlib tqdm
"""

import json
import math
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

TOKENIZER_ID = "znhoughton/opt-babylm-125m-64eps-seed964"  # tokenizer is identical across all 6 models
MAX_EVAL_TOKENS = 1_000_000   # ~1M tokens per model is fast and representative
BLOCK_SIZE = 1024
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("../Data/training_quality")

MODELS = [
    {"corpus": "BabyLM", "params": "125M", "hub_id": "znhoughton/opt-babylm-125m-64eps-seed964"},
    {"corpus": "BabyLM", "params": "350M", "hub_id": "znhoughton/opt-babylm-350m-64eps-seed964"},
    {"corpus": "BabyLM", "params": "1.3B", "hub_id": "znhoughton/opt-babylm-1.3b-64eps-seed964"},
    {"corpus": "C4",     "params": "125M", "hub_id": "znhoughton/opt-c4-125m-seed964"},
    {"corpus": "C4",     "params": "350M", "hub_id": "znhoughton/opt-c4-350m-seed964"},
    {"corpus": "C4",     "params": "1.3B", "hub_id": "znhoughton/opt-c4-1.3b-seed964"},
]

# Published OPT perplexities for reference (Zhang et al. 2022, Table 3, WikiText-103)
# Note: OPT used a different tokenizer and training corpus, so these are
# rough reference points only, not directly comparable.
OPT_REFERENCE_PPL = {
    "125M": 27.7,
    "350M": 22.0,
    "1.3B": 17.9,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_val_tokens(corpus: str, tokenizer) -> torch.Tensor:
    """
    Load and tokenize the held-out validation split for a given corpus.
    Returns a (n_chunks, BLOCK_SIZE) LongTensor.
    """
    print(f"  Loading {corpus} validation data...")
    if corpus == "BabyLM":
        dataset = load_dataset(
            "znhoughton/babylm-150m-v3",
            split="train[:5%]",
        )
    else:
        dataset = load_dataset(
            "znhoughton/c4-subset-10B-tokens",
            split="validation",
        )

    all_ids = []
    with tqdm(total=MAX_EVAL_TOKENS, desc=f"  Tokenizing {corpus}", unit="tok", leave=True) as pbar:
        for example in dataset:
            ids = tokenizer(example["text"], add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            pbar.update(len(ids))
            if len(all_ids) >= MAX_EVAL_TOKENS:
                break

    all_ids = all_ids[:MAX_EVAL_TOKENS]
    # Drop trailing tokens that don't fill a full block
    n_full = (len(all_ids) // BLOCK_SIZE) * BLOCK_SIZE
    chunks = [all_ids[i : i + BLOCK_SIZE] for i in range(0, n_full, BLOCK_SIZE)]
    tqdm.write(f"    → {len(chunks):,} chunks × {BLOCK_SIZE} tokens = {n_full:,} tokens")
    return torch.tensor(chunks, dtype=torch.long)


def compute_perplexity(model, chunks: torch.Tensor) -> float:
    """
    Compute token-level cross-entropy perplexity over pre-chunked input ids.
    Loss is averaged over all tokens, then exponentiated.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    n_batches = math.ceil(len(chunks) / BATCH_SIZE)
    with torch.inference_mode():
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), total=n_batches,
                      desc="  Evaluating", unit="batch", leave=True):
            batch = chunks[i : i + BATCH_SIZE].to(DEVICE)
            outputs = model(batch, labels=batch)
            n = batch.numel()
            total_loss += outputs.loss.item() * n
            total_tokens += n

    return math.exp(total_loss / total_tokens)


def fetch_loss_curve(hub_id: str) -> Optional[pd.DataFrame]:
    """
    Download trainer_state.json from a Hub model repo and extract the
    training loss history. Returns a DataFrame with columns
    [step, tokens_seen, loss] or None if the file is unavailable.
    """
    try:
        path = hf_hub_download(repo_id=hub_id, filename="trainer_state.json")
    except Exception as e:
        print(f"    trainer_state.json not found for {hub_id}: {e}")
        return None

    with open(path) as f:
        state = json.load(f)

    rows = [
        {"step": entry["step"], "loss": entry["loss"]}
        for entry in state.get("log_history", [])
        if "loss" in entry
    ]
    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Tokens seen = step × tokens_per_step; tokens_per_step = 819,200 for 125M/1.3B,
    # 1,228,800 for 350M. Derive from hub_id.
    if "350m" in hub_id.lower():
        tokens_per_step = 1_228_800
    else:
        tokens_per_step = 819_200
    df["tokens_seen"] = df["step"] * tokens_per_step
    return df[["step", "tokens_seen", "loss"]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Part 1: Perplexity ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 1: FINAL-CHECKPOINT PERPLEXITY")
    print("=" * 60)

    val_cache = {}  # corpus -> tokenized chunks tensor
    ppl_rows = []

    model_pbar = tqdm(MODELS, desc="Models", unit="model", position=0)
    for cfg in model_pbar:
        corpus, params, hub_id = cfg["corpus"], cfg["params"], cfg["hub_id"]
        model_pbar.set_description(f"{corpus} {params}")
        tqdm.write(f"\n{'─'*60}\n{hub_id}")

        if corpus not in val_cache:
            val_cache[corpus] = load_val_tokens(corpus, tokenizer)
        chunks = val_cache[corpus]

        tqdm.write(f"  Loading model ({params})...")
        model = AutoModelForCausalLM.from_pretrained(
            hub_id,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
        )

        ppl = compute_perplexity(model, chunks)
        tqdm.write(f"  → Perplexity: {ppl:.2f}")

        ppl_rows.append({"Corpus": corpus, "Params": params, "Perplexity": round(ppl, 2)})

        del model
        torch.cuda.empty_cache()

    ppl_df = pd.DataFrame(ppl_rows)
    ppl_df.to_csv(OUT_DIR / "perplexity_results.csv", index=False)

    print("\n── Perplexity table ──")
    print(ppl_df.pivot(index="Params", columns="Corpus", values="Perplexity")
                .reindex(["125M", "350M", "1.3B"])
                .to_string())
    print("\nExpected: perplexity decreases as params increase within each corpus.")

    # Check scaling holds
    for corpus in ["BabyLM", "C4"]:
        subset = ppl_df[ppl_df["Corpus"] == corpus].set_index("Params")
        ppls = [subset.loc[p, "Perplexity"] for p in ["125M", "350M", "1.3B"]]
        ok = ppls[0] > ppls[1] > ppls[2]
        print(f"  {corpus} scaling (125M > 350M > 1.3B): {'✓ PASS' if ok else '✗ FAIL'}")

    # ── Part 2: Loss curves ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 2: TRAINING LOSS CURVES")
    print("=" * 60)

    loss_rows = []
    for cfg in tqdm(MODELS, desc="Fetching loss curves", unit="model"):
        tqdm.write(f"  {cfg['hub_id']}")
        df = fetch_loss_curve(cfg["hub_id"])
        if df is not None:
            df["corpus"] = cfg["corpus"]
            df["params"] = cfg["params"]
            loss_rows.append(df)
            final_loss = df["loss"].iloc[-1]
            tqdm.write(f"    → Final loss: {final_loss:.4f}  ({len(df):,} steps)")

    if loss_rows:
        loss_df = pd.concat(loss_rows, ignore_index=True)
        loss_df.to_csv(OUT_DIR / "loss_curves.csv", index=False)
        print(f"\nSaved loss_curves.csv ({len(loss_df):,} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_perplexity(ppl_df)
    if loss_rows:
        _plot_loss_curves(loss_df)

    print(f"\nAll outputs saved to {OUT_DIR.resolve()}")


# ── Plotting ──────────────────────────────────────────────────────────────────

CORPUS_COLORS = {"BabyLM": "#2166ac", "C4": "#d6604d"}
PARAM_MARKERS = {"125M": "o", "350M": "s", "1.3B": "^"}
PARAM_ORDER = ["125M", "350M", "1.3B"]


def _plot_perplexity(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = range(len(PARAM_ORDER))

    for corpus, color in CORPUS_COLORS.items():
        subset = df[df["Corpus"] == corpus].set_index("Params")
        ppls = [subset.loc[p, "Perplexity"] for p in PARAM_ORDER]
        ax.plot(x, ppls, marker="o", color=color, label=corpus, linewidth=1.8)
        for xi, ppl in zip(x, ppls):
            ax.text(xi, ppl + 0.3, f"{ppl:.1f}", ha="center", va="bottom", fontsize=8, color=color)

    ax.set_xticks(list(x))
    ax.set_xticklabels(PARAM_ORDER)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Perplexity (↓ better)")
    ax.set_title("Final-checkpoint perplexity by corpus and model size")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "perplexity_table.png", dpi=150)
    plt.close(fig)
    print("Saved perplexity_table.png")


def _plot_loss_curves(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, corpus in zip(axes, ["BabyLM", "C4"]):
        subset = df[df["corpus"] == corpus]
        for params in PARAM_ORDER:
            model_df = subset[subset["params"] == params].sort_values("tokens_seen")
            if model_df.empty:
                continue
            ax.plot(
                model_df["tokens_seen"] / 1e9,
                model_df["loss"],
                label=params,
                linewidth=1.2,
                alpha=0.85,
            )
        ax.set_title(f"{corpus} training loss")
        ax.set_xlabel("Tokens seen (B)")
        ax.set_ylabel("Cross-entropy loss")
        ax.legend(title="Params")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_curves.png", dpi=150)
    plt.close(fig)
    print("Saved loss_curves.png")


if __name__ == "__main__":
    main()
