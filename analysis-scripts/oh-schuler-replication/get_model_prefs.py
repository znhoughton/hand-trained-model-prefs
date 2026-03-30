#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_model_prefs.py
==================
Score the GPT-2 family models (Oh & Schuler 2023) on our binomial ordering
stimuli and compute per-model validation perplexity on WikiText-2.

For each model:
  - preference  = mean over prompts of [log P(α ordering) − log P(β ordering)]
  - perplexity  = exp(mean NLL) on WikiText-2 test set  ← VALIDATION perplexity

Output: oh_schuler_prefs.csv
  columns: model, model_params, binom, preference
           (perplexity joined per model in oh_schuler_perplexity.csv)

Run from the oh-schuler-replication/ folder:
  python get_model_prefs.py

Requirements:
  pip install torch transformers datasets tqdm
"""

import os
import math
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
BASE_DIR     = SCRIPT_DIR.parent.parent          # hand-trained-model-prefs/
DATA_DIR     = BASE_DIR / "Data"
HUMAN_CSV    = DATA_DIR / "all_human_data.csv"
OUT_PREFS    = SCRIPT_DIR / "oh_schuler_prefs.csv"
OUT_PPL      = SCRIPT_DIR / "oh_schuler_perplexity.csv"

# ── Models (Oh & Schuler 2023: GPT-2, GPT-Neo, OPT) ─────────────────────────
# 16 models total across three families.
# OPT ≥ 13B require multi-GPU or significant VRAM; set skip=True to omit them.
MODEL_CONFIGS = {
    # GPT-2 family
    "gpt2":                          {"params": "124M",   "family": "GPT-2",    "label": "GPT-2",        "skip": False},
    "gpt2-medium":                   {"params": "355M",   "family": "GPT-2",    "label": "GPT-2 medium", "skip": False},
    "gpt2-large":                    {"params": "774M",   "family": "GPT-2",    "label": "GPT-2 large",  "skip": False},
    "gpt2-xl":                       {"params": "1542M",  "family": "GPT-2",    "label": "GPT-2 XL",     "skip": False},
    # GPT-Neo family (EleutherAI)
    "EleutherAI/gpt-neo-125m":       {"params": "125M",   "family": "GPT-Neo",  "label": "GPT-Neo 125M", "skip": False},
    "EleutherAI/gpt-neo-1.3B":       {"params": "1300M",  "family": "GPT-Neo",  "label": "GPT-Neo 1.3B", "skip": False},
    "EleutherAI/gpt-neo-2.7B":       {"params": "2700M",  "family": "GPT-Neo",  "label": "GPT-Neo 2.7B", "skip": False},
    # OPT family (Meta)
    "facebook/opt-125m":             {"params": "125M",   "family": "OPT",      "label": "OPT-125M",     "skip": False},
    "facebook/opt-350m":             {"params": "350M",   "family": "OPT",      "label": "OPT-350M",     "skip": False},
    "facebook/opt-1.3b":             {"params": "1300M",  "family": "OPT",      "label": "OPT-1.3B",     "skip": False},
    "facebook/opt-2.7b":             {"params": "2700M",  "family": "OPT",      "label": "OPT-2.7B",     "skip": False},
    "facebook/opt-6.7b":             {"params": "6700M",  "family": "OPT",      "label": "OPT-6.7B",     "skip": False},
    "facebook/opt-13b":              {"params": "13000M", "family": "OPT",      "label": "OPT-13B",      "skip": False},
    "facebook/opt-30b":              {"params": "30000M", "family": "OPT",      "label": "OPT-30B",      "skip": False},
    "facebook/opt-66b":              {"params": "66000M", "family": "OPT",      "label": "OPT-66B",      "skip": False},
    "facebook/opt-175b":             {"params": "175000M","family": "OPT",      "label": "OPT-175B",     "skip": False},
}

# ── Sentence-frame prompts (same set as model-prefs-all-ckpts.py) ─────────────
LIST_OF_PROMPTS = [
    " ",
    "Well, ",
    "So, ",
    "Then ",
    "Possibly ",
    "Or even ",
    "Maybe a ",
    "Perhaps a ",
    "At times ",
    "Suddenly, the ",
    "Honestly just ",
    "Especially the ",
    "For instance ",
    "In some cases ",
    "Every now and then ",
    "Occasionally you'll find ",
    "There can be examples like ",
    "You might notice things like ",
    "People sometimes mention ",
    "Sometimes just ",
    "Nothing specific comes to mind except the ",
    "It reminded me loosely of the ",
    "There was a vague reference to the ",
    "Unexpectedly the ",
    "It's easy to overlook the ",
    "There used to be talk of ",
    "Out in the distance was the ",
    "What puzzled everyone was the ",
    "At some point I overheard ",
    "Without warning came ",
    "A friend once described the ",
    "The scene shifted toward ",
    "Nobody expected to hear about ",
    "Things eventually turned toward ",
    "The conversation eventually returned to ",
    "I only remember a hint of the ",
    "I couldn't quite place the ",
    "It somehow led back to the ",
    "What stood out most was the ",
    "The oddest part involved the ",
    "Later on, people were discussing ",
    "There was this fleeting idea about ",
    "I once heard someone bring up the ",
    "There was a moment involving the ",
    "It all started when we noticed the ",
    "Another example floated around concerning the ",
    "I came across something about the ",
    "A situation arose involving the ",
    "The conversation drifted toward the ",
    "At one point we ended up discussing ",
    "Out of nowhere came a mention of the ",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.inference_mode()
def batch_logprobs(model, tokenizer, texts: list[str], device: str,
                   batch_size: int = 64) -> list[float]:
    """Total sequence log-probability for each text string."""
    all_lp = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding="longest", return_tensors="pt",
                        pad_to_multiple_of=8)
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs    = model(input_ids, attention_mask=attention_mask)
        logprobs   = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:]

        token_lp = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        token_lp = token_lp * target_mask
        all_lp.extend(token_lp.sum(dim=-1).tolist())
    return all_lp


def score_binomials(model, tokenizer, binoms_df: pd.DataFrame,
                    model_id: str, device: str) -> pd.DataFrame:
    """
    For every (prompt, binomial) pair compute log P(α) − log P(β).
    Returns a DataFrame with one row per binomial, preference = mean over prompts.
    """
    rows = []
    for prompt in tqdm(LIST_OF_PROMPTS, desc="  prompts", leave=False):
        alpha_texts    = (prompt + binoms_df["Alpha"]).tolist()
        nonalpha_texts = (prompt + binoms_df["Nonalpha"]).tolist()
        combined = alpha_texts + nonalpha_texts
        lps = batch_logprobs(model, tokenizer, combined, device)
        n = len(alpha_texts)
        for i, row in enumerate(binoms_df.itertuples(index=False)):
            rows.append({
                "model":      model_id,
                "binom":      row.Alpha,
                "prompt":     prompt,
                "preference": lps[i] - lps[n + i],
            })

    df = pd.DataFrame(rows)
    # Average preference across prompts
    return (df.groupby(["model", "binom"], as_index=False)["preference"].mean())


@torch.inference_mode()
def compute_validation_perplexity(model, tokenizer, device: str,
                                   max_tokens: int = 4096) -> float:
    """
    Validation perplexity on the WikiText-2 test set.
    We concatenate the test text and stride through it with the model's
    context window, taking the NLL only on the non-overlapping portion.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           split="test", trust_remote_code=True)
    text = "\n".join(t for t in dataset["text"] if t.strip())

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    # Truncate to max_tokens to keep it fast on CPU
    input_ids = input_ids[:max_tokens].to(device)

    stride      = 512
    seq_len     = input_ids.size(0)
    max_ctx     = getattr(model.config, "n_positions",
                          getattr(model.config, "max_position_embeddings", 1024))
    window      = min(max_ctx, 1024)

    nll_sum  = 0.0
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end     = min(begin + window, seq_len)
        chunk   = input_ids[begin:end].unsqueeze(0)
        # Only score tokens after the overlap with the previous window
        target_len = end - max(begin, prev_end)
        if target_len <= 0:
            prev_end = end
            continue

        outputs = model(chunk, labels=chunk)
        # outputs.loss is mean NLL over the whole chunk; recover total
        chunk_nll = outputs.loss.item() * (end - begin - 1)
        # Approximate: attribute the last target_len tokens' NLL
        nll_sum  += chunk_nll * target_len / (end - begin - 1)
        n_tokens += target_len
        prev_end  = end

        if end >= seq_len:
            break

    return math.exp(nll_sum / n_tokens)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Using device: {device}\n")

    # Load binomials that appear in the human experiment
    print("Loading human experiment binomials ...")
    human = pd.read_csv(HUMAN_CSV)
    binoms_df = (human[["Alpha", "Nonalpha"]]
                 .drop_duplicates(subset="Alpha")
                 .reset_index(drop=True))
    print(f"  {len(binoms_df)} unique binomials\n")

    all_prefs = []
    all_ppl   = []

    for model_id, cfg in MODEL_CONFIGS.items():
        if cfg.get("skip", False):
            print(f"Skipping {model_id} (skip=True — insufficient VRAM)")
            continue

        print("=" * 60)
        print(f"Model: {model_id}  ({cfg['family']}, {cfg['params']})")
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device).eval()

        # ── Binomial preferences ──
        print("  Scoring binomial preferences ...")
        prefs_df = score_binomials(model, tokenizer, binoms_df, model_id, device)
        prefs_df["model_family"] = cfg["family"]
        prefs_df["model_params"] = cfg["params"]
        prefs_df["model_label"]  = cfg["label"]
        all_prefs.append(prefs_df)

        # ── Validation perplexity (WikiText-2 test) ──
        print("  Computing validation perplexity (WikiText-2 test) ...")
        ppl = compute_validation_perplexity(model, tokenizer, device)
        print(f"  Perplexity: {ppl:.2f}")
        all_ppl.append({
            "model":        model_id,
            "model_family": cfg["family"],
            "model_params": cfg["params"],
            "model_label":  cfg["label"],
            "perplexity":   ppl,
        })

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print()

    # ── Save ──────────────────────────────────────────────────────────────────
    prefs_out = pd.concat(all_prefs, ignore_index=True)
    prefs_out.to_csv(OUT_PREFS, index=False)
    print(f"Saved preferences → {OUT_PREFS}")

    ppl_out = pd.DataFrame(all_ppl)
    ppl_out.to_csv(OUT_PPL, index=False)
    print(f"Saved perplexities → {OUT_PPL}")
    print("\nDone.")


if __name__ == "__main__":
    main()
