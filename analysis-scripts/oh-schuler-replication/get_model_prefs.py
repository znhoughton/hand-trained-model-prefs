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
import datetime
import shutil
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
OUT_PREFS         = SCRIPT_DIR / "oh_schuler_prefs.csv"           # averaged across prompts
OUT_PREFS_BY_PROMPT = SCRIPT_DIR / "oh_schuler_prefs_by_prompt.csv"  # per-prompt (for mixed models)
OUT_PPL           = SCRIPT_DIR / "oh_schuler_perplexity.csv"
STAGING_DIR  = SCRIPT_DIR / "staging"
STAGING_DIR.mkdir(exist_ok=True)

# ── Models (Oh & Schuler 2023: GPT-2, GPT-Neo, OPT) ─────────────────────────
# 16 models total across three families.
# OPT ≥ 13B require multi-GPU or significant VRAM; set skip=True to omit them.
MODEL_CONFIGS = {
    # GPT-2 family
    "gpt2":                          {"params": "124M",    "family": "GPT-2",   "label": "GPT-2",         "skip": False},
    "gpt2-medium":                   {"params": "355M",    "family": "GPT-2",   "label": "GPT-2 medium",  "skip": False},
    "gpt2-large":                    {"params": "774M",    "family": "GPT-2",   "label": "GPT-2 large",   "skip": False},
    "gpt2-xl":                       {"params": "1542M",   "family": "GPT-2",   "label": "GPT-2 XL",      "skip": False},
    # GPT-Neo family (EleutherAI)
    "EleutherAI/gpt-neo-125m":       {"params": "125M",    "family": "GPT-Neo", "label": "GPT-Neo 125M",  "skip": False},
    "EleutherAI/gpt-neo-1.3B":       {"params": "1300M",   "family": "GPT-Neo", "label": "GPT-Neo 1.3B",  "skip": False},
    "EleutherAI/gpt-neo-2.7B":       {"params": "2700M",   "family": "GPT-Neo", "label": "GPT-Neo 2.7B",  "skip": False},
    # OPT family (Meta)
    "facebook/opt-125m":             {"params": "125M",    "family": "OPT",     "label": "OPT-125M",      "skip": False},
    "facebook/opt-350m":             {"params": "350M",    "family": "OPT",     "label": "OPT-350M",      "skip": False},
    "facebook/opt-1.3b":             {"params": "1300M",   "family": "OPT",     "label": "OPT-1.3B",      "skip": False},
    "facebook/opt-2.7b":             {"params": "2700M",   "family": "OPT",     "label": "OPT-2.7B",      "skip": False},
    "facebook/opt-6.7b":             {"params": "6700M",   "family": "OPT",     "label": "OPT-6.7B",      "skip": False},
    "facebook/opt-13b":              {"params": "13000M",  "family": "OPT",     "label": "OPT-13B",       "skip": False},
    "facebook/opt-30b":              {"params": "30000M",  "family": "OPT",     "label": "OPT-30B",       "skip": False, "multi_gpu": False},  # ~60 GB — fits on one H100
    "facebook/opt-66b":              {"params": "66000M",  "family": "OPT",     "label": "OPT-66B",       "skip": False, "multi_gpu": True},   # ~132 GB — needs both H100s
    "facebook/opt-175b":             {"params": "175000M", "family": "OPT",     "label": "OPT-175B",      "skip": True,  "multi_gpu": True},  # not publicly available on HuggingFace
    # OLMo family (AllenAI) — trained on Dolma, a curated open web corpus
    "allenai/OLMo-1B":               {"params": "1000M",   "family": "OLMo",    "label": "OLMo-1B",       "skip": False},
    "allenai/OLMo-7B":               {"params": "7000M",   "family": "OLMo",    "label": "OLMo-7B",       "skip": False},
    "allenai/OLMo-2-1124-13B":       {"params": "13000M",  "family": "OLMo",    "label": "OLMo-2-13B",    "skip": False, "multi_gpu": True},
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


def atomic_csv_write(df: pd.DataFrame, path: Path) -> None:
    """Write CSV atomically: write to .tmp then rename, so crashes can't corrupt."""
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def get_completed_prompts(staging_path: Path, expected_binoms: set) -> set:
    """
    Return the set of prompts that are fully complete — i.e., every expected
    binomial has a scored row. Prompts where the script crashed mid-batch and
    only some binomials were written are treated as incomplete.
    """
    if not staging_path.exists():
        return set()
    try:
        df = pd.read_csv(staging_path)
    except Exception:
        return set()
    completed = set()
    for prompt, grp in df.groupby("prompt"):
        if expected_binoms.issubset(set(grp["binom"])):
            completed.add(prompt)
    return completed


def _first_device(model) -> torch.device:
    """Return the device of the model's first parameter (works with device_map='auto')."""
    return next(model.parameters()).device


@torch.inference_mode()
def batch_logprobs(model, tokenizer, texts: list[str],
                   batch_size: int = 64) -> list[float]:
    """Total sequence log-probability for each text string."""
    first_dev = _first_device(model)
    all_lp = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding="longest", return_tensors="pt",
                        pad_to_multiple_of=8)
        input_ids      = enc["input_ids"].to(first_dev)
        attention_mask = enc["attention_mask"].to(first_dev)

        outputs    = model(input_ids, attention_mask=attention_mask)
        logprobs   = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:]

        token_lp = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        token_lp = token_lp * target_mask
        all_lp.extend(token_lp.sum(dim=-1).tolist())
    return all_lp


def score_binomials(model, tokenizer, binoms_df: pd.DataFrame,
                    model_id: str, staging_path: Path) -> pd.DataFrame:
    """
    Score every (prompt, binomial) pair. Results are saved prompt-by-prompt to
    staging_path so a crash only loses at most one prompt's work. Already-complete
    prompts (all binomials present) are skipped without reloading the model.

    Returns averaged preferences (one row per binomial).
    """
    expected_binoms  = set(binoms_df["Alpha"])
    completed        = get_completed_prompts(staging_path, expected_binoms)
    missing_prompts  = [p for p in LIST_OF_PROMPTS if p not in completed]

    print(f"  Prompts: {len(completed)} already complete, {len(missing_prompts)} to run.")

    for prompt in tqdm(missing_prompts, desc="  prompts", leave=False):
        alpha_texts    = [prompt + str(t) for t in binoms_df["Alpha"]]
        nonalpha_texts = [prompt + str(t) for t in binoms_df["Nonalpha"]]
        lps = batch_logprobs(model, tokenizer, alpha_texts + nonalpha_texts)
        n   = len(alpha_texts)

        prompt_rows = [
            {"model": model_id, "binom": row.Alpha,
             "prompt": prompt, "preference": lps[i] - lps[n + i]}
            for i, row in enumerate(binoms_df.itertuples(index=False))
        ]
        prompt_df = pd.DataFrame(prompt_rows)

        # Atomic append to staging: read → concat → atomic write
        if staging_path.exists():
            try:
                existing = pd.read_csv(staging_path)
                combined = pd.concat([existing, prompt_df], ignore_index=True)
            except Exception:
                combined = prompt_df   # staging corrupted — restart from this prompt
        else:
            combined = prompt_df
        atomic_csv_write(combined, staging_path)

    # Aggregate across all prompts from staging
    full = pd.read_csv(staging_path)
    return full.groupby(["model", "binom"], as_index=False)["preference"].mean()


@torch.inference_mode()
def compute_validation_perplexity(model, tokenizer, max_tokens: int = 4096) -> float:
    """
    Validation perplexity on the WikiText-2 test set.
    Strides through a concatenated test text with the model's context window.
    Works with both single-GPU and device_map='auto' multi-GPU models.
    """
    first_dev = _first_device(model)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           split="test", trust_remote_code=True)
    text = "\n".join(t for t in dataset["text"] if t.strip())

    input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    input_ids = input_ids[:max_tokens]

    stride   = 512
    seq_len  = input_ids.size(0)
    max_ctx  = getattr(model.config, "n_positions",
                       getattr(model.config, "max_position_embeddings", 1024))
    window   = min(max_ctx, 1024)
    nll_sum  = 0.0
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end        = min(begin + window, seq_len)
        target_len = end - max(begin, prev_end)
        if target_len <= 0:
            prev_end = end
            continue

        chunk   = input_ids[begin:end].unsqueeze(0).to(first_dev)
        outputs = model(chunk, labels=chunk)
        chunk_nll = outputs.loss.item() * (end - begin - 1)
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
    all_unique  = human[["Alpha", "Nonalpha"]].drop_duplicates(subset="Alpha")
    binoms_df   = (all_unique
                   .dropna(subset=["Alpha", "Nonalpha"])
                   .reset_index(drop=True))
    n_dropped   = len(all_unique) - len(binoms_df)
    dropped_rows = all_unique[all_unique[["Alpha", "Nonalpha"]].isna().any(axis=1)]

    drop_msg = (
        f"Binomials dropped (NaN in Alpha or Nonalpha): {n_dropped} / {len(all_unique)}\n"
        + (("\n".join(f"  {r.Alpha!r}  |  {r.Nonalpha!r}"
                      for r in dropped_rows.itertuples()))
           if n_dropped > 0 else "  (none)")
    )
    print(drop_msg)

    LOG_PATH = SCRIPT_DIR / "get_model_prefs.log"
    with open(LOG_PATH, "a") as log:
        import datetime
        log.write(f"\n{'='*60}\n")
        log.write(f"Run: {datetime.datetime.now().isoformat()}\n")
        log.write(drop_msg + "\n")

    print(f"  {len(binoms_df)} unique binomials retained  (log → {LOG_PATH})\n")

    expected_binoms = set(binoms_df["Alpha"])

    # Load existing outputs for resume detection
    done_prefs         = pd.read_csv(OUT_PREFS)           if OUT_PREFS.exists()           else pd.DataFrame()
    done_prefs_by_prompt = pd.read_csv(OUT_PREFS_BY_PROMPT) if OUT_PREFS_BY_PROMPT.exists() else pd.DataFrame()
    done_ppl   = pd.read_csv(OUT_PPL)   if OUT_PPL.exists()   else pd.DataFrame()
    done_ppl_models = set(done_ppl["model"]) if not done_ppl.empty else set()

    for model_id, cfg in MODEL_CONFIGS.items():
        if cfg.get("skip", False):
            print(f"Skipping {model_id} (skip=True)")
            continue

        model_safe   = model_id.replace("/", "_")
        staging_path = STAGING_DIR / f"{model_safe}_staging.csv"

        print("=" * 60)
        print(f"Model: {model_id}  ({cfg['family']}, {cfg['params']})")
        print("=" * 60)

        # ── Check if this model is already fully done in both output files ────
        completed_prompts = get_completed_prompts(staging_path, expected_binoms)
        prefs_done = (not done_prefs.empty and
                      model_id in done_prefs.get("model", pd.Series()).values and
                      expected_binoms.issubset(
                          set(done_prefs[done_prefs["model"] == model_id]["binom"])))
        ppl_done = model_id in done_ppl_models

        if prefs_done and ppl_done:
            print("  ✅ Already fully scored and saved — skipping.\n")
            continue

        all_prompts_staged = len(completed_prompts) == len(LIST_OF_PROMPTS)

        # ── Track success flags for cache cleanup ─────────────────────────────
        prefs_done_now = prefs_done  # already done before this run
        ppl_done_now   = ppl_done

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model     = None
        tmp_cache = str(Path.home() / ".cache" / "hf_prefs" / model_safe)
        Path(tmp_cache).mkdir(parents=True, exist_ok=True)
        try:
            if not all_prompts_staged:
                # Need model to score missing prompts
                multi_gpu = cfg.get("multi_gpu", False)
                dtype     = torch.float16 if device == "cuda" else torch.float32
                print("  Loading model ...")
                if multi_gpu:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        cache_dir=tmp_cache,
                        max_memory={0: "75GiB", 1: "75GiB"},
                    ).eval()
                    print(f"  Loaded with device_map='auto' across {torch.cuda.device_count()} GPUs")
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=dtype,
                        low_cpu_mem_usage=True, cache_dir=tmp_cache,
                    ).to(device).eval()
            else:
                print("  All prompts already in staging — skipping model load.")

            # ── Binomial preferences ──────────────────────────────────────────
            if not prefs_done:
                print("  Scoring binomial preferences ...")
                prefs_df = score_binomials(
                    model, tokenizer, binoms_df, model_id, staging_path
                )
                prefs_df["model_family"] = cfg["family"]
                prefs_df["model_params"] = cfg["params"]
                prefs_df["model_label"]  = cfg["label"]

                # Atomic incremental save: append this model's rows (averaged)
                combined_prefs = (
                    pd.concat([done_prefs, prefs_df], ignore_index=True)
                    if not done_prefs.empty else prefs_df
                )
                atomic_csv_write(combined_prefs, OUT_PREFS)
                done_prefs = combined_prefs
                print(f"  ✅ Preferences saved → {OUT_PREFS}")

                # Also save per-prompt preferences for mixed-model analysis
                raw_staging = pd.read_csv(staging_path)
                raw_staging["model_family"] = cfg["family"]
                raw_staging["model_params"] = cfg["params"]
                raw_staging["model_label"]  = cfg["label"]
                combined_by_prompt = (
                    pd.concat([done_prefs_by_prompt, raw_staging], ignore_index=True)
                    if not done_prefs_by_prompt.empty else raw_staging
                )
                atomic_csv_write(combined_by_prompt, OUT_PREFS_BY_PROMPT)
                done_prefs_by_prompt = combined_by_prompt
                print(f"  ✅ Per-prompt preferences saved → {OUT_PREFS_BY_PROMPT}")
                prefs_done_now = True  # ← mark prefs as successfully completed

            # ── Validation perplexity ─────────────────────────────────────────
            if not ppl_done:
                if model is None:
                    # Rare case: prefs already done but ppl missing — need model
                    multi_gpu = cfg.get("multi_gpu", False)
                    dtype     = torch.float16 if device == "cuda" else torch.float32
                    print("  Loading model for perplexity ...")
                    if multi_gpu:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id, dtype=torch.float16, device_map="auto",
                            low_cpu_mem_usage=True, cache_dir=tmp_cache,
                            max_memory={0: "75GiB", 1: "75GiB"},
                        ).eval()
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id, torch_dtype=dtype,
                            low_cpu_mem_usage=True, cache_dir=tmp_cache,
                        ).to(device).eval()

                print("  Computing validation perplexity (WikiText-2 test) ...")
                ppl = compute_validation_perplexity(model, tokenizer)
                print(f"  Perplexity: {ppl:.2f}")

                ppl_row = pd.DataFrame([{
                    "model":        model_id,
                    "model_family": cfg["family"],
                    "model_params": cfg["params"],
                    "model_label":  cfg["label"],
                    "perplexity":   ppl,
                }])
                combined_ppl = (
                    pd.concat([done_ppl, ppl_row], ignore_index=True)
                    if not done_ppl.empty else ppl_row
                )
                atomic_csv_write(combined_ppl, OUT_PPL)
                done_ppl = combined_ppl
                done_ppl_models.add(model_id)
                print(f"  ✅ Perplexity saved → {OUT_PPL}")
                ppl_done_now = True  # ← mark perplexity as successfully completed

        finally:
            if model is not None:
                del model
            if device == "cuda":
                torch.cuda.empty_cache()

        # Only delete the cache if both tasks completed successfully this run.
        # If either failed (e.g. OOM), the cache is preserved so we don't
        # re-download on the next attempt.
        if prefs_done_now and ppl_done_now:
            shutil.rmtree(tmp_cache, ignore_errors=True)
            print(f"  🗑️  Model cache deleted.")

        print()

    print("\n✅ All models done.")
    print(f"   Preferences → {OUT_PREFS}")
    print(f"   Perplexity  → {OUT_PPL}")
    print(f"   Staging dir → {STAGING_DIR}  (can be deleted if no longer needed)")


if __name__ == "__main__":
    main()