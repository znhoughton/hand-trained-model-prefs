#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_ns_surprisal.py
===================
Compute per-word surprisal from HuggingFace causal LMs on the
Natural Stories Corpus (Futrell et al., 2021), following the
methodology of Oh & Schuler (2023).

For each model:
  - Per-word surprisal  = sum of -log2 P(subword token | context)
                          for all BPE tokens making up the word
  - Corpus perplexity   = exp(mean token NLL) on the full corpus text

Context-window overflow is handled by overlapping windows:
  the second half of window n serves as the first half of window n+1
  (Oh & Schuler 2023, Section 3.2).

Outputs (in ns_surprisal/):
  <safe_model_name>.csv     — item, zone, word, surprisal (bits)
  ns_perplexity.csv         — model, family, params, perplexity (nats)

Natural Stories text is downloaded automatically from GitHub.
Reading time data must be placed at:
  natural_stories/processed_RTs.tsv
Download from: https://github.com/languageMIT/naturalstories
  → naturalstories_RTs/processed_RTs.tsv

Run:
  python get_ns_surprisal.py                    # all models
  python get_ns_surprisal.py --models gpt2 gpt2-medium
  python get_ns_surprisal.py --skip-existing    # skip already-computed

Requirements:
  pip install torch transformers tqdm requests
"""

import argparse
import math
import re
import sys
from pathlib import Path

import requests
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
NS_DIR     = SCRIPT_DIR / "natural_stories"
OUT_DIR    = SCRIPT_DIR / "ns_surprisal"
PPL_CSV    = OUT_DIR / "ns_perplexity.csv"
NS_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

NS_CORPUS_URL = (
    "https://raw.githubusercontent.com/languageMIT/naturalstories"
    "/master/naturalstories_RTS/all_stories.tok"
)
RT_TSV_PATH = NS_DIR / "processed_RTs.tsv"   # fallback: extract words from RT file

# ── Model registry ────────────────────────────────────────────────────────────
# ctx: maximum context window (tokens); skip: set True to exclude
MODEL_CONFIGS = {
    # GPT-2 family
    "gpt2":                                        {"params": "124M",   "family": "GPT-2",   "ctx": 1024},
    "gpt2-medium":                                 {"params": "355M",   "family": "GPT-2",   "ctx": 1024},
    "gpt2-large":                                  {"params": "774M",   "family": "GPT-2",   "ctx": 1024},
    "gpt2-xl":                                     {"params": "1542M",  "family": "GPT-2",   "ctx": 1024},
    # GPT-Neo family
    "EleutherAI/gpt-neo-125m":                     {"params": "125M",   "family": "GPT-Neo", "ctx": 2048},
    "EleutherAI/gpt-neo-1.3B":                     {"params": "1300M",  "family": "GPT-Neo", "ctx": 2048},
    "EleutherAI/gpt-neo-2.7B":                     {"params": "2700M",  "family": "GPT-Neo", "ctx": 2048},
    # OPT family
    "facebook/opt-125m":                           {"params": "125M",   "family": "OPT",     "ctx": 2048},
    "facebook/opt-350m":                           {"params": "350M",   "family": "OPT",     "ctx": 2048},
    "facebook/opt-1.3b":                           {"params": "1300M",  "family": "OPT",     "ctx": 2048},
    "facebook/opt-2.7b":                           {"params": "2700M",  "family": "OPT",     "ctx": 2048},
    "facebook/opt-6.7b":                           {"params": "6700M",  "family": "OPT",     "ctx": 2048},
    "facebook/opt-13b":                            {"params": "13000M", "family": "OPT",     "ctx": 2048},
    "facebook/opt-30b":                            {"params": "30000M", "family": "OPT",     "ctx": 2048},
    # BabyLM (OPT architecture, trained on BabyLM corpus)
    "znhoughton/opt-babylm-125m-64eps-seed964":    {"params": "125M",   "family": "BabyLM",  "ctx": 2048},
    "znhoughton/opt-babylm-350m-64eps-seed964":    {"params": "350M",   "family": "BabyLM",  "ctx": 2048},
    "znhoughton/opt-babylm-1.3b-64eps-seed964":    {"params": "1300M",  "family": "BabyLM",  "ctx": 2048},
    # C4 (OPT architecture, trained on C4 subset)
    "znhoughton/opt-c4-125m-seed964":              {"params": "125M",   "family": "C4",      "ctx": 2048},
    "znhoughton/opt-c4-350m-seed964":              {"params": "350M",   "family": "C4",      "ctx": 2048},
    "znhoughton/opt-c4-1.3b-seed964":              {"params": "1300M",  "family": "C4",      "ctx": 2048},
}

# ── Natural Stories text download + parsing ───────────────────────────────────

def download_corpus(path: Path) -> None:
    """Download Natural Stories corpus text from GitHub if not already present.
    Tries several candidate URLs (branch name and filename vary across repo history).
    Falls back to extracting the word list from the RT TSV if already downloaded.
    """
    if path.exists():
        return

    # Try GitHub raw URL
    print(f"Downloading from {NS_CORPUS_URL} …")
    try:
        r = requests.get(NS_CORPUS_URL, timeout=60)
        if r.status_code == 200:
            path.write_bytes(r.content)
            print(f"  Downloaded → {path}")
            return
        print(f"  HTTP {r.status_code}")
    except Exception as e:
        print(f"  Failed ({e})")

    # Fallback: derive corpus text from the RT TSV (item/zone/word already present)
    if RT_TSV_PATH.exists():
        print(f"GitHub download failed. Extracting word list from {RT_TSV_PATH} …")
        import csv
        rows = []
        with RT_TSV_PATH.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            # normalise header keys to lowercase
            for raw_row in reader:
                row = {k.lower(): v for k, v in raw_row.items()}
                item = row.get("item") or row.get("story")
                zone = row.get("zone") or row.get("word_num") or row.get("position")
                word = row.get("word")
                try:
                    if item and zone and word:
                        rows.append(f"!{int(item)}!{int(zone)}\t{word}\n")
                except ValueError:
                    continue
        # Deduplicate (one row per item/zone)
        seen = set()
        unique = []
        for line in rows:
            key = line.split("\t")[0]
            if key not in seen:
                seen.add(key)
                unique.append(line)
        unique.sort(key=lambda l: tuple(int(x) for x in re.findall(r"\d+", l.split("\t")[0])))
        # Write as plain TSV matching the format of all_stories.tok
        lines = ["word\tzone\titem\n"]
        for line in unique:
            tag, word = line.split("\t")
            nums = re.findall(r"\d+", tag)
            item_n, zone_n = nums[0], nums[1]
            lines.append(f"{word.strip()}\t{zone_n}\t{item_n}\n")
        path.write_text("".join(lines), encoding="utf-8")
        print(f"  Extracted {len(unique)} words → {path}")
        return

    raise RuntimeError(
        "Could not obtain Natural Stories corpus text.\n"
        "Options:\n"
        "  1. Place processed_RTs.tsv in:\n"
        f"     {RT_TSV_PATH}\n"
        "     (download from github.com/languageMIT/naturalstories → naturalstories_RTs/)\n"
        "  2. Or manually download all_stories.tok and place it at:\n"
        f"     {path}"
    )


def parse_corpus(path: Path) -> pd.DataFrame:
    """
    Parse all_stories.tok into a DataFrame with columns:
      item (int), zone (int), word (str)

    The file is a TSV with header: word  zone  item
    """
    df = pd.read_csv(path, sep="\t", dtype={"word": str, "zone": int, "item": int})
    df = df[["item", "zone", "word"]].sort_values(["item", "zone"]).reset_index(drop=True)
    return df


# ── Surprisal computation ─────────────────────────────────────────────────────

def safe_model_name(model_id: str) -> str:
    """Convert HuggingFace model ID to safe filename."""
    return model_id.replace("/", "_")


def compute_surprisal(
    model_id: str,
    corpus_df: pd.DataFrame,
    ctx: int,
    device: torch.device,
) -> tuple[pd.DataFrame, float]:
    """
    Compute per-word surprisal (bits) and corpus perplexity (nats) for one model.

    Returns
    -------
    surprisal_df : DataFrame with columns item, zone, word, surprisal
    perplexity   : float — exp(mean NLL in nats) over all tokens in corpus
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    # Load tokenizer and model
    print("  Loading tokenizer …")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("  Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    records = []
    total_nll   = 0.0   # summed NLL in nats
    total_ntoks = 0     # total subword tokens processed

    # Process one story at a time
    for item_id, story_df in tqdm(
        corpus_df.groupby("item"), desc="  Stories", total=corpus_df["item"].nunique()
    ):
        story_df = story_df.sort_values("zone").reset_index(drop=True)
        words = story_df["word"].tolist()
        zones = story_df["zone"].tolist()

        # Build the full token sequence for this story, tracking word boundaries
        # We prepend the BOS token (or an empty prefix) so the first word gets
        # proper conditioning.
        word_token_spans = []   # list of (start_idx, end_idx) in token sequence
        token_ids = [tok.bos_token_id] if tok.bos_token_id is not None else []
        offset = len(token_ids)

        for word in words:
            # Tokenize without special tokens; add leading space for GPT-style
            # BPE where words are separated by Ġ / Ċ prefixes
            word_toks = tok.encode(" " + word, add_special_tokens=False)
            start = offset
            token_ids.extend(word_toks)
            offset += len(word_toks)
            word_token_spans.append((start, offset))

        # Convert to tensor
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        n_tokens = len(token_tensor)

        # Collect per-token NLL using sliding window to handle long stories
        # Following Oh & Schuler (2023): second half of window n = first half of n+1
        token_nlls = torch.full((n_tokens,), float("nan"))

        stride = ctx // 2
        start = 0
        while start < n_tokens:
            end = min(start + ctx, n_tokens)
            chunk = token_tensor[start:end].unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(chunk, labels=chunk)
                # out.logits shape: (1, seq_len, vocab_size)
                logits = out.logits[0]   # (seq_len, vocab_size)

            # Shift: logit[i] predicts token[i+1]
            shift_logits = logits[:-1].float()
            shift_labels = chunk[0, 1:].to(shift_logits.device)
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs[
                torch.arange(len(shift_labels)), shift_labels
            ]  # (seq_len - 1,) in nats

            # Indices in the global sequence that we can trust this window for.
            # We skip the first half of the window (it was the "context" half),
            # unless this is the very first window.
            if start == 0:
                valid_start_in_chunk = 0
            else:
                valid_start_in_chunk = stride

            # token_log_probs[i] = NLL for token at position (start + i + 1) in global
            for i in range(valid_start_in_chunk, len(token_log_probs)):
                global_pos = start + i + 1   # position of the *predicted* token
                if global_pos < n_tokens and torch.isnan(token_nlls[global_pos]):
                    token_nlls[global_pos] = -token_log_probs[i].item()

            if end == n_tokens:
                break
            start += stride

        # Accumulate corpus-level NLL (skip position 0 = BOS, no prediction for it)
        valid_nlls = token_nlls[1:]   # positions 1…n_tokens-1
        valid_nlls = valid_nlls[~torch.isnan(valid_nlls)]
        total_nll   += valid_nlls.sum().item()
        total_ntoks += len(valid_nlls)

        # Aggregate token NLLs to word-level surprisal (nats → bits via /log(2))
        log2 = math.log(2)
        for (tok_start, tok_end), zone, word in zip(word_token_spans, zones, words):
            span_nlls = token_nlls[tok_start:tok_end]
            if torch.any(torch.isnan(span_nlls)):
                word_surp = float("nan")
            else:
                # Sum NLL over subword tokens, convert nats → bits
                word_surp = span_nlls.sum().item() / log2
            records.append({
                "item":      item_id,
                "zone":      zone,
                "word":      word,
                "surprisal": word_surp,
            })

    surprisal_df = pd.DataFrame(records)
    perplexity = math.exp(total_nll / total_ntoks) if total_ntoks > 0 else float("nan")
    print(f"  Corpus perplexity: {perplexity:.4f}")

    # Free GPU memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return surprisal_df, perplexity


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model IDs to run (default: all in MODEL_CONFIGS)."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip models whose output CSV already exists."
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU (slow but useful for small models without a GPU)."
    )
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Download / parse corpus text
    corpus_path = NS_DIR / "all_stories.tok"
    download_corpus(corpus_path)   # tries GitHub URLs, then RT-TSV fallback
    print(f"Parsing corpus …")
    corpus_df = parse_corpus(corpus_path)
    print(f"  {len(corpus_df)} words across {corpus_df['item'].nunique()} stories")

    # Determine which models to run
    if args.models:
        missing = [m for m in args.models if m not in MODEL_CONFIGS]
        if missing:
            print(f"WARNING: unknown models (will skip): {missing}", file=sys.stderr)
        run_models = [m for m in args.models if m in MODEL_CONFIGS]
    else:
        run_models = list(MODEL_CONFIGS.keys())

    # Load existing perplexity CSV if it exists
    if PPL_CSV.exists():
        ppl_df = pd.read_csv(PPL_CSV)
    else:
        ppl_df = pd.DataFrame(columns=["model", "family", "params", "perplexity"])

    for model_id in run_models:
        out_csv = OUT_DIR / f"{safe_model_name(model_id)}.csv"

        if args.skip_existing and out_csv.exists():
            print(f"\nSkipping {model_id} (output exists)")
            continue

        cfg = MODEL_CONFIGS[model_id]
        try:
            surp_df, ppl = compute_surprisal(
                model_id, corpus_df, ctx=cfg["ctx"], device=device
            )
        except Exception as e:
            print(f"\nERROR computing surprisal for {model_id}: {e}", file=sys.stderr)
            continue

        # Save surprisal CSV
        surp_df.to_csv(out_csv, index=False)
        print(f"  Saved surprisal → {out_csv}")

        # Update perplexity CSV
        new_row = pd.DataFrame([{
            "model":   model_id,
            "family":  cfg["family"],
            "params":  cfg["params"],
            "perplexity": ppl,
        }])
        ppl_df = ppl_df[ppl_df["model"] != model_id]   # overwrite if re-run
        ppl_df = pd.concat([ppl_df, new_row], ignore_index=True)
        ppl_df.to_csv(PPL_CSV, index=False)
        print(f"  Perplexity table updated → {PPL_CSV}")

    print(f"\nDone. Surprisal CSVs in: {OUT_DIR}")
    print(f"Perplexity summary:       {PPL_CSV}")
    print()
    print("Next step: run reading_time_delta_ll.R")
    print("  Requires reading time data at:")
    print(f"  {NS_DIR / 'processed_RTs.tsv'}")
    print("  Download from:")
    print("  https://github.com/languageMIT/naturalstories/tree/master/naturalstories_RTs")


if __name__ == "__main__":
    main()
