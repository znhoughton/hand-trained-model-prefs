"""
compute_c4_trigrams.py
----------------------
Mirrors the C4 streaming pipeline from count_corpus_exposures.py to extract
"X and Y" trigram counts from the C4 training corpus.

Only "X and Y" trigrams (middle token == "and") are stored. This is a targeted
subset of a full trigrams file, but it is exactly what compute_bigram_logodds.py
needs to compute positional bigram log-odds for each binomial.

The babylm_eng_trigrams.csv file covers all trigrams from that corpus; creating
an equivalent all-trigrams file for C4 (10B tokens) would be impractically large
(~tens of GB). The "and"-only subset is sufficient for the analysis.

Pipeline (mirrors count_corpus_exposures.py):
  1. Stream znhoughton/c4-subset-10B-tokens, split="train"
  2. Tokenize with the BabyLM tokenizer (same as training)
  3. group_texts into 1024-token blocks (same as training)
  4. Decode each block and scan for "X and Y" trigrams
  5. Accumulate counts; save as c4_eng_trigrams.csv

Output:
  ../Data/c4_eng_trigrams.csv  — columns: trigram, count
  (trigram format: "word1 and word2", same as babylm_eng_trigrams.csv)

Usage:
  python compute_c4_trigrams.py
  python compute_c4_trigrams.py --max-blocks 1000000  # process first 1M blocks (~1B tokens)
  python compute_c4_trigrams.py --tokenizer znhoughton/opt-babylm-125m-64eps-seed964

Requirements:
  pip install datasets transformers torch
"""

import os
import re
import csv
import argparse
from collections import defaultdict
from itertools import chain
from pathlib import Path

os.chdir(Path(__file__).parent)

# ── Paths ───────────────────────────────────────────────────────────────────────
OUT_CSV    = Path("../Data/c4_eng_trigrams.csv")
BLOCK_SIZE = 1024

# ── CLI ─────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", default="znhoughton/opt-babylm-125m-64eps-seed964",
                    help="HuggingFace model/tokenizer ID or local path (same as training).")
parser.add_argument("--max-blocks", type=int, default=None,
                    help="Stop after this many blocks (default: full corpus ~9.8M blocks).")
args = parser.parse_args()


# ── Load tokenizer ───────────────────────────────────────────────────────────────
try:
    from datasets import load_dataset as hf_load
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("pip install datasets transformers")

print(f"Loading tokenizer from '{args.tokenizer}'...")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

_strip = re.compile(r"[^a-z]")


def group_texts(examples):
    """Concatenate and chunk into fixed-size blocks (mirrors training pipeline)."""
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    return {k: [t[i: i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
            for k, t in concatenated.items()}


# ── Stream C4 ────────────────────────────────────────────────────────────────────
print("Streaming znhoughton/c4-subset-10B-tokens (split='train')...")
print("No shuffle — fixed parquet-shard order (mirrors training).\n")

c4_raw = hf_load("znhoughton/c4-subset-10B-tokens", split="train", streaming=True)
text_col = "text" if "text" in c4_raw.column_names else c4_raw.column_names[0]

c4_tok = c4_raw.map(lambda ex: tokenizer(ex[text_col]), batched=True,
                    remove_columns=c4_raw.column_names)
c4_lm  = c4_tok.map(group_texts, batched=True)

# ── Scan blocks and accumulate "X and Y" trigram counts ─────────────────────────
trigram_counts = defaultdict(int)
block_count    = 0
report_every   = 500_000

max_blocks = args.max_blocks  # None = full corpus

print(f"Scanning blocks for 'X and Y' trigrams...")
print(f"(Full corpus ≈ 9.8M blocks × {BLOCK_SIZE} tokens = ~10B tokens)")
if max_blocks:
    print(f"Limiting to {max_blocks:,} blocks (~{max_blocks * BLOCK_SIZE / 1e9:.1f}B tokens)\n")
else:
    print("Processing full corpus. Estimated time: 30–90 min.\n")

for block in c4_lm:
    text  = tokenizer.decode(block["input_ids"], skip_special_tokens=True)
    words = [_strip.sub("", w) for w in text.lower().split()]
    words = [w for w in words if w]

    for j in range(len(words) - 2):
        if words[j + 1] == "and":
            trigram_counts[f"{words[j]} and {words[j + 2]}"] += 1

    block_count += 1

    if block_count % report_every == 0:
        print(f"  {block_count:,} blocks | "
              f"{block_count * BLOCK_SIZE / 1e9:.2f}B tokens | "
              f"{len(trigram_counts):,} unique 'X and Y' trigrams")

    if max_blocks and block_count >= max_blocks:
        break

print(f"\nDone: {block_count:,} blocks, "
      f"{block_count * BLOCK_SIZE / 1e9:.2f}B tokens, "
      f"{len(trigram_counts):,} unique 'X and Y' trigrams.")

# ── Write output ─────────────────────────────────────────────────────────────────
print(f"Writing {OUT_CSV} ...")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["trigram", "count"])
    for trigram, count in sorted(trigram_counts.items(),
                                  key=lambda x: x[1], reverse=True):
        writer.writerow([trigram, count])

print(f"Saved {len(trigram_counts):,} rows → {OUT_CSV}")
print("Now run compute_bigram_logodds.py to compute bigram log-odds for all binomials.")
