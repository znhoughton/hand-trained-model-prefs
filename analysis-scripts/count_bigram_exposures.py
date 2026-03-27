"""
count_bigram_exposures.py
--------------------------
Mirrors count_corpus_exposures.py to extract MARGINAL bigram counts at each
training checkpoint, rather than full "W1 and W2" trigram counts.

For each attested binomial ("W1 and W2"), we count:
    w1_and   — times W1 appeared immediately before "and" (in any "W1 and X" context)
    and_w1   — times W1 appeared immediately after  "and" (in any "X and W1" context)
    w2_and   — times W2 appeared immediately before "and"
    and_w2   — times W2 appeared immediately after  "and"
    w1_total — total corpus occurrences of W1 (needed for P(and|W1) normalisation)
    w2_total — total corpus occurrences of W2

Outputs (../Data/):
    babylm_bigram_exposures.csv   columns: tokens, binom, w1_and, and_w1, w2_and, and_w2, w1_total, w2_total
    c4_bigram_exposures.csv       same columns

Parallelisation (Linux/fork):
    BabyLM blocks are split across --workers processes for the scan step.
    C4 blocks are buffered (--c4-buffer blocks at a time) and scanned in parallel;
    checkpoint snapshots are taken at buffer boundaries (off by at most
    --c4-buffer blocks — negligible relative to checkpoint spacing).

Usage:
    python count_bigram_exposures.py                        # both corpora, auto workers
    python count_bigram_exposures.py --skip-c4              # BabyLM only
    python count_bigram_exposures.py --workers 16           # limit parallelism
    python count_bigram_exposures.py --c4-buffer 5000       # smaller C4 buffer
    python count_bigram_exposures.py --babylm-tokenizer <path_or_hf_id>

Requirements:
    pip install pandas numpy datasets transformers torch
"""

import os
import re
import math
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from itertools import chain
from pathlib import Path

os.chdir(Path(__file__).parent)

# ── Paths ──────────────────────────────────────────────────────────────────────
BINOMS_CSV  = Path("../Data/nonce_and_attested_binoms.csv")
CORR_CSV    = Path("../Data/processed/nonce_correlations.csv")
OUT_BABYLM  = Path("../Data/babylm_bigram_exposures.csv")
OUT_C4      = Path("../Data/c4_bigram_exposures.csv")

# ── Training config (mirrors count_corpus_exposures.py) ────────────────────────
BABYLM_DATASET     = "znhoughton/babylm-150m-v3"
BABYLM_TRAIN_SPLIT = "train[5%:]"
BABYLM_SEED        = 964
BABYLM_EPOCHS      = 64
WORLD_SIZE         = 2
BLOCK_SIZE         = 1024
SCAN_BATCH         = 500   # blocks decoded per batch within each worker

C4_DATASET     = "znhoughton/c4-subset-10B-tokens"
C4_TRAIN_SPLIT = "train"

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skip-c4", action="store_true",
                    help="Skip C4 (set C4 counts to 0).")
parser.add_argument("--babylm-tokenizer",
                    default="znhoughton/opt-babylm-125m-64eps-seed964",
                    help="HuggingFace model/tokenizer ID or local path.")
parser.add_argument("--workers", type=int, default=None,
                    help="Number of parallel worker processes (default: all cores).")
parser.add_argument("--c4-buffer", type=int, default=10_000,
                    help="Number of C4 blocks to buffer before parallel scan (default: 10000).")
args = parser.parse_args()

N_WORKERS  = args.workers or mp.cpu_count()
C4_BUFFER  = args.c4_buffer
print(f"Parallelism: {N_WORKERS} workers  |  C4 buffer: {C4_BUFFER:,} blocks")


# ── 1. Load binomials — only attested ones need bigram counts ──────────────────
print("Loading binomials...")
binoms_df = pd.read_csv(BINOMS_CSV)

# We want marginal counts for every word that appears in any attested binomial.
# Non-attested binomials are excluded (they have no preference data anyway).
attested = binoms_df[binoms_df["Attested"].astype(str).str.strip() == "1"].copy()
alpha_list = attested["Alpha"].str.lower().str.strip().tolist()
w1_list    = attested["Word1"].str.lower().str.strip().tolist()
w2_list    = attested["Word2"].str.lower().str.strip().tolist()
n_binoms   = len(attested)
print(f"  {n_binoms} attested binomials.")

# Build a word → index map over the union of all W1 and W2 words.
# Multiple binomials may share a word (e.g. "cats and dogs" / "cats and fish").
all_words = sorted(set(w1_list) | set(w2_list))
word_to_idx = {w: i for i, w in enumerate(all_words)}
n_words = len(all_words)
print(f"  {n_words} unique words across all binomials.")

# For each binomial, store indices into the word array.
w1_idx = np.array([word_to_idx[w] for w in w1_list], dtype=np.int32)
w2_idx = np.array([word_to_idx[w] for w in w2_list], dtype=np.int32)


# ── 2. Load checkpoint token lists (same source as count_corpus_exposures.py) ──
print("Loading checkpoint token lists...")
corr = pd.read_csv(CORR_CSV)

babylm_steps = (
    corr[corr["model"].str.contains("babylm", case=False)]
    [["tokens"]].drop_duplicates().sort_values("tokens").reset_index(drop=True)
)
c4_steps = (
    corr[corr["model"].str.contains("opt-c4", case=False)]
    [["tokens"]].drop_duplicates().sort_values("tokens").reset_index(drop=True)
)
print(f"  BabyLM: {len(babylm_steps)} checkpoints  "
      f"({babylm_steps['tokens'].min():,.0f} – {babylm_steps['tokens'].max():,.0f})")
print(f"  C4:     {len(c4_steps)} checkpoints  "
      f"({c4_steps['tokens'].min():,.0f} – {c4_steps['tokens'].max():,.0f})")


# ── Shared helpers ─────────────────────────────────────────────────────────────
try:
    import torch
    from datasets import load_dataset as hf_load
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("pip install torch datasets transformers")

_strip = re.compile(r"[^a-z]")


def scan_text(text):
    """
    For a decoded text block, return three dicts:
        w_and   : {word_idx: count}  — word appeared immediately BEFORE "and"
        and_w   : {word_idx: count}  — word appeared immediately AFTER  "and"
        w_total : {word_idx: count}  — word appeared anywhere in the text
    Only words in our target set (word_to_idx) are counted.
    w_total is needed for the properly-normalised bigram log-odds:
        log[P(and|W1) * P(W2|and)] - log[P(and|W2) * P(W1|and)]
      = log(w1_and) + log(and_w2) - log(w2_and) - log(and_w1)
        + log(W2_total) - log(W1_total)
    """
    words = [_strip.sub("", w) for w in text.lower().split()]
    words = [w for w in words if w]
    w_and_counts   = {}
    and_w_counts   = {}
    w_total_counts = {}
    for word in words:
        if word in word_to_idx:
            idx = word_to_idx[word]
            w_total_counts[idx] = w_total_counts.get(idx, 0) + 1
    for j in range(len(words) - 2):
        if words[j + 1] != "and":
            continue
        left, right = words[j], words[j + 2]
        if left in word_to_idx:
            idx = word_to_idx[left]
            w_and_counts[idx] = w_and_counts.get(idx, 0) + 1
        if right in word_to_idx:
            idx = word_to_idx[right]
            and_w_counts[idx] = and_w_counts.get(idx, 0) + 1
    return w_and_counts, and_w_counts, w_total_counts


# ── Parallel worker functions (module-level so multiprocessing can pickle them) ─
# On Linux the default start method is "fork", so workers inherit all globals
# (blm_lm, tokenizer, word_to_idx, n_words, etc.) via copy-on-write.

def _scan_block_chunk(args):
    """
    BabyLM worker: decode and scan blocks[start:end] from the pre-loaded blm_lm.
    Returns (start, wa_chunk, aw_chunk, wt_chunk) where each chunk has shape
    (end - start, n_words) — one row per block.  These are assembled by the
    main process into the full block_w_and / block_and_w / block_w_total arrays
    before the shuffle replay.
    """
    start, end = args
    chunk = end - start
    wa = np.zeros((chunk, n_words), dtype=np.int16)
    aw = np.zeros((chunk, n_words), dtype=np.int16)
    wt = np.zeros((chunk, n_words), dtype=np.int32)
    for bi in range(start, end, SCAN_BATCH):
        batch_end = min(bi + SCAN_BATCH, end)
        batch = blm_lm[bi:batch_end]
        texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        for local_i, text in enumerate(texts):
            gi = bi - start + local_i
            wa_, aw_, wt_ = scan_text(text)
            for idx, cnt in wa_.items(): wa[gi, idx] = min(cnt, 32767)
            for idx, cnt in aw_.items(): aw[gi, idx] = min(cnt, 32767)
            for idx, cnt in wt_.items(): wt[gi, idx] = min(cnt, 2_147_483_647)
    return start, wa, aw, wt


def _scan_texts(texts):
    """
    C4 worker: scan a list of already-decoded text strings.
    Returns (wa, aw, wt) arrays of shape (n_words,) — the summed counts for
    all texts in the list.  The main process accumulates these into running
    totals and snapshots at checkpoint boundaries.
    """
    wa = np.zeros(n_words, dtype=np.int64)
    aw = np.zeros(n_words, dtype=np.int64)
    wt = np.zeros(n_words, dtype=np.int64)
    for text in texts:
        wa_, aw_, wt_ = scan_text(text)
        for idx, cnt in wa_.items(): wa[idx] += cnt
        for idx, cnt in aw_.items(): aw[idx] += cnt
        for idx, cnt in wt_.items(): wt[idx] += cnt
    return wa, aw, wt


print(f"\nLoading tokenizer from '{args.babylm_tokenizer}'...")
tokenizer = AutoTokenizer.from_pretrained(args.babylm_tokenizer)


def group_texts(examples):
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    return {k: [t[i: i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
            for k, t in concatenated.items()}


def build_output_df(sorted_T, w_and_arr, and_w_arr, w_total_arr):
    """
    Convert word-level count arrays into a per-binomial DataFrame.

    w_and_arr   : shape (n_ckpts, n_words)
    and_w_arr   : shape (n_ckpts, n_words)
    w_total_arr : shape (n_ckpts, n_words)  — total corpus count of each word
    """
    n_ckpts = len(sorted_T)
    rows_tokens   = np.repeat(sorted_T, n_binoms)
    rows_binom    = np.tile(alpha_list, n_ckpts)
    rows_w1_and   = w_and_arr[:,   w1_idx].ravel()
    rows_and_w1   = and_w_arr[:,   w1_idx].ravel()
    rows_w2_and   = w_and_arr[:,   w2_idx].ravel()
    rows_and_w2   = and_w_arr[:,   w2_idx].ravel()
    rows_w1_total = w_total_arr[:, w1_idx].ravel()
    rows_w2_total = w_total_arr[:, w2_idx].ravel()
    return pd.DataFrame({
        "tokens":   rows_tokens,
        "binom":    rows_binom,
        "w1_and":   rows_w1_and,
        "and_w1":   rows_and_w1,
        "w2_and":   rows_w2_and,
        "and_w2":   rows_and_w2,
        "w1_total": rows_w1_total,
        "w2_total": rows_w2_total,
    })


# ── 3. BabyLM: exact replay ────────────────────────────────────────────────────
print(f"\n=== BabyLM: Exact Replay ===")
print(f"  Loading {BABYLM_DATASET} split='{BABYLM_TRAIN_SPLIT}'")

raw_blm  = hf_load(BABYLM_DATASET, split=BABYLM_TRAIN_SPLIT)
text_col = "text" if "text" in raw_blm.column_names else raw_blm.column_names[0]

blm_tok = raw_blm.map(lambda ex: tokenizer(ex[text_col]), batched=True,
                       remove_columns=raw_blm.column_names,
                       load_from_cache_file=True, desc="Tokenizing")
blm_lm  = blm_tok.map(group_texts, batched=True,
                       load_from_cache_file=True,
                       desc=f"Grouping into {BLOCK_SIZE}-token blocks")

N = len(blm_lm)
print(f"  {N:,} blocks × {BLOCK_SIZE} tokens = {N * BLOCK_SIZE:,} tokens per epoch")

# Pre-compute per-block bigram counts for all target words (parallel).
print(f"  Scanning {N:,} blocks for bigram counts using {N_WORKERS} workers...")
block_w_and   = np.zeros((N, n_words), dtype=np.int16)
block_and_w   = np.zeros((N, n_words), dtype=np.int16)
block_w_total = np.zeros((N, n_words), dtype=np.int32)

chunk_size = math.ceil(N / N_WORKERS)
chunks = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]
print(f"  {len(chunks)} chunks of ~{chunk_size:,} blocks each.")

with mp.Pool(N_WORKERS) as pool:
    for i, (start, wa, aw, wt) in enumerate(pool.imap_unordered(_scan_block_chunk, chunks)):
        end = start + len(wa)
        block_w_and[start:end]   = wa
        block_and_w[start:end]   = aw
        block_w_total[start:end] = wt
        print(f"    chunk {i+1}/{len(chunks)} done (blocks {start:,}–{end:,})")

total_w_and_epoch   = block_w_and.sum(axis=0).astype(np.float64)
total_and_w_epoch   = block_and_w.sum(axis=0).astype(np.float64)
total_w_total_epoch = block_w_total.sum(axis=0).astype(np.float64)
print(f"  Done. {(total_w_and_epoch > 0).sum()} / {n_words} words found before 'and'; "
      f"{(total_and_w_epoch > 0).sum()} found after 'and'.")

# Reproduce per-epoch DistributedSampler shuffle.
print("  Replaying training sequence...")


def get_epoch_shuffle(epoch):
    g = torch.Generator()
    g.manual_seed(BABYLM_SEED + epoch)
    return torch.randperm(N, generator=g).numpy()


sorted_blm_T = sorted(babylm_steps["tokens"].tolist())
n_blm_ckpts  = len(sorted_blm_T)
blm_w_and    = np.zeros((n_blm_ckpts, n_words), dtype=np.float64)
blm_and_w    = np.zeros((n_blm_ckpts, n_words), dtype=np.float64)
blm_w_total  = np.zeros((n_blm_ckpts, n_words), dtype=np.float64)

current_epoch         = -1
epoch_shuffle         = None
epoch_partial_w_and   = np.zeros(n_words, dtype=np.float64)
epoch_partial_and_w   = np.zeros(n_words, dtype=np.float64)
epoch_partial_w_total = np.zeros(n_words, dtype=np.float64)
prev_partial          = 0

for ci, T in enumerate(sorted_blm_T):
    total_blocks = int(T) // BLOCK_SIZE
    full_epochs  = total_blocks // N
    partial      = total_blocks % N

    if full_epochs != current_epoch:
        epoch_partial_w_and[:]   = 0.0
        epoch_partial_and_w[:]   = 0.0
        epoch_partial_w_total[:] = 0.0
        prev_partial  = 0
        current_epoch = full_epochs
        epoch_shuffle = None

    if partial > prev_partial:
        if epoch_shuffle is None:
            epoch_shuffle = get_epoch_shuffle(current_epoch)
        new_idxs = epoch_shuffle[prev_partial:partial]
        epoch_partial_w_and   += block_w_and[new_idxs].sum(axis=0)
        epoch_partial_and_w   += block_and_w[new_idxs].sum(axis=0)
        epoch_partial_w_total += block_w_total[new_idxs].sum(axis=0)
        prev_partial = partial

    blm_w_and[ci]   = full_epochs * total_w_and_epoch   + epoch_partial_w_and
    blm_and_w[ci]   = full_epochs * total_and_w_epoch   + epoch_partial_and_w
    blm_w_total[ci] = full_epochs * total_w_total_epoch + epoch_partial_w_total

babylm_exp = build_output_df(sorted_blm_T, blm_w_and, blm_and_w, blm_w_total)
babylm_exp.to_csv(OUT_BABYLM, index=False)
print(f"  Saved {OUT_BABYLM.name}  ({len(babylm_exp):,} rows)")


# ── 4. C4: exact streaming replay ─────────────────────────────────────────────
sorted_c4_T = sorted(c4_steps["tokens"].tolist())
n_c4_ckpts  = len(sorted_c4_T)
c4_w_and    = np.zeros((n_c4_ckpts, n_words), dtype=np.float64)
c4_and_w    = np.zeros((n_c4_ckpts, n_words), dtype=np.float64)
c4_w_total  = np.zeros((n_c4_ckpts, n_words), dtype=np.float64)

if args.skip_c4:
    print("\n--skip-c4: C4 counts set to 0.")
else:
    print(f"\n=== C4: Exact Streaming Replay ===")
    print(f"  Streaming {C4_DATASET} split='{C4_TRAIN_SPLIT}'")
    print("  No shuffle — fixed parquet-shard order.")
    print("  Expected: 30–90 min.\n")

    c4_raw      = hf_load(C4_DATASET, split=C4_TRAIN_SPLIT, streaming=True)
    c4_text_col = "text" if "text" in c4_raw.column_names else c4_raw.column_names[0]

    c4_tok = c4_raw.map(lambda ex: tokenizer(ex[c4_text_col]), batched=True,
                         remove_columns=c4_raw.column_names)
    c4_lm  = c4_tok.map(group_texts, batched=True)

    ckpt_blocks = {int(T) // BLOCK_SIZE: ci for ci, T in enumerate(sorted_c4_T)}
    max_block   = max(ckpt_blocks)

    cumulative_w_and   = np.zeros(n_words, dtype=np.float64)
    cumulative_and_w   = np.zeros(n_words, dtype=np.float64)
    cumulative_w_total = np.zeros(n_words, dtype=np.float64)
    block_count = 0
    buffer_texts = []   # decoded texts waiting to be scanned

    def flush_buffer(pool):
        """Scan all texts in buffer in parallel, accumulate into cumulative arrays."""
        if not buffer_texts:
            return
        chunk_size = math.ceil(len(buffer_texts) / N_WORKERS)
        chunks = [buffer_texts[i:i + chunk_size]
                  for i in range(0, len(buffer_texts), chunk_size)]
        for wa, aw, wt in pool.map(_scan_texts, chunks):
            cumulative_w_and[:]   += wa
            cumulative_and_w[:]   += aw
            cumulative_w_total[:] += wt
        buffer_texts.clear()

    with mp.Pool(N_WORKERS) as pool:
        for block in c4_lm:
            block_count += 1
            buffer_texts.append(
                tokenizer.decode(block["input_ids"], skip_special_tokens=True)
            )

            # Flush and snapshot at checkpoint boundaries.
            # Checkpoints are ~500M tokens apart; C4_BUFFER is much smaller,
            # so snapshots are off by at most C4_BUFFER blocks (negligible).
            if len(buffer_texts) >= C4_BUFFER or block_count in ckpt_blocks:
                flush_buffer(pool)

            if block_count in ckpt_blocks:
                ci = ckpt_blocks[block_count]
                c4_w_and[ci]   = cumulative_w_and.copy()
                c4_and_w[ci]   = cumulative_and_w.copy()
                c4_w_total[ci] = cumulative_w_total.copy()

            if block_count % 500_000 == 0:
                print(f"  {block_count:,} blocks | "
                      f"{block_count * BLOCK_SIZE / 1e9:.2f}B tokens", flush=True)

            if block_count >= max_block:
                break

        flush_buffer(pool)  # drain any remaining buffer

    print(f"  C4 replay done: {block_count:,} blocks, "
          f"{block_count * BLOCK_SIZE:,} training tokens.")
    print(f"  {(cumulative_w_and > 0).sum()} / {n_words} words found before 'and'; "
          f"{(cumulative_and_w > 0).sum()} found after 'and'.")

c4_exp = build_output_df(sorted_c4_T, c4_w_and, c4_and_w, c4_w_total)
c4_exp.to_csv(OUT_C4, index=False)
print(f"  Saved {OUT_C4.name}  ({len(c4_exp):,} rows)")


# ── 5. Sanity check ────────────────────────────────────────────────────────────
print("\n=== Sanity check: top 5 binomials by w1_and at final checkpoint ===")
for label, exp_df in [("BabyLM", babylm_exp), ("C4", c4_exp)]:
    final = exp_df[exp_df["tokens"] == exp_df["tokens"].max()]
    top   = final.nlargest(5, "w1_and")[["binom", "w1_and", "and_w1", "w2_and", "and_w2", "w1_total", "w2_total"]]
    print(f"\n{label}:")
    print(top.to_string(index=False))

print("\nAll done.")
print("Next: update analysis2.Rmd to join babylm_bigram_exposures.csv / c4_bigram_exposures.csv")
print("      and compute bigram_logodds = log(w1_and) + log(and_w2) - log(w2_and) - log(and_w1)")
print("                                  + log(w2_total) - log(w1_total)")
