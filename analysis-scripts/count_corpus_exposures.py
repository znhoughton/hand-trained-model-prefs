"""
count_corpus_exposures.py

Exact replay of training data to get cumulative binomial exposure counts at
each checkpoint step.

BabyLM (non-streaming, 64 epochs):
  - Dataset znhoughton/babylm-150m-v3 has splits "train"/"dev" (no "validation"),
    so train_autoreg.py used split="train[5%:]" (the 5% fallback code).
  - Loads that split, applies tokenize + group_texts (block_size=1024).
  - Reproduces the DistributedSampler shuffle per epoch (seed + epoch = 964+e).
  - Outputs exact integer cumulative counts.

C4 (streaming, 1 epoch):
  - Dataset znhoughton/c4-subset-10B-tokens already has a "validation" split,
    so train_autoreg.py used the pre-defined split="train" (all 17.7M docs).
  - The Trainer does NOT shuffle streaming datasets (IterableDataset).
    IterableDatasetShard only shards across GPUs — no shuffle. Document order is
    fixed by parquet-shard reading order on HuggingFace Hub.
  - Applies tokenize + group_texts (streaming), decodes each 1024-token block,
    scans within-block text for binomials (exact; no cross-block approximation).
  - Outputs exact integer cumulative counts.

NOTE: BabyLM exact replay requires the same PyTorch major version used during
training (torch.randperm may differ across major versions for the same seed).

Outputs (../Data/):
    babylm_step_exposures.csv   columns: tokens, binom, alpha_seen, beta_seen
    c4_step_exposures.csv       columns: tokens, binom, alpha_seen, beta_seen

Usage:
    python count_corpus_exposures.py             # both corpora
    python count_corpus_exposures.py --skip-c4   # BabyLM only
    python count_corpus_exposures.py --babylm-tokenizer <path_or_hf_id>

Requirements:
    pip install pandas numpy datasets transformers torch
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from itertools import chain
from pathlib import Path

os.chdir(Path(__file__).parent)

# ── Paths ──────────────────────────────────────────────────────────────────────
BINOMS_CSV       = Path("../Data/nonce_and_attested_binoms.csv")
CORRELATIONS_CSV = Path("../Data/processed/nonce_correlations.csv")
OUT_BABYLM       = Path("../Data/babylm_step_exposures.csv")
OUT_C4           = Path("../Data/c4_step_exposures.csv")

# ── Training config ────────────────────────────────────────────────────────────
# BabyLM: babylm-150m-v3 has "train"/"dev" splits (no "validation"),
#   so train_autoreg.py's fallback fires → training used split="train[5%:]"
BABYLM_DATASET    = "znhoughton/babylm-150m-v3"
BABYLM_TRAIN_SPLIT = "train[5%:]"
BABYLM_SEED       = 964
BABYLM_EPOCHS     = 64
WORLD_SIZE        = 2
BLOCK_SIZE        = 1024

# C4: c4-subset-10B-tokens already has a "validation" split,
#   so train_autoreg.py used the dataset's pre-defined "train" split directly
C4_DATASET        = "znhoughton/c4-subset-10B-tokens"
C4_TRAIN_SPLIT    = "train"

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skip-c4", action="store_true",
                    help="Skip C4 (set C4 counts to 0).")
parser.add_argument("--babylm-tokenizer",
                    default="znhoughton/opt-babylm-125m-64eps-seed964",
                    help="HuggingFace model/tokenizer ID or local path.")
args = parser.parse_args()


# ── 1. Load binomials ──────────────────────────────────────────────────────────
print("Loading binomials...")
binoms          = pd.read_csv(BINOMS_CSV)
alpha_list      = binoms["Alpha"].str.lower().str.strip().tolist()
nonalpha_list   = binoms["Nonalpha"].str.lower().str.strip().tolist()
alpha_to_idx    = {s: i for i, s in enumerate(alpha_list)}
nonalpha_to_idx = {s: i for i, s in enumerate(nonalpha_list)}
n_binoms        = len(binoms)
print(f"  {n_binoms} binomials.")


# ── 2. Load checkpoint token lists ────────────────────────────────────────────
print("Loading checkpoint token lists...")
corr = pd.read_csv(CORRELATIONS_CSV)

babylm_steps = (
    corr[corr["model"].str.contains("babylm", case=False)]
    [["tokens"]].drop_duplicates().sort_values("tokens").reset_index(drop=True)
)
c4_steps = (
    corr[corr["model"].str.contains("opt-c4", case=False)]
    [["tokens"]].drop_duplicates().sort_values("tokens").reset_index(drop=True)
)
print(f"  BabyLM: {len(babylm_steps)} unique token counts  "
      f"({babylm_steps['tokens'].min():,.0f} to {babylm_steps['tokens'].max():,.0f})")
print(f"  C4:     {len(c4_steps)} unique token counts  "
      f"({c4_steps['tokens'].min():,.0f} to {c4_steps['tokens'].max():,.0f})")


# ── Shared imports and helpers ─────────────────────────────────────────────────
try:
    import torch
    from datasets import load_dataset as hf_load
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("pip install torch datasets transformers")

_strip = re.compile(r"[^a-z]")

def scan_text(text):
    """Returns ({alpha_idx: count}, {nonalpha_idx: count}) for one text span."""
    words = [_strip.sub("", w) for w in text.lower().split()]
    words = [w for w in words if w]
    ha, hb = {}, {}
    for j in range(len(words) - 2):
        if words[j + 1] != "and":
            continue
        tg = f"{words[j]} and {words[j + 2]}"
        if tg in alpha_to_idx:
            idx = alpha_to_idx[tg]
            ha[idx] = ha.get(idx, 0) + 1
        elif tg in nonalpha_to_idx:
            idx = nonalpha_to_idx[tg]
            hb[idx] = hb.get(idx, 0) + 1
    return ha, hb

print(f"\nLoading tokenizer from '{args.babylm_tokenizer}'...")
tokenizer = AutoTokenizer.from_pretrained(args.babylm_tokenizer)

def group_texts(examples):
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total = len(concatenated["input_ids"])
    total = (total // BLOCK_SIZE) * BLOCK_SIZE
    return {k: [t[i:i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
            for k, t in concatenated.items()}


# ── 3. BabyLM: exact replay ────────────────────────────────────────────────────
print(f"\n=== BabyLM: Exact Replay ===")
print(f"  Loading {BABYLM_DATASET} split='{BABYLM_TRAIN_SPLIT}'")
print("  (Cached after first run; may take a few minutes initially)")

raw_blm   = hf_load(BABYLM_DATASET, split=BABYLM_TRAIN_SPLIT)
text_col  = "text" if "text" in raw_blm.column_names else raw_blm.column_names[0]

def tokenize_fn(examples):
    return tokenizer(examples[text_col])

print("  Tokenizing and grouping into blocks...")
blm_tok = raw_blm.map(tokenize_fn, batched=True,
                       remove_columns=raw_blm.column_names,
                       load_from_cache_file=True, desc="Tokenizing")
blm_lm  = blm_tok.map(group_texts, batched=True,
                       load_from_cache_file=True,
                       desc=f"Grouping into {BLOCK_SIZE}-token blocks")

N = len(blm_lm)
print(f"  {N:,} blocks x {BLOCK_SIZE} tokens = {N * BLOCK_SIZE:,} tokens per epoch")

# Pre-compute per-block binomial counts (decode each block → scan)
print(f"  Scanning {N:,} blocks for binomials (may take 5-15 min)...")
block_alpha = np.zeros((N, n_binoms), dtype=np.int16)
block_beta  = np.zeros((N, n_binoms), dtype=np.int16)

SCAN_BATCH = 500
for batch_start in range(0, N, SCAN_BATCH):
    if batch_start % 10000 == 0:
        print(f"    {batch_start:,} / {N:,} blocks ...")
    batch_end = min(batch_start + SCAN_BATCH, N)
    batch = blm_lm[batch_start:batch_end]
    texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    for local_i, text in enumerate(texts):
        gi = batch_start + local_i
        ha, hb = scan_text(text)
        for idx, cnt in ha.items():
            block_alpha[gi, idx] = min(cnt, 32767)
        for idx, cnt in hb.items():
            block_beta[gi, idx] = min(cnt, 32767)

total_alpha_epoch = block_alpha.sum(axis=0).astype(np.float64)
total_beta_epoch  = block_beta.sum(axis=0).astype(np.float64)
print(f"  Done: {(total_alpha_epoch > 0).sum()} alpha forms, "
      f"{(total_beta_epoch > 0).sum()} beta forms found.")

# Reproduce per-epoch DistributedSampler shuffle and accumulate at checkpoints
print("  Replaying training sequence...")

def get_epoch_shuffle(epoch):
    g = torch.Generator()
    g.manual_seed(BABYLM_SEED + epoch)
    return torch.randperm(N, generator=g).numpy()

sorted_blm_T = sorted(babylm_steps["tokens"].tolist())
n_blm_ckpts  = len(sorted_blm_T)
alpha_blm    = np.zeros((n_blm_ckpts, n_binoms), dtype=np.float64)
beta_blm     = np.zeros((n_blm_ckpts, n_binoms), dtype=np.float64)

current_epoch       = -1
epoch_shuffle       = None
epoch_partial_alpha = np.zeros(n_binoms, dtype=np.float64)
epoch_partial_beta  = np.zeros(n_binoms, dtype=np.float64)
prev_partial        = 0

for ci, T in enumerate(sorted_blm_T):
    total_blocks = int(T) // BLOCK_SIZE
    full_epochs  = total_blocks // N
    partial      = total_blocks % N

    if full_epochs != current_epoch:
        epoch_partial_alpha[:] = 0.0
        epoch_partial_beta[:] = 0.0
        prev_partial  = 0
        current_epoch = full_epochs
        epoch_shuffle = None

    if partial > prev_partial:
        if epoch_shuffle is None:
            epoch_shuffle = get_epoch_shuffle(current_epoch)
        new_idxs = epoch_shuffle[prev_partial:partial]
        epoch_partial_alpha += block_alpha[new_idxs].sum(axis=0)
        epoch_partial_beta  += block_beta[new_idxs].sum(axis=0)
        prev_partial = partial

    alpha_blm[ci] = full_epochs * total_alpha_epoch + epoch_partial_alpha
    beta_blm[ci]  = full_epochs * total_beta_epoch  + epoch_partial_beta

babylm_exp = pd.DataFrame({
    "tokens":     np.repeat(sorted_blm_T, n_binoms),
    "binom":      np.tile(alpha_list, n_blm_ckpts),
    "alpha_seen": alpha_blm.ravel(),
    "beta_seen":  beta_blm.ravel(),
})
babylm_exp.to_csv(OUT_BABYLM, index=False)
print(f"  Saved {OUT_BABYLM.name}  ({len(babylm_exp):,} rows)")


# ── 4. C4: exact streaming replay ─────────────────────────────────────────────
# c4-subset-10B-tokens has a pre-defined "validation" split, so train_autoreg.py
# used the full "train" split (17.7M docs) without the 5% holdout logic.
#
# The Trainer does NOT shuffle streaming datasets. IterableDatasetShard only
# shards across GPUs (no shuffle). Document order = fixed parquet-shard order.
#
# Exact block-level approach: apply tokenize + group_texts (streaming), decode
# each assembled 1024-token block, scan decoded text. This correctly handles
# cross-document trigrams that fall within the same block (no approximation).

sorted_c4_T = sorted(c4_steps["tokens"].tolist())
n_c4_ckpts  = len(sorted_c4_T)
alpha_c4    = np.zeros((n_c4_ckpts, n_binoms), dtype=np.float64)
beta_c4     = np.zeros((n_c4_ckpts, n_binoms), dtype=np.float64)

if args.skip_c4:
    print("\n--skip-c4: C4 counts set to 0.")
else:
    print(f"\n=== C4: Exact Streaming Replay ===")
    print(f"  Streaming {C4_DATASET} split='{C4_TRAIN_SPLIT}'")
    print("  No shuffle — fixed parquet-shard order.")
    print("  Tokenizing + grouping + decoding blocks. Expected: 30-90 min.\n")

    c4_raw = hf_load(C4_DATASET, split=C4_TRAIN_SPLIT, streaming=True)
    c4_text_col = "text" if "text" in c4_raw.column_names else c4_raw.column_names[0]

    def tokenize_fn_c4(examples):
        return tokenizer(examples[c4_text_col])

    # Apply tokenize + group_texts as streaming maps (mirrors training pipeline)
    c4_tok = c4_raw.map(tokenize_fn_c4, batched=True,
                         remove_columns=c4_raw.column_names)
    c4_lm  = c4_tok.map(group_texts, batched=True)

    # Precompute checkpoint boundaries in blocks (T tokens = T/BLOCK_SIZE blocks)
    ckpt_blocks = {int(T) // BLOCK_SIZE: ci for ci, T in enumerate(sorted_c4_T)}
    max_block   = max(ckpt_blocks)

    cumulative_alpha = np.zeros(n_binoms, dtype=np.float64)
    cumulative_beta  = np.zeros(n_binoms, dtype=np.float64)
    block_count = 0
    doc_count   = 0

    for block in c4_lm:
        block_count += 1
        text = tokenizer.decode(block["input_ids"], skip_special_tokens=True)
        ha, hb = scan_text(text)
        for idx, cnt in ha.items(): cumulative_alpha[idx] += cnt
        for idx, cnt in hb.items(): cumulative_beta[idx] += cnt

        if block_count in ckpt_blocks:
            ci = ckpt_blocks[block_count]
            alpha_c4[ci] = cumulative_alpha.copy()
            beta_c4[ci]  = cumulative_beta.copy()

        if block_count % 500_000 == 0:
            print(f"  {block_count:,} blocks | "
                  f"{block_count * BLOCK_SIZE:,} tokens | "
                  f"{sum(alpha_c4[:, 0] > 0 if n_binoms > 0 else [0])} ckpts done")

        if block_count >= max_block:
            break

    print(f"  C4 replay done: {block_count:,} blocks, "
          f"{block_count * BLOCK_SIZE:,} training tokens.")
    print(f"  {(cumulative_alpha > 0).sum()} alpha forms, "
          f"{(cumulative_beta  > 0).sum()} beta forms found.")

c4_exp = pd.DataFrame({
    "tokens":     np.repeat(sorted_c4_T, n_binoms),
    "binom":      np.tile(alpha_list, n_c4_ckpts),
    "alpha_seen": alpha_c4.ravel(),
    "beta_seen":  beta_c4.ravel(),
})
c4_exp.to_csv(OUT_C4, index=False)
print(f"  Saved {OUT_C4.name}  ({len(c4_exp):,} rows)")


# ── 5. Sanity check ────────────────────────────────────────────────────────────
print("\n=== Sanity check: top 5 alpha-seen at final checkpoint ===")
for label, exp_df in [("BabyLM (exact)", babylm_exp), ("C4 (exact)", c4_exp)]:
    final = exp_df[exp_df["tokens"] == exp_df["tokens"].max()]
    top   = final.nlargest(5, "alpha_seen")[["binom", "alpha_seen", "beta_seen"]]
    print(f"\n{label}:")
    print(top.to_string(index=False))

print("\nAll done.")
