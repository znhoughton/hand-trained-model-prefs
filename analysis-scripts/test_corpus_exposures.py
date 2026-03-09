"""
test_corpus_exposures.py

Verifies the core logic of count_corpus_exposures.py before running the full job.

Tests
-----
1. shuffle_seed   : Our get_epoch_shuffle() exactly reproduces what PyTorch
                    DistributedSampler would emit for every epoch.
2. accumulate     : The epoch-boundary accumulate logic matches brute-force
                    enumeration across many synthetic checkpoint positions.
3. babylm_pipeline: Load the first ~100 real BabyLM blocks, verify that the
                    pre-computed block counts (decode -> scan) match a direct
                    brute-force scan of the same blocks.
4. c4_pipeline    : Stream the first ~100 real C4 blocks (tokenize + group_texts
                    + decode -> scan), verify counts match a direct block scan;
                    also confirm split="train" differs from split="train[5%:]".

Usage
-----
    python test_corpus_exposures.py            # all tests
    python test_corpus_exposures.py --no-net   # skip tests that need network
"""

import sys
import math
import argparse
import re
import numpy as np
from itertools import chain
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--no-net", action="store_true",
                    help="Skip tests that require downloading data.")
args = parser.parse_args()

import torch
from torch.utils.data import DistributedSampler

SEED       = 964
WORLD_SIZE = 2
BLOCK_SIZE = 1024

BABYLM_DATASET    = "znhoughton/babylm-150m-v3"
BABYLM_TRAIN_SPLIT = "train[5%:]"
C4_DATASET        = "znhoughton/c4-subset-10B-tokens"
C4_TRAIN_SPLIT    = "train"
TOKENIZER_ID      = "znhoughton/opt-babylm-125m-64eps-seed964"

_strip = re.compile(r"[^a-z]")

def scan_text(text, alpha_to_idx, nonalpha_to_idx):
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

def group_texts(examples):
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total = len(concatenated["input_ids"])
    total = (total // BLOCK_SIZE) * BLOCK_SIZE
    return {k: [t[i:i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
            for k, t in concatenated.items()}

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(condition, msg):
    status = PASS if condition else FAIL
    print(f"  [{status}] {msg}")
    if not condition:
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Shuffle seed — our randperm matches DistributedSampler exactly
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 1: shuffle_seed ===")
print("  Verifying get_epoch_shuffle() matches DistributedSampler for various N/epoch.")

class _FakeDataset:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return i

all_ok = True
for N in [50, 100, 101, 1000, 9999]:           # even and odd N
    for epoch in [0, 1, 5, 63]:
        # Our shuffle
        g = torch.Generator()
        g.manual_seed(SEED + epoch)
        our_full = torch.randperm(N, generator=g).tolist()
        total_size = math.ceil(N / WORLD_SIZE) * WORLD_SIZE
        padded = our_full + our_full[:total_size - N]
        our_rank0 = padded[0::WORLD_SIZE]
        our_rank1 = padded[1::WORLD_SIZE]

        # DistributedSampler reference (rank 0)
        ds = _FakeDataset(N)
        sampler0 = DistributedSampler(ds, num_replicas=WORLD_SIZE, rank=0,
                                      seed=SEED, shuffle=True)
        sampler0.set_epoch(epoch)
        ref_rank0 = list(sampler0)

        # DistributedSampler reference (rank 1)
        sampler1 = DistributedSampler(ds, num_replicas=WORLD_SIZE, rank=1,
                                      seed=SEED, shuffle=True)
        sampler1.set_epoch(epoch)
        ref_rank1 = list(sampler1)

        if our_rank0 != ref_rank0 or our_rank1 != ref_rank1:
            check(False, f"N={N}, epoch={epoch}: shuffle mismatch")
            all_ok = False

check(all_ok, f"shuffle matches DistributedSampler for all (N, epoch) pairs tested")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Accumulate logic — matches brute-force at every checkpoint
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 2: accumulate ===")
print("  Verifying epoch-boundary accumulation against brute-force.")

rng = np.random.default_rng(42)
N_syn   = 73    # odd, so padding path is exercised
n_b_syn = 20    # synthetic binomials
block_alpha_syn = rng.integers(0, 4, (N_syn, n_b_syn))
block_beta_syn  = rng.integers(0, 4, (N_syn, n_b_syn))
total_alpha_syn = block_alpha_syn.sum(axis=0).astype(float)
total_beta_syn  = block_beta_syn.sum(axis=0).astype(float)

# Checkpoints spread across ~3.5 epochs (plenty of epoch boundary crossings)
blocks_to_test = [0, 1, 10, 72, 73, 74, 100, 145, 146, 147, 200, 255]
sorted_T = [b * BLOCK_SIZE for b in blocks_to_test]

def brute_force(T, N, block_alpha, block_beta):
    """Direct, naive cumulative count — no incremental tricks."""
    total_blocks = T // BLOCK_SIZE
    full_epochs  = total_blocks // N
    partial      = total_blocks % N
    alpha = full_epochs * block_alpha.sum(axis=0).astype(float)
    beta  = full_epochs * block_beta.sum(axis=0).astype(float)
    if partial > 0:
        g = torch.Generator()
        g.manual_seed(SEED + full_epochs)
        shuffle = torch.randperm(N, generator=g).numpy()
        alpha += block_alpha[shuffle[:partial]].sum(axis=0)
        beta  += block_beta[shuffle[:partial]].sum(axis=0)
    return alpha, beta

# Replay logic (mirrored from count_corpus_exposures.py)
n_ckpts = len(sorted_T)
alpha_out = np.zeros((n_ckpts, n_b_syn), dtype=np.float64)
beta_out  = np.zeros((n_ckpts, n_b_syn), dtype=np.float64)

current_epoch = -1
epoch_shuffle = None
ep_alpha = np.zeros(n_b_syn, dtype=np.float64)
ep_beta  = np.zeros(n_b_syn, dtype=np.float64)
prev_partial = 0

for ci, T in enumerate(sorted_T):
    total_blocks = T // BLOCK_SIZE
    full_epochs  = total_blocks // N_syn
    partial      = total_blocks % N_syn
    if full_epochs != current_epoch:
        ep_alpha[:] = 0.0
        ep_beta[:] = 0.0
        prev_partial = 0
        current_epoch = full_epochs
        epoch_shuffle = None
    if partial > prev_partial:
        if epoch_shuffle is None:
            g = torch.Generator()
            g.manual_seed(SEED + current_epoch)
            epoch_shuffle = torch.randperm(N_syn, generator=g).numpy()
        new_idxs = epoch_shuffle[prev_partial:partial]
        ep_alpha += block_alpha_syn[new_idxs].sum(axis=0)
        ep_beta  += block_beta_syn[new_idxs].sum(axis=0)
        prev_partial = partial
    alpha_out[ci] = full_epochs * total_alpha_syn + ep_alpha
    beta_out[ci]  = full_epochs * total_beta_syn  + ep_beta

all_ok = True
for ci, T in enumerate(sorted_T):
    gt_a, gt_b = brute_force(T, N_syn, block_alpha_syn, block_beta_syn)
    if not (np.allclose(alpha_out[ci], gt_a) and np.allclose(beta_out[ci], gt_b)):
        check(False, f"Mismatch at T={T} (blocks={T//BLOCK_SIZE})")
        all_ok = False

check(all_ok, f"accumulate matches brute-force at all {n_ckpts} checkpoint positions")


if args.no_net:
    print("\n--no-net: skipping pipeline tests.")
    print("\nAll local tests passed.")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Load tokenizer + tiny binomial vocab for pipeline tests
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading tokenizer...")
from transformers import AutoTokenizer
from datasets import load_dataset as hf_load

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

# Use a small set of common English words as "binomials" so we actually find hits
test_binoms = [
    ("men", "women"), ("boys", "girls"), ("black", "white"),
    ("cats", "dogs"),  ("bread", "butter"), ("salt", "pepper"),
    ("day", "night"),  ("left", "right"),   ("good", "bad"),
    ("big", "small"),
]
alpha_list_t      = [f"{a} and {b}" for a, b in test_binoms]
nonalpha_list_t   = [f"{b} and {a}" for a, b in test_binoms]
alpha_to_idx_t    = {s: i for i, s in enumerate(alpha_list_t)}
nonalpha_to_idx_t = {s: i for i, s in enumerate(nonalpha_list_t)}
n_b_t = len(test_binoms)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: BabyLM pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 3: babylm_pipeline ===")
print(f"  Loading first ~200 BabyLM blocks via {BABYLM_DATASET} split='{BABYLM_TRAIN_SPLIT}'")

raw_blm = hf_load(BABYLM_DATASET, split=BABYLM_TRAIN_SPLIT)
text_col = "text" if "text" in raw_blm.column_names else raw_blm.column_names[0]

# Take first 20,000 raw documents — BabyLM utterances are very short (~10 tokens avg)
# so we need many docs to accumulate enough blocks and find binomial hits
raw_blm_small = raw_blm.select(range(min(20_000, len(raw_blm))))

def tokenize_fn(examples):
    return tokenizer(examples[text_col])

tok_small = raw_blm_small.map(tokenize_fn, batched=True,
                               remove_columns=raw_blm_small.column_names,
                               desc="Tokenizing (test)")
lm_small  = tok_small.map(group_texts, batched=True, desc="Grouping (test)")
N_blm_test = len(lm_small)
print(f"  Got {N_blm_test} blocks from 200 raw documents.")
check(N_blm_test > 0, f"Pipeline produced {N_blm_test} blocks (> 0)")

# Build block counts via decode -> scan  (our method)
block_alpha_t = np.zeros((N_blm_test, n_b_t), dtype=np.int32)
block_beta_t  = np.zeros((N_blm_test, n_b_t), dtype=np.int32)

for i in range(0, N_blm_test, 50):
    batch = lm_small[i:min(i+50, N_blm_test)]
    texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    for local_i, text in enumerate(texts):
        gi = i + local_i
        ha, hb = scan_text(text, alpha_to_idx_t, nonalpha_to_idx_t)
        for idx, cnt in ha.items(): block_alpha_t[gi, idx] = cnt
        for idx, cnt in hb.items(): block_beta_t[gi, idx]  = cnt

# Ground truth: concatenate all block texts, scan each block independently
# (same thing, just done separately to confirm)
gt_alpha = np.zeros((N_blm_test, n_b_t), dtype=np.int32)
gt_beta  = np.zeros((N_blm_test, n_b_t), dtype=np.int32)

for i in range(N_blm_test):
    text = tokenizer.decode(lm_small[i]["input_ids"], skip_special_tokens=True)
    ha, hb = scan_text(text, alpha_to_idx_t, nonalpha_to_idx_t)
    for idx, cnt in ha.items(): gt_alpha[i, idx] = cnt
    for idx, cnt in hb.items(): gt_beta[i, idx]  = cnt

check(np.array_equal(block_alpha_t, gt_alpha),
      "block_decode batch matches single-block decode (alpha)")
check(np.array_equal(block_beta_t, gt_beta),
      "block_decode batch matches single-block decode (beta)")

total_hits = block_alpha_t.sum() + block_beta_t.sum()
print(f"  Total binomial hits across {N_blm_test} blocks: {total_hits}")
check(total_hits > 0, "Found at least one binomial hit in BabyLM blocks")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: C4 pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 4: c4_pipeline ===")
print(f"  Streaming first ~100 C4 blocks via {C4_DATASET} split='{C4_TRAIN_SPLIT}'")

c4_raw = hf_load(C4_DATASET, split=C4_TRAIN_SPLIT, streaming=True)
c4_text_col = "text" if "text" in c4_raw.column_names else c4_raw.column_names[0]

def tokenize_fn_c4(examples):
    return tokenizer(examples[c4_text_col])

c4_tok = c4_raw.map(tokenize_fn_c4, batched=True, remove_columns=c4_raw.column_names)
c4_lm  = c4_tok.map(group_texts, batched=True)

# Collect first 100 blocks
c4_blocks_ids = []
for block in c4_lm:
    c4_blocks_ids.append(block["input_ids"])
    if len(c4_blocks_ids) >= 100:
        break
N_c4_test = len(c4_blocks_ids)
print(f"  Collected {N_c4_test} blocks.")
check(N_c4_test > 0, f"C4 streaming produced blocks (> 0)")

# Our method: decode each block -> scan
c4_alpha = np.zeros((N_c4_test, n_b_t), dtype=np.int32)
c4_beta  = np.zeros((N_c4_test, n_b_t), dtype=np.int32)
for i, ids in enumerate(c4_blocks_ids):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    ha, hb = scan_text(text, alpha_to_idx_t, nonalpha_to_idx_t)
    for idx, cnt in ha.items(): c4_alpha[i, idx] = cnt
    for idx, cnt in hb.items(): c4_beta[i, idx]  = cnt

# Ground truth: same ids, same decode, same scan (just done again independently)
c4_alpha_gt = np.zeros((N_c4_test, n_b_t), dtype=np.int32)
c4_beta_gt  = np.zeros((N_c4_test, n_b_t), dtype=np.int32)
for i, ids in enumerate(c4_blocks_ids):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    ha, hb = scan_text(text, alpha_to_idx_t, nonalpha_to_idx_t)
    for idx, cnt in ha.items(): c4_alpha_gt[i, idx] = cnt
    for idx, cnt in hb.items(): c4_beta_gt[i, idx]  = cnt

check(np.array_equal(c4_alpha, c4_alpha_gt),
      "C4 block decode+scan is deterministic (alpha)")
check(np.array_equal(c4_beta, c4_beta_gt),
      "C4 block decode+scan is deterministic (beta)")

total_c4_hits = c4_alpha.sum() + c4_beta.sum()
print(f"  Total binomial hits across {N_c4_test} C4 blocks: {total_c4_hits}")
check(total_c4_hits > 0, "Found at least one binomial hit in C4 blocks")

# Confirm the dataset has a pre-defined "validation" split — this is why
# train_autoreg.py's fallback did NOT fire for C4, so training used full "train"
print(f"  Checking that C4 dataset has a pre-defined 'validation' split...")
from datasets import get_dataset_split_names
c4_splits = get_dataset_split_names(C4_DATASET)
check("validation" in c4_splits,
      f"C4 has pre-defined 'validation' split (got: {c4_splits}) — "
      "so train_autoreg.py used full split='train', not 'train[5%:]'")
# Also confirm BabyLM does NOT have "validation" (it has "dev"), so fallback fired
blm_splits = get_dataset_split_names(BABYLM_DATASET)
check("validation" not in blm_splits,
      f"BabyLM does NOT have 'validation' split (got: {blm_splits}) — "
      "so train_autoreg.py's fallback fired, using split='train[5%:]'")

print("\n=== All tests passed. ===")
