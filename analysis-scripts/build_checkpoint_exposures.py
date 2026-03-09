"""
build_checkpoint_exposures.py

Combines all per-checkpoint, per-prompt model preference data with
cumulative corpus exposure counts from count_corpus_exposures.py.

Output: Data/processed/checkpoint_results_with_exposures.csv

One row per model × checkpoint × prompt × binomial.
Only BabyLM and C4 models are included (Pythia and OLMo models are skipped).

Columns in output:
  model, checkpoint, step, tokens, prompt, binom,
  alpha_logprob, nonalpha_logprob, preference,
  alpha_seen, beta_seen

alpha_seen / beta_seen = cumulative count of times the alpha/beta form
of that binomial appeared in the training data up to and including the
token count at that checkpoint.

NOTE ON SIZE:
  ~5,663 checkpoint files × 36,975 rows each ≈ 209 million rows total.
  The output CSV will be on the order of 25-30 GB.  If you only need
  a subset, see --model and --max-files flags below.

Usage:
    python build_checkpoint_exposures.py               # full run
    python build_checkpoint_exposures.py --model opt-babylm-125m-64eps-seed964
    python build_checkpoint_exposures.py --max-files 100   # quick test
"""

import os
import glob
import argparse
import pandas as pd
from pathlib import Path

os.chdir(Path(__file__).parent)

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR    = Path("../Data/checkpoint_results")
BABYLM_EXP_CSV = Path("../Data/babylm_step_exposures.csv")
C4_EXP_CSV     = Path("../Data/c4_step_exposures.csv")
OUT_CSV        = Path("../Data/processed/checkpoint_results_with_exposures.csv")

# ── Model name patterns ────────────────────────────────────────────────────────
BABYLM_PATTERN = "opt-babylm-*"
C4_PATTERN     = "opt-c4-*"

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default=None,
    help="Process only files matching this model name substring (e.g. 'opt-babylm-125m')."
)
parser.add_argument(
    "--max-files", type=int, default=None,
    help="Stop after processing this many checkpoint files (for testing)."
)
parser.add_argument(
    "--batch-size", type=int, default=200,
    help="Number of checkpoint files to concatenate per write batch (default 200)."
)
args = parser.parse_args()

# ── 1. Discover checkpoint files ───────────────────────────────────────────────
babylm_files = sorted(glob.glob(str(RESULTS_DIR / "opt-babylm-*.csv")))
c4_files     = sorted(glob.glob(str(RESULTS_DIR / "opt-c4-*.csv")))

if args.model:
    babylm_files = [f for f in babylm_files if args.model in f]
    c4_files     = [f for f in c4_files     if args.model in f]

all_files = babylm_files + c4_files

if args.max_files:
    all_files = all_files[:args.max_files]

n_babylm = sum(1 for f in all_files if "opt-babylm" in f)
n_c4     = sum(1 for f in all_files if "opt-c4"     in f)

print(f"Checkpoint files found: {n_babylm} BabyLM + {n_c4} C4 = {len(all_files)} total")
print(f"Expected output rows:   ~{len(all_files) * 36_975:,}")
print(f"Estimated output size:  ~{len(all_files) * 36_975 * 130 / 1e9:.1f} GB")

if len(all_files) == 0:
    raise FileNotFoundError(
        f"No checkpoint files found in {RESULTS_DIR}. "
        "Check that RESULTS_DIR is correct."
    )

# ── 2. Load exposure lookup tables ────────────────────────────────────────────
print("\nLoading exposure tables...")
babylm_exp = pd.read_csv(BABYLM_EXP_CSV)
c4_exp     = pd.read_csv(C4_EXP_CSV)

# Build fast lookup dicts: (tokens, binom) -> (alpha_seen, beta_seen)
# Using a single dict with tuple keys is faster than repeated DataFrame merges
# when the same (tokens, binom) pair is looked up many times.
print(f"  BabyLM exposure table: {len(babylm_exp):,} rows "
      f"({babylm_exp['tokens'].nunique():,} token counts)")
print(f"  C4 exposure table:     {len(c4_exp):,} rows "
      f"({c4_exp['tokens'].nunique():,} token counts)")

def build_lookup(exp_df):
    """Returns dict: (tokens, binom_lowercase) -> (alpha_seen, beta_seen).

    The exposure table stores binomials in lowercase (count_corpus_exposures.py
    applies .str.lower()), but checkpoint result files use original casing
    (e.g. 'Africa and Asia').  We key on lowercase to handle both.
    """
    lookup = {}
    for row in exp_df.itertuples(index=False):
        lookup[(row.tokens, row.binom.lower())] = (row.alpha_seen, row.beta_seen)
    return lookup

print("  Building BabyLM lookup dict...")
babylm_lookup = build_lookup(babylm_exp)
print("  Building C4 lookup dict...")
c4_lookup     = build_lookup(c4_exp)
del babylm_exp, c4_exp   # free memory

print("  Done.\n")

# ── 3. Process checkpoint files in batches and write output ───────────────────
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

KEEP_COLS = ["model", "checkpoint", "step", "tokens", "prompt",
             "binom", "alpha_logprob", "nonalpha_logprob", "preference"]

files_done = 0
header_written = False

with open(OUT_CSV, "w", newline="", encoding="utf-8") as out_fh:

    for batch_start in range(0, len(all_files), args.batch_size):
        batch_files = all_files[batch_start:batch_start + args.batch_size]

        # Read and concatenate this batch
        dfs = []
        for fpath in batch_files:
            try:
                df = pd.read_csv(fpath, usecols=lambda c: c in KEEP_COLS + ["revision"])
                # Drop the redundant 'revision' column (same as 'checkpoint')
                df = df[[c for c in KEEP_COLS if c in df.columns]]
                dfs.append(df)
            except Exception as e:
                print(f"  WARNING: could not read {fpath}: {e}")

        if not dfs:
            continue

        batch_df = pd.concat(dfs, ignore_index=True)

        # Determine which lookup to use per row based on model name
        is_babylm = batch_df["model"].str.contains("babylm", case=False, na=False)

        # Vectorised lookup: build two aligned Series for alpha_seen / beta_seen
        alpha_seen = pd.Series(index=batch_df.index, dtype="float64")
        beta_seen  = pd.Series(index=batch_df.index, dtype="float64")

        for mask, lookup in [(is_babylm, babylm_lookup), (~is_babylm, c4_lookup)]:
            sub = batch_df.loc[mask]
            if sub.empty:
                continue
            keys = list(zip(sub["tokens"].astype(int), sub["binom"].str.lower()))
            vals = [lookup.get(k, (float("nan"), float("nan"))) for k in keys]
            alpha_seen.loc[sub.index] = [v[0] for v in vals]
            beta_seen.loc[sub.index]  = [v[1] for v in vals]

        batch_df["alpha_seen"] = alpha_seen
        batch_df["beta_seen"]  = beta_seen

        # Write (header only on first batch)
        batch_df.to_csv(out_fh, index=False, header=not header_written)
        header_written = True

        files_done += len(batch_files)
        pct = 100 * files_done / len(all_files)
        rows_so_far = files_done * 36_975
        print(f"  [{pct:5.1f}%] {files_done:,}/{len(all_files):,} files | "
              f"~{rows_so_far:,} rows written")

print(f"\nDone. Output: {OUT_CSV}")
print(f"Total files processed: {files_done:,}")
