"""
prepare_data.py

Aggregates raw checkpoint CSVs and saves smaller CSV files for R to load.
R handles all regression analyses (constraint models, GenPref, Bayesian).
Python handles aggregation, correlations, and accuracy.

Strategy to avoid OOM on 209M-row raw data:
  - Process one checkpoint CSV at a time
  - Aggregate preference across prompts immediately (725 binomals → avg)
  - Merge with human data at the per-file level
  - Never materialise the full 209M-row joined dataset

Outputs (written to ../Data/processed/):
  nonce_agg.csv            full nonce aggregated data  (for R regressions)
  attested_agg.csv         full attested aggregated data (for R regressions)
  nonce_correlations.csv
  attested_correlations.csv
  nonce_accuracy.csv
  attested_accuracy.csv
  final_nonce.csv          final checkpoint only, for brms
  final_attested.csv       final checkpoint only, for brms
  training_nonce.csv       ~20 log-sampled checkpoints, for brms trajectories
  training_attested.csv    ~20 log-sampled checkpoints, for brms trajectories

Requirements: pip install pandas numpy scipy
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# Set working directory to this script's location
os.chdir(Path(__file__).parent)

out_path = Path("../Data/processed")
out_path.mkdir(parents=True, exist_ok=True)

def save_csv(df, name):
    path = out_path / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {name}.csv  ({len(df):,} rows)")


# ── 1. Load and prepare human data (small — done once up front) ───────────────
print("Loading human data...")
human_data = pd.read_csv("../Data/all_human_data.csv")

BSTRESS = "*BStress"

human_data["Attested"]    = (human_data["OverallFreq"] > 0)
human_data["resp_binary"] = (human_data["resp"] == "alpha").astype(float)
human_data["binom"]       = human_data["Alpha"]

# Compute GenPref from constraint weights (matching analysis-script.Rmd formula)
human_data["y_vals"] = (
    0.02191943
    + 0.23925834 * human_data["Form"]
    + 0.24889543 * human_data["Percept"]
    + 0.41836997 * human_data["Culture"]
    + 0.25967334 * human_data["Power"]
    + 0.01867604 * human_data["Intense"]
    + 1.30365980 * human_data["Icon"]
    + 0.08553552 * human_data["Freq"]
    + 0.15241566 * human_data["Len"]
    - 0.19381657 * human_data["Lapse"]
    + 0.36019221 * human_data[BSTRESS]
)
human_data["GenPref"] = 1 / (1 + np.exp(-human_data["y_vals"])) - 0.5

first_cols = [
    "Attested", "Word1", "Word2", "OverallFreq", "RelFreq",
    "Form", "Percept", "Culture", "Power", "Intense", "Icon",
    "Freq", "Len", "Lapse", BSTRESS, "GenPref",
]

avg_human_pref = (
    human_data
    .groupby("binom", sort=False)
    .agg(hum_pref=("resp_binary", "mean"),
         **{col: (col, "first") for col in first_cols})
    .reset_index()
    .rename(columns={BSTRESS: "BStress"})
)
print(f"  avg_human_pref: {len(avg_human_pref):,} binomials")


# ── 2. Process checkpoint CSVs one at a time ──────────────────────────────────
# Each file has ~36,975 rows (725 binomials × 51 prompts).
# We average across prompts immediately, reducing to 725 rows per file,
# then split into nonce / attested and accumulate those small chunks.
# This keeps peak memory at ~one file + the growing aggregated lists.
print("Aggregating checkpoint CSVs (one file at a time)...")
csv_files = sorted(glob.glob("../Data/checkpoint_results/*.csv"))
print(f"  Found {len(csv_files)} files")

KEEP_COLS = ["model", "checkpoint", "step", "tokens", "binom", "preference"]

nonce_chunks    = []
attested_chunks = []

for i, f in enumerate(csv_files):
    if (i + 1) % 500 == 0 or i == 0:
        print(f"  File {i+1}/{len(csv_files)} ...")

    df = pd.read_csv(f, keep_default_na=False, usecols=KEEP_COLS)
    df["step"]       = pd.to_numeric(df["step"],       errors="coerce")
    df["tokens"]     = pd.to_numeric(df["tokens"],     errors="coerce")
    df["preference"] = pd.to_numeric(df["preference"], errors="coerce")

    # Average preference across prompt variants
    agg = (
        df.groupby(["model", "checkpoint", "step", "tokens", "binom"], sort=False)
        ["preference"].mean()
        .reset_index()
    )

    # Attach human data
    agg = agg.merge(avg_human_pref, on="binom", how="left")

    nonce_chunks.append(agg[~agg["Attested"]].copy())
    attested_chunks.append(agg[ agg["Attested"]].copy())

print("  Concatenating chunks...")
nonce    = pd.concat(nonce_chunks,    ignore_index=True)
attested = pd.concat(attested_chunks, ignore_index=True)
print(f"  Nonce: {len(nonce):,} rows | Attested: {len(attested):,} rows")


# ── 3. Save full aggregated data (for R regressions) ──────────────────────────
print("Saving aggregated data for R...")

# Add derived columns that R's analyses need.
# GenPref is already centered at 0 (computed as logistic(y_vals) - 0.5,
# ranging -0.5 to +0.5), so GenPref_centered == GenPref.
for data in [nonce, attested]:
    data["GenPref_centered"] = data["GenPref"]

attested["RelFreq_centered"]   = attested["RelFreq"] - 0.5
attested["OverallFreq_log"]    = np.log(attested["OverallFreq"].astype(float))
# Compute mean/std on unique binomials (not all rows, which repeat each binomial
# once per model×checkpoint and would give a wrong ddof denominator).
log_freq_unique = attested.drop_duplicates("binom")["OverallFreq_log"]
freq_mean = log_freq_unique.mean()
freq_std  = log_freq_unique.std()
attested["OverallFreq_scaled"] = (attested["OverallFreq_log"] - freq_mean) / freq_std

save_csv(nonce,    "nonce_agg")
save_csv(attested, "attested_agg")


# ── 4. Correlations (Pearson r at every checkpoint) ───────────────────────────
print("Computing correlations...")

def compute_correlations(data):
    rows = []
    for keys, g in data.groupby(["model", "checkpoint", "step", "tokens"], sort=False):
        mask = g["preference"].notna() & g["hum_pref"].notna()
        if mask.sum() < 2:
            continue
        r, _ = stats.pearsonr(g.loc[mask, "preference"], g.loc[mask, "hum_pref"])
        rows.append(dict(zip(["model", "checkpoint", "step", "tokens"], keys),
                         correlation=r, n_items=int(mask.sum())))
    return pd.DataFrame(rows)

save_csv(compute_correlations(nonce),    "nonce_correlations")
save_csv(compute_correlations(attested), "attested_correlations")


# ── 5. Accuracy (at every checkpoint) ─────────────────────────────────────────
print("Computing accuracy...")

def compute_accuracy(data):
    data = data.copy()
    mask = data["preference"].notna() & data["hum_pref"].notna()
    data = data[mask]
    data["correct"] = (
        (data["preference"] > 0.5).astype(int) ==
        (data["hum_pref"]   > 0.5).astype(int)
    )
    return (
        data
        .groupby(["model", "checkpoint", "step", "tokens"], sort=False)
        .agg(accuracy=("correct", "mean"), n_items=("correct", "count"))
        .reset_index()
    )

save_csv(compute_accuracy(nonce),    "nonce_accuracy")
save_csv(compute_accuracy(attested), "attested_accuracy")


# ── 6. Final checkpoint subsets (for brms in R) ───────────────────────────────
print("Preparing final checkpoint data...")

def get_final_checkpoint(data):
    return (
        data
        .groupby("model", sort=False, group_keys=False)
        .apply(lambda g: g[g["tokens"] == g["tokens"].max()])
        .reset_index(drop=True)
    )

save_csv(get_final_checkpoint(nonce),    "final_nonce")
save_csv(get_final_checkpoint(attested), "final_attested")


# ── 7. Log-sampled training trajectory subsets (for brms in R) ───────────────
print("Preparing training trajectory data...")

def get_sampled_checkpoints(model_df, n_samples=20):
    ckpt_tokens = (
        model_df.groupby("checkpoint", sort=False)["tokens"]
        .first()
        .sort_values()
    )
    checkpoints = ckpt_tokens.index.tolist()
    n = len(checkpoints)
    if n <= n_samples:
        return checkpoints
    log_idx = np.exp(np.linspace(np.log(1), np.log(n), n_samples))
    indices = np.unique(np.clip(np.round(log_idx).astype(int) - 1, 0, n - 1))
    return [checkpoints[i] for i in indices]

def get_training_subset(data):
    chunks = []
    for model_name, g in data.groupby("model", sort=False):
        ckpts = get_sampled_checkpoints(g)
        chunks.append(g[g["checkpoint"].isin(ckpts)])
    return pd.concat(chunks, ignore_index=True)

save_csv(get_training_subset(nonce),    "training_nonce")
save_csv(get_training_subset(attested), "training_attested")

# ── 8. Raw human binary responses for logistic regression (Analysis 2) ───────
print("Preparing human responses for logistic regression...")
attested_binoms = set(avg_human_pref.loc[avg_human_pref["Attested"], "binom"])
human_att = human_data[human_data["binom"].isin(attested_binoms)].copy()
human_att["resp_binary"] = (human_att["resp"] == "alpha").astype(int)
save_csv(human_att[["binom", "resp_binary"]], "human_responses_attested")

print(f"\nAll done. Files written to {out_path.resolve()}")
print("\nR still needs to fit:")
print("  - Constraint regressions  (from nonce_agg / attested_agg)")
print("  - GenPref regressions     (from nonce_agg / attested_agg)")
print("  - Bayesian brms models    (from final_nonce / final_attested)")
print("  - Bayesian trajectory models (from training_nonce / training_attested)")
