#!/usr/bin/env python3
"""
compute_pile_freq.py

Compute trigram (binomial) frequencies from The Pile for attested binomials
(e.g., "bread and butter" vs "butter and bread").

Used as a corpus-frequency approximation for LLMs trained on web text
(OPT, GPT-Neo, GPT-2), replacing Google Books RelFreq in plot_delta_ll.R.

Output columns match batch_dyn_freq() in abspref_casestudy.Rmd:
  binom, alpha, nonalpha, alpha_seen, nonalpha_seen, freq_prob_c, log_total

Resume-safe: saves a progress checkpoint every SAVE_EVERY documents and
resumes from the last checkpoint on restart.

Usage:
    pip install datasets pandas scipy
    python compute_pile_freq.py

Note: The Pile is ~800 GB; streaming avoids a full download but requires
a stable internet connection for the duration. To limit to a specific subset
(e.g. Pile-CC only), set PILE_SUBSET below.
"""

import json
import logging
import re
import sys
from math import log
from pathlib import Path

import pandas as pd
from scipy.special import expit  # plogis

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Install dependencies: pip install datasets pandas scipy")

# ── Config ────────────────────────────────────────────────────────────────────
# monology/pile-uncopyrighted is The Pile in standard Parquet format (no loading
# script required). It covers all subsets except copyrighted books (Books3,
# BookCorpus2), which is fine for a web-text frequency proxy.
PILE_DATASET = "monology/pile-uncopyrighted"
PILE_SUBSET  = "train"    # this dataset has a single "train" split

SAVE_EVERY   = 50_000    # save progress checkpoint every N documents
REPORT_EVERY = 10_000    # log progress every N documents

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
BASE_DIR    = SCRIPT_DIR.parent.parent
DATA_DIR    = BASE_DIR / "Data"
BINOMS_CSV  = DATA_DIR / "nonce_and_attested_binoms.csv"
OUT_CSV     = SCRIPT_DIR / "pile_corpus_freq.csv"
CKPT_JSON   = SCRIPT_DIR / "pile_freq_checkpoint.json"
LOG_PATH    = SCRIPT_DIR / "compute_pile_freq.log"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Load attested binomials ───────────────────────────────────────────────────
binoms_df = pd.read_csv(BINOMS_CSV)
attested  = binoms_df[binoms_df["Attested"] == 1].copy()

# List of (binom_key, alpha_lower, nonalpha_lower, alpha_pattern, nonalpha_pattern)
# Word-boundary regex ensures we don't match substrings of longer words.
entries = []
for _, row in attested.iterrows():
    key     = str(row["Alpha"])
    alpha   = str(row["Alpha"]).lower()
    nonalpha = str(row["Nonalpha"]).lower()
    entries.append((
        key,
        alpha,
        nonalpha,
        re.compile(r"\b" + re.escape(alpha)    + r"\b"),
        re.compile(r"\b" + re.escape(nonalpha) + r"\b"),
    ))

logger.info(f"Loaded {len(entries)} attested binomials from {BINOMS_CSV}")

# ── Resume from checkpoint ────────────────────────────────────────────────────
counts         = {key: {"alpha_seen": 0, "nonalpha_seen": 0} for key, *_ in entries}
docs_processed = 0

if CKPT_JSON.exists():
    try:
        ckpt = json.loads(CKPT_JSON.read_text(encoding="utf-8"))
        docs_processed = int(ckpt.get("docs_processed", 0))
        saved_counts   = ckpt.get("counts", {})
        for key in counts:
            if key in saved_counts:
                counts[key] = saved_counts[key]
        logger.info(f"Resuming from checkpoint: {docs_processed:,} documents already processed")
    except Exception as e:
        logger.warning(f"Could not read checkpoint ({e}) — starting fresh")
        docs_processed = 0
else:
    logger.info("No checkpoint found — starting fresh")

# ── Helper: save checkpoint ───────────────────────────────────────────────────
def save_checkpoint():
    tmp = CKPT_JSON.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"docs_processed": docs_processed, "counts": counts}, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(CKPT_JSON)

# ── Stream The Pile ───────────────────────────────────────────────────────────
logger.info(f"Loading {PILE_DATASET!r} ({PILE_SUBSET!r}) in streaming mode ...")
try:
    dataset = load_dataset(
        PILE_DATASET,
        split=PILE_SUBSET,
        streaming=True,
    )
except Exception as e:
    sys.exit(
        f"Could not load dataset: {e}\n"
        "Check that 'datasets' is installed and the dataset is accessible.\n"
        "You may need: huggingface-cli login"
    )

# Skip already-processed documents when resuming
if docs_processed > 0:
    logger.info(f"Skipping first {docs_processed:,} documents (already counted) ...")
    dataset = dataset.skip(docs_processed)

logger.info("Counting binomial trigram occurrences ...")
try:
    for doc in dataset:
        text = doc.get("text", "")
        if not text:
            continue

        text_lower = text.lower()

        for key, alpha, nonalpha, alpha_pat, nonalpha_pat in entries:
            # Fast substring pre-filter before regex (cheap string search first)
            if alpha in text_lower:
                counts[key]["alpha_seen"] += len(alpha_pat.findall(text_lower))
            if nonalpha in text_lower:
                counts[key]["nonalpha_seen"] += len(nonalpha_pat.findall(text_lower))

        docs_processed += 1

        if docs_processed % REPORT_EVERY == 0:
            logger.info(f"  {docs_processed:,} documents processed ...")

        if docs_processed % SAVE_EVERY == 0:
            save_checkpoint()
            logger.info(f"  Checkpoint saved at {docs_processed:,} documents")

except KeyboardInterrupt:
    logger.info("Interrupted — saving checkpoint before exit ...")
    save_checkpoint()
    logger.info(f"Checkpoint saved. Re-run the script to resume from {docs_processed:,} documents.")
    sys.exit(0)

# Final checkpoint
save_checkpoint()
logger.info(f"Finished streaming. {docs_processed:,} documents processed total.")

# ── Compute derived columns & write output ────────────────────────────────────
rows = []
for key, alpha, nonalpha, _, _ in entries:
    a = counts[key]["alpha_seen"]
    b = counts[key]["nonalpha_seen"]
    if a > 0 and b > 0:
        log_freq_ratio = log(a / b)
        freq_prob_c    = float(expit(log_freq_ratio)) - 0.5
        log_total      = log(a + b)
    else:
        log_freq_ratio = None
        freq_prob_c    = None
        log_total      = None
    rows.append({
        "binom":         key,
        "alpha":         alpha,
        "nonalpha":      nonalpha,
        "alpha_seen":    a,
        "nonalpha_seen": b,
        "log_freq_ratio": log_freq_ratio,
        "freq_prob_c":   freq_prob_c,
        "log_total":     log_total,
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)

n_obs   = (out_df["alpha_seen"] + out_df["nonalpha_seen"] > 0).sum()
n_both  = ((out_df["alpha_seen"] > 0) & (out_df["nonalpha_seen"] > 0)).sum()
n_total = len(out_df)
logger.info(f"Saved {n_total} rows → {OUT_CSV}")
logger.info(f"  {n_obs}/{n_total} binomials observed at least once")
logger.info(f"  {n_both}/{n_total} binomials observed in both orderings (usable for freq_prob_c)")

# Warn about unusable binomials
unusable = out_df[(out_df["alpha_seen"] == 0) | (out_df["nonalpha_seen"] == 0)]["binom"].tolist()
if unusable:
    logger.warning(f"  {len(unusable)} binomials with zero count in one or both orderings "
                   f"(freq_prob_c = NA): {', '.join(unusable)}")
