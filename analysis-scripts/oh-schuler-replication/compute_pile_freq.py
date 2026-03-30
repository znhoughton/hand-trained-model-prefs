#!/usr/bin/env python3
"""
compute_pile_freq.py

Compute trigram (binomial) frequencies from The Pile for attested binomials
(e.g., "bread and butter" vs "butter and bread").

Optimised for multi-core machines:
  - Aho-Corasick automaton: all ~1,200 patterns found in a single O(n) text
    scan per document, instead of ~600 separate regex calls.
  - Dataset sharding: splits the corpus into N_WORKERS shards; each worker
    process streams and counts its own shard independently.
  - Resume-safe: completed shards are skipped on restart.

Usage:
    pip install datasets pyahocorasick pandas scipy
    python compute_pile_freq.py
"""

import json
import logging
import multiprocessing as mp
import sys
from math import log
from pathlib import Path

import pandas as pd
from scipy.special import expit  # plogis

try:
    import ahocorasick
except ImportError:
    sys.exit("Install: pip install pyahocorasick")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("Install: pip install tqdm")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Install: pip install datasets pandas scipy")

# ── Config ────────────────────────────────────────────────────────────────────
PILE_DATASET = "monology/pile-uncopyrighted"
PILE_SPLIT   = "train"
N_WORKERS    = 32      # set to number of available CPU cores

# Approximate total docs in monology/pile-uncopyrighted (from dataset card).
# Used only for tqdm ETA — does not affect correctness.
DATASET_TOTAL_DOCS = 210_607_728

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
BASE_DIR    = SCRIPT_DIR.parent.parent
DATA_DIR    = BASE_DIR / "Data"
BINOMS_CSV  = DATA_DIR / "nonce_and_attested_binoms.csv"
OUT_CSV     = SCRIPT_DIR / "pile_corpus_freq.csv"
PARTIAL_DIR = SCRIPT_DIR / "pile_freq_partials"
LOG_PATH    = SCRIPT_DIR / "compute_pile_freq.log"

# ── Logging (main process only) ───────────────────────────────────────────────
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


# ── Build Aho-Corasick automaton ──────────────────────────────────────────────
def build_automaton(entries):
    """
    Build an Aho-Corasick automaton over all alpha/nonalpha phrase forms.

    entries: list of (binom_key, alpha_lower, nonalpha_lower)

    Returns (automaton, phrase_to_slots) where phrase_to_slots maps each
    lowercase phrase to a list of (entry_idx, "alpha"|"nonalpha") so that a
    single phrase that appears as both the alpha of one binom and the nonalpha
    of another is counted correctly for both.
    """
    phrase_to_slots: dict[str, list] = {}
    for idx, (_, alpha, nonalpha) in enumerate(entries):
        phrase_to_slots.setdefault(alpha,    []).append((idx, "alpha"))
        phrase_to_slots.setdefault(nonalpha, []).append((idx, "nonalpha"))

    A = ahocorasick.Automaton()
    for phrase, slots in phrase_to_slots.items():
        A.add_word(phrase, slots)
    A.make_automaton()
    return A, phrase_to_slots


# ── Worker function (runs in each subprocess) ─────────────────────────────────
def process_shard(args):
    shard_idx, n_shards, entries, partial_dir = args
    partial_path = Path(partial_dir) / f"shard_{shard_idx:04d}.json"

    if partial_path.exists():
        print(f"[shard {shard_idx:02d}] already complete — skipping", flush=True)
        return

    automaton, _ = build_automaton(entries)
    n = len(entries)
    alpha_counts    = [0] * n
    nonalpha_counts = [0] * n

    try:
        dataset = load_dataset(PILE_DATASET, split=PILE_SPLIT, streaming=True)
        shard   = dataset.shard(num_shards=n_shards, index=shard_idx)
        est_total = DATASET_TOTAL_DOCS // n_shards

        docs = 0
        with tqdm(
            total=est_total,
            desc=f"shard {shard_idx:02d}",
            unit="doc",
            unit_scale=True,
            position=shard_idx % 8,   # stagger bars so ≤8 show at once
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            for doc in shard:
                text = doc.get("text", "")
                if not text:
                    continue
                text_lower = text.lower()
                for _, slots in automaton.iter(text_lower):
                    for idx, form_type in slots:
                        if form_type == "alpha":
                            alpha_counts[idx] += 1
                        else:
                            nonalpha_counts[idx] += 1
                docs += 1
                pbar.update(1)

    except KeyboardInterrupt:
        print(f"[shard {shard_idx:02d}] interrupted at {docs:,} docs — partial not saved",
              flush=True)
        return

    result = {
        "shard_idx":       shard_idx,
        "docs_processed":  docs,
        "alpha_counts":    alpha_counts,
        "nonalpha_counts": nonalpha_counts,
    }
    tmp = partial_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    tmp.replace(partial_path)
    print(f"[shard {shard_idx:02d}] done  ({docs:,} docs) → {partial_path.name}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logger.info(f"Loading binomials from {BINOMS_CSV}")
    binoms_df = pd.read_csv(BINOMS_CSV)
    attested  = binoms_df[binoms_df["Attested"] == 1]
    entries   = [
        (str(r["Alpha"]), str(r["Alpha"]).lower(), str(r["Nonalpha"]).lower())
        for _, r in attested.iterrows()
    ]
    logger.info(f"  {len(entries)} attested binomials ({len(entries) * 2} patterns in automaton)")

    PARTIAL_DIR.mkdir(exist_ok=True)

    # Determine which shards still need processing
    done_shards = {
        int(p.stem.split("_")[1])
        for p in PARTIAL_DIR.glob("shard_*.json")
    }
    todo_shards = [i for i in range(N_WORKERS) if i not in done_shards]

    if not todo_shards:
        logger.info("All shards already complete — proceeding to merge.")
    else:
        logger.info(
            f"Processing {len(todo_shards)} shards "
            f"({len(done_shards)} already done) with {len(todo_shards)} workers ..."
        )
        worker_args = [
            (i, N_WORKERS, entries, str(PARTIAL_DIR))
            for i in todo_shards
        ]
        with mp.Pool(processes=len(todo_shards)) as pool:
            with tqdm(total=N_WORKERS, initial=len(done_shards),
                      desc="shards complete", unit="shard",
                      position=8, leave=True) as overall:
                for _ in pool.imap_unordered(process_shard, worker_args):
                    overall.update(1)

    # ── Merge shard partials ───────────────────────────────────────────────────
    logger.info("Merging shard results ...")
    alpha_totals    = [0] * len(entries)
    nonalpha_totals = [0] * len(entries)
    total_docs = 0

    for i in range(N_WORKERS):
        pfile = PARTIAL_DIR / f"shard_{i:04d}.json"
        if not pfile.exists():
            logger.error(f"Missing shard file: {pfile} — cannot merge. Re-run script.")
            sys.exit(1)
        data = json.loads(pfile.read_text(encoding="utf-8"))
        for j in range(len(entries)):
            alpha_totals[j]    += data["alpha_counts"][j]
            nonalpha_totals[j] += data["nonalpha_counts"][j]
        total_docs += data["docs_processed"]

    logger.info(f"  Total documents processed across all shards: {total_docs:,}")

    # ── Compute derived columns & write output ─────────────────────────────────
    rows = []
    for j, (key, alpha, nonalpha) in enumerate(entries):
        a = alpha_totals[j]
        b = nonalpha_totals[j]
        if a > 0 and b > 0:
            log_freq_ratio = log(a / b)
            freq_prob_c    = float(expit(log_freq_ratio)) - 0.5
            log_total      = log(a + b)
        else:
            log_freq_ratio = None
            freq_prob_c    = None
            log_total      = None
        rows.append({
            "binom":          key,
            "alpha":          alpha,
            "nonalpha":       nonalpha,
            "alpha_seen":     a,
            "nonalpha_seen":  b,
            "log_freq_ratio": log_freq_ratio,
            "freq_prob_c":    freq_prob_c,
            "log_total":      log_total,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    n_obs   = (out_df["alpha_seen"] + out_df["nonalpha_seen"] > 0).sum()
    n_both  = ((out_df["alpha_seen"] > 0) & (out_df["nonalpha_seen"] > 0)).sum()
    n_total = len(out_df)
    logger.info(f"Saved {n_total} rows → {OUT_CSV}")
    logger.info(f"  {n_obs}/{n_total} binomials observed at least once")
    logger.info(f"  {n_both}/{n_total} binomials observed in both orderings (freq_prob_c defined)")

    unusable = out_df[out_df["freq_prob_c"].isna()]["binom"].tolist()
    if unusable:
        logger.warning(
            f"  {len(unusable)} binomials with zero count in one or both orderings "
            f"(freq_prob_c = NA): {', '.join(unusable)}"
        )


if __name__ == "__main__":
    main()
