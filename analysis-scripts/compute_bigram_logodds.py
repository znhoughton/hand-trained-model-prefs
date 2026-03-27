"""
compute_bigram_logodds.py
-------------------------
For each attested binomial "W1 and W2", computes a bigram log-odds predictor
that captures the positional tendency of each word in "X and Y" frames:

    bigram_logodds = log(count("W1 and") * count("and W2"))
                   - log(count("W2 and") * count("and W1"))

A value > 0 means W1 tends to appear before "and" more than W2 does, and W2
tends to appear after "and" more than W1 does — i.e. the corpus supports the
alpha (alphabetical) ordering on bigram grounds, independent of the full
trigram "W1 and W2" (which is what RelFreq already captures).

Requires:
  ../Data/babylm_eng_trigrams.csv   — trigram,count (BabyLM training corpus)
  ../Data/binomial_corpus_counts.csv — list of binomials with Word1, Word2

Optional (for C4 models):
  ../Data/c4_eng_trigrams.csv        — same format; generate from raw C4 subset

Output:
  ../Data/processed/bigram_logodds.csv
    binom, bigram_logodds_babylm, bigram_logodds_c4  (c4 column = NaN if file absent)

Usage:
  python compute_bigram_logodds.py
"""

import csv
import math
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "../Data"))
OUT_DIR    = os.path.join(DATA_DIR, "processed")

BINOM_CSV      = os.path.join(DATA_DIR, "binomial_corpus_counts.csv")
BABYLM_TRIGRAM = os.path.join(DATA_DIR, "babylm_eng_trigrams.csv")
C4_TRIGRAM     = os.path.join(DATA_DIR, "c4_eng_trigrams.csv")
OUT_CSV        = os.path.join(OUT_DIR,  "bigram_logodds.csv")

PSEUDOCOUNT = 0.5   # add to each raw count before taking log


def load_words(binom_csv):
    """Return (binomials, word_set) from the corpus-counts file."""
    binomials = []
    words = set()
    with open(binom_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Attested", "0").strip() == "1":
                binom = row["binom"].strip()
                w1    = row["Word1"].strip().lower()
                w2    = row["Word2"].strip().lower()
                binomials.append((binom, w1, w2))
                words.add(w1)
                words.add(w2)
    return binomials, words


def count_bigrams(trigram_csv, target_words):
    """
    Scan a trigram file for rows of the form "W1 and W2, count" and
    accumulate:
        w_and[w]  = total count of trigrams where w appears immediately before "and"
        and_w[w]  = total count of trigrams where w appears immediately after  "and"
    Only tracks words that are in target_words.
    """
    w_and = defaultdict(int)
    and_w = defaultdict(int)
    n_read = 0

    print(f"  Scanning {os.path.basename(trigram_csv)} …")
    with open(trigram_csv, newline="", encoding="utf-8") as f:
        next(f)  # skip header line
        for line in f:
            # Format: "w1 w2 w3,count"  — split on the last comma
            last = line.rfind(",")
            if last == -1:
                continue
            trigram   = line[:last].strip()
            count_str = line[last + 1:].strip()
            try:
                count = int(count_str)
            except ValueError:
                continue

            tokens = trigram.split()
            if len(tokens) != 3:
                continue
            t1, conj, t2 = tokens
            if conj != "and":
                continue

            if t1 in target_words:
                w_and[t1] += count
            if t2 in target_words:
                and_w[t2] += count

            n_read += 1
            if n_read % 1_000_000 == 0:
                print(f"    … {n_read:,} 'X and Y' trigrams processed", flush=True)

    print(f"  Done — {n_read:,} 'X and Y' trigrams found.")
    return w_and, and_w


def bigram_logodds(w1, w2, w_and, and_w):
    """
    log-odds favouring alpha (W1 and W2) over non-alpha (W2 and W1) ordering,
    based on bigram position counts.
    """
    alpha = (w_and[w1] + PSEUDOCOUNT) * (and_w[w2] + PSEUDOCOUNT)
    beta  = (w_and[w2] + PSEUDOCOUNT) * (and_w[w1] + PSEUDOCOUNT)
    return math.log(alpha) - math.log(beta)


def main():
    print("Loading binomials …")
    binomials, words = load_words(BINOM_CSV)
    print(f"  {len(binomials)} attested binomials, {len(words)} unique words")

    # ── BabyLM ────────────────────────────────────────────────────────────────
    babylm_w_and, babylm_and_w = count_bigrams(BABYLM_TRIGRAM, words)

    # ── C4 (optional) ─────────────────────────────────────────────────────────
    if os.path.exists(C4_TRIGRAM):
        c4_w_and, c4_and_w = count_bigrams(C4_TRIGRAM, words)
        has_c4 = True
    else:
        print(f"  {os.path.basename(C4_TRIGRAM)} not found — c4 column will be NA")
        has_c4 = False

    # ── Compute log-odds per binomial ──────────────────────────────────────────
    print("Computing log-odds …")
    rows = []
    for binom, w1, w2 in binomials:
        lo_babylm = bigram_logodds(w1, w2, babylm_w_and, babylm_and_w)
        lo_c4     = bigram_logodds(w1, w2, c4_w_and, c4_and_w) if has_c4 else float("nan")
        rows.append({"binom": binom,
                     "bigram_logodds_babylm": round(lo_babylm, 6),
                     "bigram_logodds_c4":     round(lo_c4, 6) if has_c4 else ""})

    # ── Write output ───────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["binom", "bigram_logodds_babylm",
                                                "bigram_logodds_c4"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
