#!/usr/bin/env python3
"""
parse_surprisal_output.py
=========================
Convert Oh & Schuler's word-surprisal output files into item/zone CSVs
and compute corpus perplexity.

Their output format (space-delimited):
  word llmsurp
  If 8.806...
  you 1.494...
  ...

We align these words back to item/zone using all_stories.tok (same word order).

Usage:
  python parse_surprisal_output.py <surprisal_file> <model_id> <family> <params>

Output:
  ns_surprisal/<safe_model_name>.csv   — item, zone, word, surprisal
  ns_surprisal/ns_perplexity.csv       — updated with this model's perplexity
"""

import sys
import math
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
TOK_FILE   = SCRIPT_DIR / "natural_stories" / "all_stories.tok"
OUT_DIR    = SCRIPT_DIR / "ns_surprisal"
PPL_CSV    = OUT_DIR / "ns_perplexity.csv"
OUT_DIR.mkdir(exist_ok=True)

def safe_name(model_id):
    return model_id.replace("/", "_")

def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <surprisal_file> <model_id> <family> <params>")
        sys.exit(1)

    surp_file = Path(sys.argv[1])
    model_id  = sys.argv[2]
    family    = sys.argv[3]
    params    = sys.argv[4]

    # Load corpus word list (ground truth item/zone)
    corpus = pd.read_csv(TOK_FILE, sep="\t",
                         dtype={"word": str, "zone": int, "item": int})
    corpus = corpus.sort_values(["item", "zone"]).reset_index(drop=True)

    # Parse surprisal output (skip header line)
    surp_lines = surp_file.read_text(encoding="utf-8").splitlines()
    assert surp_lines[0].strip() == "word llmsurp", \
        f"Unexpected header: {surp_lines[0]}"

    surp_data = []
    for line in surp_lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        surp_data.append((parts[0], float(parts[1])))

    if len(surp_data) != len(corpus):
        print(f"WARNING: {len(surp_data)} surprisal values vs "
              f"{len(corpus)} corpus words — mismatch!")

    # Zip surprisal values onto corpus item/zone
    n = min(len(surp_data), len(corpus))
    records = []
    for i in range(n):
        row  = corpus.iloc[i]
        word, surp = surp_data[i]
        records.append({
            "item":      int(row["item"]),
            "zone":      int(row["zone"]),
            "word":      row["word"],
            "surprisal": surp,
        })

    result_df = pd.DataFrame(records)

    # Save surprisal CSV atomically
    out_csv = OUT_DIR / f"{safe_name(model_id)}.csv"
    tmp_csv = out_csv.with_suffix(".tmp")
    result_df.to_csv(tmp_csv, index=False)
    tmp_csv.replace(out_csv)
    print(f"Saved surprisal → {out_csv}")

    # Perplexity: surprisal values are in bits → mean * log(2) gives mean NLL in nats
    valid_surp = [r["surprisal"] for r in records if not math.isnan(r["surprisal"])]
    mean_nll_nats = (sum(valid_surp) / len(valid_surp)) * math.log(2)
    perplexity = math.exp(mean_nll_nats)
    print(f"Corpus perplexity: {perplexity:.4f}")

    # Update perplexity CSV
    ppl_df = pd.read_csv(PPL_CSV) if PPL_CSV.exists() else \
             pd.DataFrame(columns=["model", "family", "params", "perplexity"])
    ppl_df = ppl_df[ppl_df["model"] != model_id]
    ppl_df = pd.concat([ppl_df, pd.DataFrame([{
        "model":      model_id,
        "family":     family,
        "params":     params,
        "perplexity": perplexity,
    }])], ignore_index=True)
    ppl_df.to_csv(PPL_CSV, index=False)
    print(f"Perplexity table updated → {PPL_CSV}")

if __name__ == "__main__":
    main()
