#!/usr/bin/env python3
"""
prepare_ns_input.py
===================
Convert Natural Stories all_stories.tok into the .sentitems format
expected by Oh & Schuler's get_llm_surprisal.py.

Format:
  !ARTICLE
  <all words of story 1 as a single space-joined line>
  !ARTICLE
  <all words of story 2 as a single space-joined line>
  ...

Output: natural_stories/ns.sentitems
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR  = Path(__file__).parent
TOK_FILE    = SCRIPT_DIR / "natural_stories" / "all_stories.tok"
OUT_FILE    = SCRIPT_DIR / "natural_stories" / "ns.sentitems"

df = pd.read_csv(TOK_FILE, sep="\t", header=None, names=["tag", "word"], dtype=str)
parsed = df["tag"].str.extract(r"!(\d+)!(\d+)")
df["item"] = parsed[0].astype(int)
df["zone"]  = parsed[1].astype(int)
df = df.sort_values(["item", "zone"])

lines = []
for item_id, group in df.groupby("item"):
    words = group["word"].tolist()
    lines.append("!ARTICLE\n")
    lines.append(" ".join(words) + "\n")

OUT_FILE.write_text("".join(lines), encoding="utf-8")
print(f"Written {df['item'].nunique()} stories → {OUT_FILE}")
