#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_ns_surprisal.py
===================
Compute per-word surprisal from HuggingFace causal LMs on the
Natural Stories Corpus (Futrell et al., 2021), following the
methodology of Oh & Schuler (2023, TACL).

For each model:
  - Per-word surprisal  = sum of -log2 P(subword token | context)
                          for all BPE tokens making up the word  (bits)
  - Corpus perplexity   = exp(mean token NLL in nats) on the full corpus

Context-window overflow is handled with a sliding window:
  stride = ctx // 2; the second half of window n serves as the first
  half of window n+1, so every token's surprisal is conditioned on a
  full half-window of context.  (Oh & Schuler 2023, Section 3.2)

Multi-GPU:
  Uses HuggingFace device_map="auto" (via accelerate) to shard large
  models across all visible GPUs.  Works on 1..N GPUs transparently.
  Install accelerate:  pip install accelerate

Outputs (in ns_surprisal/):
  <safe_model_name>.csv   — item, zone, word, surprisal (bits)
  ns_perplexity.csv       — model, family, params, perplexity (nats)

Usage:
  python get_ns_surprisal.py                     # all models
  python get_ns_surprisal.py --models gpt2 gpt2-medium
  python get_ns_surprisal.py --skip-existing     # skip already-computed

Requirements:
  pip install torch transformers accelerate tqdm requests pandas
"""

import argparse
import math
import sys
from pathlib import Path

import requests
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
NS_DIR     = SCRIPT_DIR / "natural_stories"
OUT_DIR    = SCRIPT_DIR / "ns_surprisal"
PPL_CSV    = OUT_DIR / "ns_perplexity.csv"
NS_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

NS_CORPUS_URL = (
    "https://raw.githubusercontent.com/languageMIT/naturalstories"
    "/master/naturalstories_RTS/all_stories.tok"
)
RT_TSV_PATH = NS_DIR / "processed_RTs.tsv"   # fallback word source

# ── Model registry ────────────────────────────────────────────────────────────
# ctx: maximum context window in tokens (used for sliding-window stride)
MODEL_CONFIGS = {
    # GPT-2 family (ctx = 1024)
    "gpt2":                                        {"params": "124M",   "family": "GPT-2",   "ctx": 1024},
    "gpt2-medium":                                 {"params": "355M",   "family": "GPT-2",   "ctx": 1024},
    "gpt2-large":                                  {"params": "774M",   "family": "GPT-2",   "ctx": 1024},
    "gpt2-xl":                                     {"params": "1542M",  "family": "GPT-2",   "ctx": 1024},
    # GPT-Neo family (ctx = 2048)
    "EleutherAI/gpt-neo-125m":                     {"params": "125M",   "family": "GPT-Neo", "ctx": 2048},
    "EleutherAI/gpt-neo-1.3B":                     {"params": "1300M",  "family": "GPT-Neo", "ctx": 2048},
    "EleutherAI/gpt-neo-2.7B":                     {"params": "2700M",  "family": "GPT-Neo", "ctx": 2048},
    # OPT family (ctx = 2048)
    "facebook/opt-125m":                           {"params": "125M",   "family": "OPT",     "ctx": 2048},
    "facebook/opt-350m":                           {"params": "350M",   "family": "OPT",     "ctx": 2048},
    "facebook/opt-1.3b":                           {"params": "1300M",  "family": "OPT",     "ctx": 2048},
    "facebook/opt-2.7b":                           {"params": "2700M",  "family": "OPT",     "ctx": 2048},
    "facebook/opt-6.7b":                           {"params": "6700M",  "family": "OPT",     "ctx": 2048},
    "facebook/opt-13b":                            {"params": "13000M", "family": "OPT",     "ctx": 2048},
    "facebook/opt-30b":                            {"params": "30000M", "family": "OPT",     "ctx": 2048},
    # BabyLM — OPT architecture trained on BabyLM corpus (ctx = 2048)
    "znhoughton/opt-babylm-125m-64eps-seed964":    {"params": "125M",   "family": "BabyLM",  "ctx": 2048},
    "znhoughton/opt-babylm-350m-64eps-seed964":    {"params": "350M",   "family": "BabyLM",  "ctx": 2048},
    "znhoughton/opt-babylm-1.3b-64eps-seed964":    {"params": "1300M",  "family": "BabyLM",  "ctx": 2048},
    # C4 — OPT architecture trained on C4 subset (ctx = 2048)
    "znhoughton/opt-c4-125m-seed964":              {"params": "125M",   "family": "C4",      "ctx": 2048},
    "znhoughton/opt-c4-350m-seed964":              {"params": "350M",   "family": "C4",      "ctx": 2048},
    "znhoughton/opt-c4-1.3b-seed964":              {"params": "1300M",  "family": "C4",      "ctx": 2048},
}

# ── Natural Stories corpus download + parsing ─────────────────────────────────

def download_corpus(path: Path) -> None:
    """Download all_stories.tok from GitHub if not already present.
    Falls back to extracting the word list from processed_RTs.tsv.
    """
    if path.exists():
        return

    print(f"Downloading corpus from {NS_CORPUS_URL} …")
    try:
        r = requests.get(NS_CORPUS_URL, timeout=60)
        if r.status_code == 200:
            path.write_bytes(r.content)
            print(f"  Saved → {path}")
            return
        print(f"  HTTP {r.status_code}")
    except Exception as e:
        print(f"  Request failed: {e}")

    # Fallback: build from the RT file (which contains item/zone/word)
    if RT_TSV_PATH.exists():
        print(f"Falling back to extracting words from {RT_TSV_PATH} …")
        import csv
        rows = []
        with RT_TSV_PATH.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for raw_row in reader:
                row = {k.lower(): v for k, v in raw_row.items()}
                item = row.get("item") or row.get("story")
                zone = row.get("zone") or row.get("word_num") or row.get("position")
                word = row.get("word")
                try:
                    if item and zone and word:
                        rows.append((int(item), int(zone), word.strip()))
                except ValueError:
                    continue
        # Deduplicate and sort
        seen = set()
        unique = []
        for item, zone, word in rows:
            key = (item, zone)
            if key not in seen:
                seen.add(key)
                unique.append((item, zone, word))
        unique.sort()
        # Write TSV matching all_stories.tok format: word\tzone\titem
        lines = ["word\tzone\titem\n"] + [f"{w}\t{z}\t{i}\n" for i, z, w in unique]
        path.write_text("".join(lines), encoding="utf-8")
        print(f"  Extracted {len(unique)} words → {path}")
        return

    raise RuntimeError(
        "Cannot obtain Natural Stories corpus text.\n"
        "Place processed_RTs.tsv in:\n"
        f"  {RT_TSV_PATH}\n"
        "Download from: github.com/languageMIT/naturalstories → naturalstories_RTs/"
    )


def parse_corpus(path: Path) -> pd.DataFrame:
    """Parse all_stories.tok (TSV with header: word  zone  item)."""
    df = pd.read_csv(path, sep="\t", dtype={"word": str, "zone": int, "item": int})
    df = df[["item", "zone", "word"]].sort_values(["item", "zone"]).reset_index(drop=True)
    return df


# ── Surprisal computation ─────────────────────────────────────────────────────

def safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def delete_model_cache(model_id: str) -> None:
    """Delete the HuggingFace disk cache for model_id after inference is done.

    HF stores models at:
      <HF_HOME>/hub/models--<org>--<name>/
    e.g. facebook/opt-125m → models--facebook--opt-125m
    """
    import shutil
    import os

    hf_home = Path(
        os.environ.get("HF_HOME")
        or os.environ.get("TRANSFORMERS_CACHE")
        or Path.home() / ".cache" / "huggingface"
    )
    cache_name = "models--" + model_id.replace("/", "--")
    cache_path = hf_home / "hub" / cache_name

    if cache_path.exists():
        print(f"  Deleting cache: {cache_path}")
        shutil.rmtree(cache_path, ignore_errors=True)
        print("  Cache deleted.")
    else:
        print(f"  Cache not found (already deleted or never downloaded): {cache_path}")


def get_input_device(model) -> torch.device:
    """Return the device that model inputs should be placed on.

    With device_map="auto" the model may be sharded across multiple GPUs.
    Inputs always go to the device of the model's first parameter.
    """
    return next(model.parameters()).device


def compute_surprisal(
    model_id: str,
    corpus_df: pd.DataFrame,
    ctx: int,
) -> tuple[pd.DataFrame, float]:
    """
    Compute per-word surprisal (bits) and corpus perplexity (nats).

    Multi-GPU note
    --------------
    The model is loaded with device_map="auto" which uses HuggingFace
    accelerate to shard layers across all visible CUDA devices.  No
    manual GPU assignment is needed; just ensure CUDA_VISIBLE_DEVICES
    includes all GPUs you want to use.

    Sliding window
    --------------
    For each story we build the full token sequence, then scan it with
    a window of size `ctx` and stride `ctx // 2`.  The first window
    starts at position 0 and scores all tokens it covers.  Subsequent
    windows score only the second half (tokens stride..end), so every
    token is scored with a full half-window of context and no token is
    scored twice.

    Surprisal
    ---------
    Per-word surprisal = sum over subword tokens of -log2 P(token | context),
    i.e. in bits.  The BOS token (position 0) is never predicted, so it
    is excluded from both surprisal records and perplexity.
    """
    use_cuda = torch.cuda.is_available()
    dtype    = torch.float16 if use_cuda else torch.float32

    print(f"\n{'='*60}")
    print(f"Model : {model_id}")
    print(f"dtype : {dtype}  |  device_map : {'auto' if use_cuda else 'cpu'}")
    print(f"{'='*60}")

    print("  Loading tokenizer …")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("  Loading model …")
    load_kwargs = dict(
        torch_dtype    = dtype,
        low_cpu_mem_usage = True,
    )
    if use_cuda:
        load_kwargs["device_map"] = "auto"   # shards across all visible GPUs

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if not use_cuda:
        model = model.to("cpu")
    model.eval()

    # Determine the device that receives model inputs
    input_device = get_input_device(model)
    print(f"  Input device: {input_device}")
    if use_cuda and torch.cuda.device_count() > 1:
        print(f"  Sharding across {torch.cuda.device_count()} GPUs via device_map='auto'")

    records     = []
    total_nll   = 0.0
    total_ntoks = 0
    log2        = math.log(2)   # constant for nats → bits conversion

    for item_id, story_df in tqdm(
        corpus_df.groupby("item"),
        desc  = "  Stories",
        total = corpus_df["item"].nunique(),
    ):
        story_df = story_df.sort_values("zone").reset_index(drop=True)
        words = story_df["word"].tolist()
        zones = story_df["zone"].tolist()

        # ── Build token sequence ──────────────────────────────────────────
        # Prepend BOS when the tokenizer has one (e.g. OPT uses </s> as BOS).
        # GPT-2's bos_token_id == eos_token_id == 50256; prepending it is the
        # standard way to initialise the context for GPT-style models.
        token_ids        = []
        word_token_spans = []   # (start, end) index into token_ids for each word

        if tok.bos_token_id is not None:
            token_ids.append(tok.bos_token_id)

        for word in words:
            # Leading space ensures correct BPE segmentation for mid-sentence words.
            word_toks = tok.encode(" " + word, add_special_tokens=False)
            start = len(token_ids)
            token_ids.extend(word_toks)
            word_token_spans.append((start, len(token_ids)))

        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        n_tokens     = len(token_tensor)

        # ── Sliding-window NLL ────────────────────────────────────────────
        # token_nlls[i] = NLL (nats) for predicting token_ids[i]
        # Position 0 (BOS) is never predicted; leave it as NaN.
        token_nlls = [float("nan")] * n_tokens
        stride     = ctx // 2
        win_start  = 0

        while win_start < n_tokens:
            win_end = min(win_start + ctx, n_tokens)
            chunk   = token_tensor[win_start:win_end].unsqueeze(0).to(input_device)

            with torch.no_grad():
                logits = model(chunk).logits[0]   # (seq_len, vocab)

            # logit[i] predicts token[i+1]; work in float32 for log_softmax stability
            shift_logits = logits[:-1].float()                        # (seq_len-1, vocab)
            shift_labels = chunk[0, 1:].to(shift_logits.device)      # (seq_len-1,)
            log_probs    = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_lp     = log_probs[torch.arange(len(shift_labels)), shift_labels]
            # token_lp[i] = log P(token at win_start+i+1 | context)  in nats

            # Which positions in the global sequence does this window score?
            # First window: score everything from position 1 onward.
            # Later windows: only score the second half (positions win_start+stride .. win_end-1)
            # to avoid re-scoring with less context.
            score_from = 0 if win_start == 0 else stride   # index within token_lp

            for i in range(score_from, len(token_lp)):
                global_pos = win_start + i + 1   # global token index being predicted
                if global_pos < n_tokens and math.isnan(token_nlls[global_pos]):
                    token_nlls[global_pos] = -token_lp[i].item()   # nats

            if win_end == n_tokens:
                break
            win_start += stride

        # ── Accumulate corpus NLL ─────────────────────────────────────────
        # Skip position 0 (BOS — never predicted)
        valid = [x for x in token_nlls[1:] if not math.isnan(x)]
        total_nll   += sum(valid)
        total_ntoks += len(valid)

        # ── Word-level surprisal (bits) ───────────────────────────────────
        for (tok_start, tok_end), zone, word in zip(word_token_spans, zones, words):
            span = token_nlls[tok_start:tok_end]
            if any(math.isnan(x) for x in span):
                word_surp = float("nan")
            else:
                word_surp = sum(span) / log2   # nats → bits
            records.append({"item": item_id, "zone": zone,
                            "word": word, "surprisal": word_surp})

    surprisal_df = pd.DataFrame(records)
    perplexity   = math.exp(total_nll / total_ntoks) if total_ntoks > 0 else float("nan")
    print(f"  Corpus perplexity (nats): {perplexity:.4f}")

    # Free GPU memory across all devices
    del model
    if use_cuda:
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

    return surprisal_df, perplexity


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model IDs to run (default: all). E.g. --models gpt2 gpt2-medium"
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Re-run models even if their output CSV already exists. "
             "Default behaviour is to skip completed models so a crashed "
             "run can be safely restarted without redoing work."
    )
    parser.add_argument(
        "--keep-cache", action="store_true",
        help="Do NOT delete the HuggingFace model cache after each model. "
             "By default the cache is deleted to free disk space."
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference (overrides CUDA detection)."
    )
    args = parser.parse_args()

    # Optionally force CPU
    if args.cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Report GPU environment
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        print(f"CUDA available — {n_gpus} GPU(s):")
        for i, name in enumerate(gpu_names):
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  [{i}] {name}  ({mem_gb:.0f} GB)")
        print("  device_map='auto' will shard large models across all GPUs.")
    else:
        print("No CUDA detected — running on CPU (will be slow).")

    # Download / parse corpus
    corpus_path = NS_DIR / "all_stories.tok"
    download_corpus(corpus_path)
    print("Parsing corpus …")
    corpus_df = parse_corpus(corpus_path)
    print(f"  {len(corpus_df)} words across {corpus_df['item'].nunique()} stories")

    # Determine which models to run
    if args.models:
        unknown = [m for m in args.models if m not in MODEL_CONFIGS]
        if unknown:
            print(f"WARNING: unknown model IDs (skipping): {unknown}", file=sys.stderr)
        run_models = [m for m in args.models if m in MODEL_CONFIGS]
    else:
        run_models = list(MODEL_CONFIGS.keys())

    # Load or initialise perplexity CSV
    ppl_df = pd.read_csv(PPL_CSV) if PPL_CSV.exists() else \
             pd.DataFrame(columns=["model", "family", "params", "perplexity"])

    for model_id in run_models:
        out_csv = OUT_DIR / f"{safe_model_name(model_id)}.csv"

        # Skip already-completed models unless --recompute is set.
        # This makes the script safe to restart after a crash.
        if not args.recompute and out_csv.exists():
            print(f"\nSkipping {model_id} (output exists — use --recompute to redo)")
            continue

        cfg = MODEL_CONFIGS[model_id]
        try:
            surp_df, ppl = compute_surprisal(model_id, corpus_df, ctx=cfg["ctx"])
        except Exception as e:
            print(f"\nERROR on {model_id}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            # Do not delete cache on failure — may help with debugging
            continue

        # Save outputs atomically: write to a temp file then rename,
        # so a crash during write doesn't leave a corrupt CSV that would
        # be skipped on the next restart.
        tmp_csv = out_csv.with_suffix(".tmp")
        surp_df.to_csv(tmp_csv, index=False)
        tmp_csv.replace(out_csv)
        print(f"  Saved surprisal → {out_csv}")

        new_row = pd.DataFrame([{
            "model":      model_id,
            "family":     cfg["family"],
            "params":     cfg["params"],
            "perplexity": ppl,
        }])
        ppl_df = ppl_df[ppl_df["model"] != model_id]
        ppl_df = pd.concat([ppl_df, new_row], ignore_index=True)
        ppl_df.to_csv(PPL_CSV, index=False)
        print(f"  Perplexity table updated → {PPL_CSV}")

        # Delete HF disk cache to free space, unless --keep-cache is set
        if not args.keep_cache:
            delete_model_cache(model_id)

    print(f"\nAll done.")
    print(f"  Surprisal CSVs : {OUT_DIR}")
    print(f"  Perplexity CSV : {PPL_CSV}")
    print(f"\nNext: source reading_time_delta_ll.R (requires {RT_TSV_PATH})")


if __name__ == "__main__":
    main()
