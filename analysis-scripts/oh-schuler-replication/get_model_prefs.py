#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_model_prefs.py
==================
Score the GPT-2 family models (Oh & Schuler 2023) on our binomial ordering
stimuli and compute per-model validation perplexity on WikiText-2.

For each model:
  - preference  = mean over prompts of [log P(α ordering) − log P(β ordering)]
  - perplexity  = exp(mean NLL) on WikiText-2 test set  ← VALIDATION perplexity

Output: oh_schuler_prefs.csv
  columns: model, model_params, binom, preference
           (perplexity joined per model in oh_schuler_perplexity.csv)

Run from the oh-schuler-replication/ folder:
  python get_model_prefs.py

Requirements:
  pip install torch transformers datasets tqdm
"""

import os
import math
import re
import datetime
import shutil
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def _discover_olmo_checkpoints(model_id, stage_prefix=None):
    """
    Query the HuggingFace Hub for branch/tag revisions of `model_id` and return
    a dict mapping phase label → revision name.

    Naming conventions by OLMo generation (confirmed from HF Hub):
      OLMo-1  : branches  "step{N}-tokens{M}B"
      OLMo-2-1124 : tags  "step{N}-tokens{M}B"  (no stage prefix for stage 1)
      OLMo-2-0425 : tags  "stage1-step{N}-tokens{M}B"  → pass stage_prefix="stage1"
      OLMo-3  : branches  "stage1-step{N}"  (no token count in name, but token count
                can be derived: 4,194,304 tokens/step = 4096 seqs × 1024 tokens)
                → pass stage_prefix="stage1"

    Returns {} if no matching revisions are found or the Hub is unreachable.
    """
    try:
        from huggingface_hub import list_repo_refs
    except ImportError:
        return {}
    try:
        refs = list_repo_refs(model_id)
    except Exception as e:
        print(f"    [warn] list_repo_refs({model_id}) failed: {e}")
        return {}

    all_ref_names = [b.name for b in refs.branches] + [t.name for t in refs.tags]
    step_refs = [n for n in all_ref_names if "step" in n.lower()]
    print(f"    refs with 'step' in name ({len(step_refs)} total): "
          f"{step_refs[:5]}{'...' if len(step_refs) > 5 else ''}")

    ck_list = []      # list of (sort_value, token_B_or_None, rev_name)

    for name in all_ref_names:
        if stage_prefix:
            # Pattern with token count: "stage1-step{N}-tokens{M}B"
            m = re.match(
                rf'^{re.escape(stage_prefix)}-step\d+-tokens(\d+(?:\.\d+)?)B$',
                name, re.IGNORECASE)
            if m:
                tok = float(m.group(1))
                ck_list.append((tok, tok, name))
                continue
            # Pattern without token count: "stage1-step{N}"  (OLMo-3)
            # Token count derived from fixed batch size: 4096 seqs × 1024 tokens = 4,194,304/step
            m = re.match(rf'^{re.escape(stage_prefix)}-step(\d+)$', name, re.IGNORECASE)
            if m:
                step = int(m.group(1))
                tok = step * 4_194_304 / 1e9   # convert to billions
                ck_list.append((tok, tok, name))
        else:
            # No stage prefix: "step{N}-tokens{M}B"
            m = re.match(r'^step\d+-tokens(\d+(?:\.\d+)?)B$', name, re.IGNORECASE)
            if m:
                tok = float(m.group(1))
                ck_list.append((tok, tok, name))

    if not ck_list:
        return {}

    ck_list.sort()
    max_val = ck_list[-1][0]           # tokens_B when available, else step count

    result = {}

    # Absolute token-count anchors: 1B, then 5B, 10B, 15B, ..., 100B
    targets = [1.0] + list(range(5, 105, 5))   # [1, 5, 10, 15, ..., 100]
    for target_b in targets:
        if max_val >= target_b:
            best = min(ck_list, key=lambda x: abs(x[0] - target_b))
            label = f"{int(target_b)}B tokens" if target_b == int(target_b) else f"{target_b}B tokens"
            result[label] = best[2]

    return result

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
BASE_DIR     = SCRIPT_DIR.parent.parent          # hand-trained-model-prefs/
DATA_DIR     = BASE_DIR / "Data"
HUMAN_CSV    = DATA_DIR / "all_human_data.csv"
OUT_PREFS         = SCRIPT_DIR / "oh_schuler_prefs.csv"
OUT_PREFS_BY_PROMPT = SCRIPT_DIR / "oh_schuler_prefs_by_prompt.csv"
OUT_PPL           = SCRIPT_DIR / "oh_schuler_perplexity.csv"
STAGING_DIR  = SCRIPT_DIR / "staging"
STAGING_DIR.mkdir(exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    # GPT-2 family
    "gpt2":                          {"params": "124M",    "family": "GPT-2",   "label": "GPT-2",         "skip": False},
    "gpt2-medium":                   {"params": "355M",    "family": "GPT-2",   "label": "GPT-2 medium",  "skip": False},
    "gpt2-large":                    {"params": "774M",    "family": "GPT-2",   "label": "GPT-2 large",   "skip": False},
    "gpt2-xl":                       {"params": "1542M",   "family": "GPT-2",   "label": "GPT-2 XL",      "skip": False},
    # GPT-Neo family (EleutherAI)
    "EleutherAI/gpt-neo-125m":       {"params": "125M",    "family": "GPT-Neo", "label": "GPT-Neo 125M",  "skip": False},
    "EleutherAI/gpt-neo-1.3B":       {"params": "1300M",   "family": "GPT-Neo", "label": "GPT-Neo 1.3B",  "skip": False},
    "EleutherAI/gpt-neo-2.7B":       {"params": "2700M",   "family": "GPT-Neo", "label": "GPT-Neo 2.7B",  "skip": False},
    # OPT family (Meta)
    "facebook/opt-125m":             {"params": "125M",    "family": "OPT",     "label": "OPT-125M",      "skip": False},
    "facebook/opt-350m":             {"params": "350M",    "family": "OPT",     "label": "OPT-350M",      "skip": False},
    "facebook/opt-1.3b":             {"params": "1300M",   "family": "OPT",     "label": "OPT-1.3B",      "skip": False},
    "facebook/opt-2.7b":             {"params": "2700M",   "family": "OPT",     "label": "OPT-2.7B",      "skip": False},
    "facebook/opt-6.7b":             {"params": "6700M",   "family": "OPT",     "label": "OPT-6.7B",      "skip": False},
    "facebook/opt-13b":              {"params": "13000M",  "family": "OPT",     "label": "OPT-13B",       "skip": False},
    "facebook/opt-30b":              {"params": "30000M",  "family": "OPT",     "label": "OPT-30B",       "skip": False, "multi_gpu": False},
    "facebook/opt-66b":              {"params": "66000M",  "family": "OPT",     "label": "OPT-66B",       "skip": False, "multi_gpu": True},
    "facebook/opt-175b":             {"params": "175000M", "family": "OPT",     "label": "OPT-175B",      "skip": True,  "multi_gpu": True},
    # Pythia family (EleutherAI) — trained on The Pile
    "EleutherAI/pythia-160m":        {"params": "160M",    "family": "Pythia",  "label": "Pythia-160M",   "skip": False, "ppl_only": True},
    "EleutherAI/pythia-410m":        {"params": "410M",    "family": "Pythia",  "label": "Pythia-410M",   "skip": False, "ppl_only": True},
    "EleutherAI/pythia-1.4b":        {"params": "1400M",   "family": "Pythia",  "label": "Pythia-1.4B",   "skip": False, "ppl_only": True},
    "EleutherAI/pythia-2.8b":        {"params": "2800M",   "family": "Pythia",  "label": "Pythia-2.8B",   "skip": False, "ppl_only": True},
    # OLMo Gen 1 (AllenAI, Feb 2024)
    "allenai/OLMo-1B-hf":            {"params": "1000M",   "family": "OLMo-1",  "label": "OLMo-1B",       "skip": False},
    "allenai/OLMo-7B-hf":            {"params": "7000M",   "family": "OLMo-1",  "label": "OLMo-7B",       "skip": False},
    "allenai/OLMo-1B-0724-hf":       {"params": "1000M",   "family": "OLMo-1",  "label": "OLMo-1B-0724",  "skip": False},
    "allenai/OLMo-7B-0424-hf":       {"params": "7000M",   "family": "OLMo-1",  "label": "OLMo-7B-0424",  "skip": False},
    "allenai/OLMo-7B-0724-hf":       {"params": "7000M",   "family": "OLMo-1",  "label": "OLMo-7B-0724",  "skip": False},
    # OLMo Gen 2 (Nov 2024 – Apr 2025)
    "allenai/OLMo-2-0425-1B":        {"params": "1000M",   "family": "OLMo-2",  "label": "OLMo-2-1B",     "skip": False},
    "allenai/OLMo-2-1124-7B":        {"params": "7000M",   "family": "OLMo-2",  "label": "OLMo-2-7B",     "skip": False},
    "allenai/OLMo-2-1124-13B":       {"params": "13000M",  "family": "OLMo-2",  "label": "OLMo-2-13B",    "skip": False},
    "allenai/OLMo-2-0325-32B":       {"params": "32000M",  "family": "OLMo-2",  "label": "OLMo-2-32B",    "skip": False, "multi_gpu": True},
    # OLMo Gen 3 (Oct–Nov 2025)
    "allenai/Olmo-3-1025-7B":        {"params": "7000M",   "family": "OLMo-3",  "label": "OLMo-3-7B",     "skip": False},
    "allenai/Olmo-3-1125-32B":       {"params": "32000M",  "family": "OLMo-3",  "label": "OLMo-3-32B",    "skip": False, "multi_gpu": True},
    # BabyLM models (znhoughton) — OPT architecture, trained on BabyLM corpus
    "znhoughton/opt-babylm-125m-64eps-seed964": {"params": "125M",   "family": "BabyLM", "label": "BabyLM-125M", "skip": False},
    "znhoughton/opt-babylm-350m-64eps-seed964": {"params": "350M",   "family": "BabyLM", "label": "BabyLM-350M", "skip": False},
    "znhoughton/opt-babylm-1.3b-64eps-seed964": {"params": "1300M",  "family": "BabyLM", "label": "BabyLM-1.3B", "skip": False},
    # C4 models (znhoughton) — OPT architecture, trained on C4 subset
    "znhoughton/opt-c4-125m-seed964":            {"params": "125M",   "family": "C4",     "label": "C4-125M",     "skip": False},
    "znhoughton/opt-c4-350m-seed964":            {"params": "350M",   "family": "C4",     "label": "C4-350M",     "skip": False},
    "znhoughton/opt-c4-1.3b-seed964":            {"params": "1300M",  "family": "C4",     "label": "C4-1.3B",     "skip": False},
}

# ── BabyLM / C4 training checkpoints (10 evenly spaced by token count) ────────
# Reads the checkpoint manifest from training_attested.csv and selects 10
# checkpoints per model at evenly spaced token-count fractions (0/9 … 9/9).
_BC_PARAMS = {
    "znhoughton/opt-babylm-125m-64eps-seed964": ("125M",  "BabyLM"),
    "znhoughton/opt-babylm-350m-64eps-seed964": ("350M",  "BabyLM"),
    "znhoughton/opt-babylm-1.3b-64eps-seed964": ("1300M", "BabyLM"),
    "znhoughton/opt-c4-125m-seed964":           ("125M",  "C4"),
    "znhoughton/opt-c4-350m-seed964":           ("350M",  "C4"),
    "znhoughton/opt-c4-1.3b-seed964":           ("1300M", "C4"),
}
_N_CK = 10   # number of evenly spaced checkpoints per model

_TRAIN_CSV = DATA_DIR / "processed" / "training_attested.csv"
if _TRAIN_CSV.exists():
    print("Discovering BabyLM/C4 checkpoints from training_attested.csv ...")
    _ck_meta = {}   # model → sorted list of (tokens, checkpoint_str)
    with open(_TRAIN_CSV, newline="", encoding="utf-8") as _f:
        import csv as _csv
        _reader = _csv.DictReader(_f)
        for _row in _reader:
            _m = _row.get("model", "")
            if _m not in _BC_PARAMS:
                continue
            try:
                _tok = int(_row["tokens"])
                _ck  = _row["checkpoint"]
            except (KeyError, ValueError):
                continue
            _ck_meta.setdefault(_m, set()).add((_tok, _ck))

    for _model_id, _ck_set in _ck_meta.items():
        _params, _fam = _BC_PARAMS[_model_id]
        _sorted = sorted(_ck_set)          # ascending by token count
        _n      = len(_sorted)
        if _n < 2:
            continue
        # Pick indices evenly spaced across [0, n-1], _N_CK points
        _indices = [round(i * (_n - 1) / (_N_CK - 1)) for i in range(_N_CK)]
        _indices = sorted(set(_indices))   # deduplicate in case n < N_CK
        print(f"  {_model_id}: {_n} checkpoints available, selecting {len(_indices)}")
        for _rank, _idx in enumerate(_indices):
            _tok, _ck = _sorted[_idx]
            _tok_b    = _tok / 1e9
            _phase    = f"ck{_rank+1:02d} ({_tok_b:.1f}B)"
            _ck_id    = f"{_model_id}@{_ck}"
            if _ck_id not in MODEL_CONFIGS:
                MODEL_CONFIGS[_ck_id] = {
                    "params":   _params,
                    "family":   f"{_fam} ({_phase})",
                    "label":    f"{_fam} ({_phase})-{_params}",
                    "skip":     False,
                    "revision": _ck,
                }
else:
    print(f"WARNING: {_TRAIN_CSV} not found — BabyLM/C4 checkpoints skipped.")

# ── OLMo training checkpoints (discovered dynamically from HF Hub) ─────────────
# For each OLMo base model, query available step-based revisions and add entries
# for four training phases: early, early-mid, mid, mid-late.
_OLMO_CK_SOURCES = [
    # OLMo-1: 1B checkpoints are on the -hf repo; 7B checkpoints are on the non-hf repo
    ("allenai/OLMo-1B-hf",      "1000M",  "OLMo-1", None),
    ("allenai/OLMo-7B",         "7000M",  "OLMo-1", None),
    # OLMo-2: all variants use "stage1-step{N}-tokens{M}B" for pretraining checkpoints
    ("allenai/OLMo-2-1124-7B",  "7000M",  "OLMo-2", "stage1"),
    ("allenai/OLMo-2-1124-13B", "13000M", "OLMo-2", "stage1"),
    ("allenai/OLMo-2-0425-1B",  "1000M",  "OLMo-2", "stage1"),
    # OLMo-3: "stage1-step{N}" branches (no token count in name)
    ("allenai/Olmo-3-1025-7B",  "7000M",  "OLMo-3", "stage1"),
    ("allenai/Olmo-3-1125-32B", "32000M", "OLMo-3", "stage1"),
]
print("Discovering OLMo training checkpoints from HuggingFace Hub ...")
for _base_id, _params, _fam, _stage in _OLMO_CK_SOURCES:
    _revs = _discover_olmo_checkpoints(_base_id, stage_prefix=_stage)
    if not _revs:
        print(f"  No step-based revisions found for {_base_id} — skipping checkpoints.")
        continue
    print(f"  {_base_id}: found {len(_revs)} checkpoint phases")
    for _phase, _rev in _revs.items():
        _ck_id = f"{_base_id}@{_rev}"
        if _ck_id not in MODEL_CONFIGS:
            MODEL_CONFIGS[_ck_id] = {
                "params":   _params,
                "family":   f"{_fam} ({_phase})",
                "label":    f"{_fam} ({_phase})-{_params}",
                "skip":     False,
                "revision": _rev,
            }

# ── Sentence-frame prompts ────────────────────────────────────────────────────
LIST_OF_PROMPTS = [
    " ",
    "Well, ",
    "So, ",
    "Then ",
    "Possibly ",
    "Or even ",
    "Maybe a ",
    "Perhaps a ",
    "At times ",
    "Suddenly, the ",
    "Honestly just ",
    "Especially the ",
    "For instance ",
    "In some cases ",
    "Every now and then ",
    "Occasionally you'll find ",
    "There can be examples like ",
    "You might notice things like ",
    "People sometimes mention ",
    "Sometimes just ",
    "Nothing specific comes to mind except the ",
    "It reminded me loosely of the ",
    "There was a vague reference to the ",
    "Unexpectedly the ",
    "It's easy to overlook the ",
    "There used to be talk of ",
    "Out in the distance was the ",
    "What puzzled everyone was the ",
    "At some point I overheard ",
    "Without warning came ",
    "A friend once described the ",
    "The scene shifted toward ",
    "Nobody expected to hear about ",
    "Things eventually turned toward ",
    "The conversation eventually returned to ",
    "I only remember a hint of the ",
    "I couldn't quite place the ",
    "It somehow led back to the ",
    "What stood out most was the ",
    "The oddest part involved the ",
    "Later on, people were discussing ",
    "There was this fleeting idea about ",
    "I once heard someone bring up the ",
    "There was a moment involving the ",
    "It all started when we noticed the ",
    "Another example floated around concerning the ",
    "I came across something about the ",
    "A situation arose involving the ",
    "The conversation drifted toward the ",
    "At one point we ended up discussing ",
    "Out of nowhere came a mention of the ",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def atomic_csv_write(df: pd.DataFrame, path: Path) -> None:
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def get_completed_prompts(staging_path: Path, expected_binoms: set) -> set:
    if not staging_path.exists():
        return set()
    try:
        df = pd.read_csv(staging_path)
    except Exception:
        return set()
    completed = set()
    for prompt, grp in df.groupby("prompt"):
        if expected_binoms.issubset(set(grp["binom"])):
            completed.add(prompt)
    return completed


def _first_device(model) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def batch_logprobs(model, tokenizer, texts: list[str],
                   batch_size: int = 64) -> list[float]:
    first_dev = _first_device(model)
    all_lp = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding="longest", return_tensors="pt",
                        pad_to_multiple_of=8)
        input_ids      = enc["input_ids"].to(first_dev)
        attention_mask = enc["attention_mask"].to(first_dev)

        outputs    = model(input_ids, attention_mask=attention_mask)
        logprobs   = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:]

        token_lp = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        token_lp = token_lp * target_mask
        all_lp.extend(token_lp.sum(dim=-1).tolist())
    return all_lp


def score_binomials(model, tokenizer, binoms_df: pd.DataFrame,
                    model_id: str, staging_path: Path) -> pd.DataFrame:
    expected_binoms  = set(binoms_df["Alpha"])
    completed        = get_completed_prompts(staging_path, expected_binoms)
    missing_prompts  = [p for p in LIST_OF_PROMPTS if p not in completed]

    print(f"  Prompts: {len(completed)} already complete, {len(missing_prompts)} to run.")

    for prompt in tqdm(missing_prompts, desc="  prompts", leave=False):
        alpha_texts    = [prompt + str(t) for t in binoms_df["Alpha"]]
        nonalpha_texts = [prompt + str(t) for t in binoms_df["Nonalpha"]]
        lps = batch_logprobs(model, tokenizer, alpha_texts + nonalpha_texts)
        n   = len(alpha_texts)

        prompt_rows = [
            {"model": model_id, "binom": row.Alpha,
             "prompt": prompt, "preference": lps[i] - lps[n + i]}
            for i, row in enumerate(binoms_df.itertuples(index=False))
        ]
        prompt_df = pd.DataFrame(prompt_rows)

        if staging_path.exists():
            try:
                existing = pd.read_csv(staging_path)
                combined = pd.concat([existing, prompt_df], ignore_index=True)
            except Exception:
                combined = prompt_df
        else:
            combined = prompt_df
        atomic_csv_write(combined, staging_path)

    full = pd.read_csv(staging_path)
    return full.groupby(["model", "binom"], as_index=False)["preference"].mean()


@torch.inference_mode()
def compute_validation_perplexity(model, tokenizer, max_tokens: int = 4096) -> float:
    first_dev = _first_device(model)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                           split="test", trust_remote_code=True)
    text = "\n".join(t for t in dataset["text"] if t.strip())

    input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    input_ids = input_ids[:max_tokens]

    stride   = 512
    seq_len  = input_ids.size(0)
    max_ctx  = getattr(model.config, "n_positions",
                       getattr(model.config, "max_position_embeddings", 1024))
    window   = min(max_ctx, 1024)
    nll_sum  = 0.0
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end        = min(begin + window, seq_len)
        target_len = end - max(begin, prev_end)
        if target_len <= 0:
            prev_end = end
            continue

        chunk   = input_ids[begin:end].unsqueeze(0).to(first_dev)
        outputs = model(chunk, labels=chunk)
        chunk_nll = outputs.loss.item() * (end - begin - 1)
        nll_sum  += chunk_nll * target_len / (end - begin - 1)
        n_tokens += target_len
        prev_end  = end

        if end >= seq_len:
            break

    return math.exp(nll_sum / n_tokens)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Using device: {device}\n")

    print("Loading human experiment binomials ...")
    human = pd.read_csv(HUMAN_CSV)
    all_unique  = human[["Alpha", "Nonalpha"]].drop_duplicates(subset="Alpha")
    binoms_df   = (all_unique
                   .dropna(subset=["Alpha", "Nonalpha"])
                   .reset_index(drop=True))
    n_dropped   = len(all_unique) - len(binoms_df)
    dropped_rows = all_unique[all_unique[["Alpha", "Nonalpha"]].isna().any(axis=1)]

    drop_msg = (
        f"Binomials dropped (NaN in Alpha or Nonalpha): {n_dropped} / {len(all_unique)}\n"
        + (("\n".join(f"  {r.Alpha!r}  |  {r.Nonalpha!r}"
                      for r in dropped_rows.itertuples()))
           if n_dropped > 0 else "  (none)")
    )
    print(drop_msg)

    LOG_PATH = SCRIPT_DIR / "get_model_prefs.log"
    with open(LOG_PATH, "a") as log:
        import datetime
        log.write(f"\n{'='*60}\n")
        log.write(f"Run: {datetime.datetime.now().isoformat()}\n")
        log.write(drop_msg + "\n")

    print(f"  {len(binoms_df)} unique binomials retained  (log → {LOG_PATH})\n")

    expected_binoms = set(binoms_df["Alpha"])

    done_prefs         = pd.read_csv(OUT_PREFS)           if OUT_PREFS.exists()           else pd.DataFrame()
    done_prefs_by_prompt = pd.read_csv(OUT_PREFS_BY_PROMPT) if OUT_PREFS_BY_PROMPT.exists() else pd.DataFrame()
    done_ppl   = pd.read_csv(OUT_PPL)   if OUT_PPL.exists()   else pd.DataFrame()
    done_ppl_models = set(done_ppl["model"]) if not done_ppl.empty else set()

    for model_id, cfg in MODEL_CONFIGS.items():
        if cfg.get("skip", False):
            print(f"Skipping {model_id} (skip=True)")
            continue

        # model_id may be "hf/model@revision" for checkpoint entries
        revision = cfg.get("revision", None)
        hf_id    = model_id.split("@")[0] if "@" in model_id else model_id

        model_safe   = model_id.replace("/", "_").replace("@", "_")
        staging_path = STAGING_DIR / f"{model_safe}_staging.csv"

        print("=" * 60)
        print(f"Model: {model_id}  ({cfg['family']}, {cfg['params']})")
        print("=" * 60)

        ppl_only = cfg.get("ppl_only", False)

        completed_prompts = get_completed_prompts(staging_path, expected_binoms)
        prefs_done = ppl_only or (
            not done_prefs.empty and
            model_id in done_prefs.get("model", pd.Series()).values and
            expected_binoms.issubset(
                set(done_prefs[done_prefs["model"] == model_id]["binom"]))
        )
        ppl_done = model_id in done_ppl_models

        if prefs_done and ppl_done:
            print("  ✅ Already fully scored and saved — skipping.\n")
            continue

        if ppl_only:
            print("  [ppl_only] Binomial scoring skipped — computing WikiText-2 perplexity only.")

        all_prompts_staged = ppl_only or len(completed_prompts) == len(LIST_OF_PROMPTS)

        prefs_done_now = prefs_done
        ppl_done_now   = ppl_done

        rev_kwargs = {"revision": revision} if revision else {}

        # Load tokenizer from the base model (no revision) — tokenizers are
        # identical across checkpoints and some revisions fail to load the
        # tokenizer due to transformers version incompatibilities.
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model     = None
        tmp_cache = str(Path(os.environ.get("HF_MODEL_CACHE", Path.home() / ".cache" / "hf_prefs")) / model_safe)
        Path(tmp_cache).mkdir(parents=True, exist_ok=True)
        try:
            if not all_prompts_staged:
                multi_gpu = cfg.get("multi_gpu", False)
                dtype     = torch.float16 if device == "cuda" else torch.float32
                print("  Loading model ...")
                if multi_gpu:
                    model = AutoModelForCausalLM.from_pretrained(
                        hf_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        cache_dir=tmp_cache,
                        max_memory={0: "75GiB", 1: "75GiB"},
                        trust_remote_code=True,
                        **rev_kwargs,
                    ).eval()
                    print(f"  Loaded with device_map='auto' across {torch.cuda.device_count()} GPUs")
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        hf_id, torch_dtype=dtype,
                        low_cpu_mem_usage=True, cache_dir=tmp_cache,
                        trust_remote_code=True,
                        **rev_kwargs,
                    ).to(device).eval()
            else:
                print("  All prompts already in staging — skipping model load.")

            if not prefs_done:
                print("  Scoring binomial preferences ...")
                prefs_df = score_binomials(
                    model, tokenizer, binoms_df, model_id, staging_path
                )
                prefs_df["model_family"] = cfg["family"]
                prefs_df["model_params"] = cfg["params"]
                prefs_df["model_label"]  = cfg["label"]

                combined_prefs = (
                    pd.concat([done_prefs, prefs_df], ignore_index=True)
                    if not done_prefs.empty else prefs_df
                )
                atomic_csv_write(combined_prefs, OUT_PREFS)
                done_prefs = combined_prefs
                print(f"  ✅ Preferences saved → {OUT_PREFS}")

                raw_staging = pd.read_csv(staging_path)
                raw_staging["model_family"] = cfg["family"]
                raw_staging["model_params"] = cfg["params"]
                raw_staging["model_label"]  = cfg["label"]
                combined_by_prompt = (
                    pd.concat([done_prefs_by_prompt, raw_staging], ignore_index=True)
                    if not done_prefs_by_prompt.empty else raw_staging
                )
                atomic_csv_write(combined_by_prompt, OUT_PREFS_BY_PROMPT)
                done_prefs_by_prompt = combined_by_prompt
                print(f"  ✅ Per-prompt preferences saved → {OUT_PREFS_BY_PROMPT}")
                prefs_done_now = True

            if not ppl_done:
                if model is None:
                    multi_gpu = cfg.get("multi_gpu", False)
                    dtype     = torch.float16 if device == "cuda" else torch.float32
                    print("  Loading model for perplexity ...")
                    if multi_gpu:
                        model = AutoModelForCausalLM.from_pretrained(
                            hf_id, dtype=torch.float16, device_map="auto",
                            low_cpu_mem_usage=True, cache_dir=tmp_cache,
                            max_memory={0: "75GiB", 1: "75GiB"},
                            trust_remote_code=True,
                            **rev_kwargs,
                        ).eval()
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            hf_id, torch_dtype=dtype,
                            low_cpu_mem_usage=True, cache_dir=tmp_cache,
                            trust_remote_code=True,
                            **rev_kwargs,
                        ).to(device).eval()

                print("  Computing validation perplexity (WikiText-2 test) ...")
                ppl = compute_validation_perplexity(model, tokenizer)
                print(f"  Perplexity: {ppl:.2f}")

                ppl_row = pd.DataFrame([{
                    "model":        model_id,
                    "model_family": cfg["family"],
                    "model_params": cfg["params"],
                    "model_label":  cfg["label"],
                    "perplexity":   ppl,
                }])
                combined_ppl = (
                    pd.concat([done_ppl, ppl_row], ignore_index=True)
                    if not done_ppl.empty else ppl_row
                )
                atomic_csv_write(combined_ppl, OUT_PPL)
                done_ppl = combined_ppl
                done_ppl_models.add(model_id)
                print(f"  ✅ Perplexity saved → {OUT_PPL}")
                ppl_done_now = True

        finally:
            if model is not None:
                del model
            if device == "cuda":
                torch.cuda.empty_cache()

        if prefs_done_now and ppl_done_now:
            shutil.rmtree(tmp_cache, ignore_errors=True)
            print(f"  🗑️  Model cache deleted.")

        print()

    print("\n✅ All models done.")
    print(f"   Preferences → {OUT_PREFS}")
    print(f"   Perplexity  → {OUT_PPL}")
    print(f"   Staging dir → {STAGING_DIR}  (can be deleted if no longer needed)")


if __name__ == "__main__":
    main()
