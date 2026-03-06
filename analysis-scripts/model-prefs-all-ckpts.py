#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-GPU checkpoint scoring (2x H100):
- Shards checkpoints across GPUs (process-per-GPU)
- Scores AandB + BandA in one call per prompt
- Uses torch.inference_mode, optional torch.compile once per worker
- OOM-safe adaptive batch sizing
- Resume-safe: skips complete prompts, re-runs incomplete prompts, atomic CSV writes
"""

import os
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import tempfile
import shutil

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# CONFIG
# =========================
_BINOMS_DF = None
OUT_DIR = "../Data/checkpoint_results"
BINOMS_CSV = "../Data/nonce_and_attested_binoms.csv"

# Enable compile on Linux (Inductor). If this ever misbehaves, set to False.
ENABLE_COMPILE = True
COMPILE_MODE = "reduce-overhead"  # or "max-autotune" (often slower to compile)



# If you know your machine is stable, keep True.
# If you see occasional hangs / compile weirdness, set False.
USE_TORCH_COMPILE = ENABLE_COMPILE and (os.name != "nt")

# Prompts
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
    "Occasionally you’ll find ",
    "There can be examples like ",
    "You might notice things like ",
    "People sometimes mention ",
    "Sometimes just ",
    "Nothing specific comes to mind except the ",
    "It reminded me loosely of the ",
    "There was a vague reference to the ",
    "Unexpectedly the ",
    "It’s easy to overlook the ",
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
    "I couldn’t quite place the ",
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


MODEL_CONFIGS = {
    # ── OPT / BabyLM ──────────────────────────────────────────────────────────
    # checkpoint_source="step_tags"  →  HF tags named "step-{N}"
    # log_sample=False               →  use every checkpoint (already sparse)
    "znhoughton/opt-babylm-125m-64eps-seed964": {
        "tokens_per_step": 819_200,   # 1024 × 400 × 1 × 2
        "tokenizer": "znhoughton/opt-babylm-125m-64eps-seed964",
        "checkpoint_source": "step_tags",
        "log_sample": False,
    },
    "znhoughton/opt-babylm-350m-64eps-seed964": {
        "tokens_per_step": 1_228_800,  # 1024 × 300 × 2 × 2
        "tokenizer": "znhoughton/opt-babylm-350m-64eps-seed964",
        "checkpoint_source": "step_tags",
        "log_sample": False,
    },
    "znhoughton/opt-babylm-1.3b-64eps-seed964": {
        "tokens_per_step": 819_200,  # 1024 × 100 × 4 × 2
        "tokenizer": "znhoughton/opt-babylm-1.3b-64eps-seed964",
        "checkpoint_source": "step_tags",
        "log_sample": False,
    },
    # ── OPT / C4 ──────────────────────────────────────────────────────────────
    "znhoughton/opt-c4-125m-seed964": {
        "tokens_per_step": 819_200,    # 1024 × 400 × 1 × 2
        "tokenizer": "znhoughton/opt-c4-125m-seed964",
        "checkpoint_source": "step_tags",
        "log_sample": False,
    },
    "znhoughton/opt-c4-350m-seed964": {
        "tokens_per_step": 1_228_800,  # 1024 × 300 × 2 × 2
        "tokenizer": "znhoughton/opt-c4-350m-seed964",
        "checkpoint_source": "step_tags",
        "log_sample": False,
    },
    "znhoughton/opt-c4-1.3b-seed964": {
        "tokens_per_step": 819_200,    # 1024 × 100 × 4 × 2
        "tokenizer": "znhoughton/opt-c4-1.3b-seed964",
        "checkpoint_source": "step_tags",
        "log_sample": False,
    },
    # ── Pythia (EleutherAI) ────────────────────────────────────────────────────
    # checkpoint_source="step_branches"  →  HF git branches named "step{N}"
    # tokens_per_step = 2048 seq_len × 1024 batch = 2,097,152 (all Pythia sizes)
    # ~154 checkpoints total; log_sample=True selects 20
    "EleutherAI/pythia-160m": {
        "tokens_per_step": 2_097_152,
        "tokenizer": "EleutherAI/pythia-160m",
        "checkpoint_source": "step_branches",
        "log_sample": True,
    },
    "EleutherAI/pythia-410m": {
        "tokens_per_step": 2_097_152,
        "tokenizer": "EleutherAI/pythia-410m",
        "checkpoint_source": "step_branches",
        "log_sample": True,
    },
    "EleutherAI/pythia-1.4b": {
        "tokens_per_step": 2_097_152,
        "tokenizer": "EleutherAI/pythia-1.4b",
        "checkpoint_source": "step_branches",
        "log_sample": True,
    },
    "EleutherAI/pythia-2.8b": {
        "tokens_per_step": 2_097_152,
        "tokenizer": "EleutherAI/pythia-2.8b",
        "checkpoint_source": "step_branches",
        "log_sample": True,
    },
    # ── OLMo 3 (Allen AI) ─────────────────────────────────────────────────────
    # Checkpoints are git branches named "stage1-step{N}" (stage1 = pretraining).
    # tokens_per_step = 8192 seq_len × 1024 global batch = 8,388,608
    "allenai/Olmo-3-1025-7B": {
        "tokens_per_step": 8_388_608,  # 8192 × 1024
        "tokenizer": "allenai/Olmo-3-1025-7B",
        "checkpoint_source": "stage1_branches",
        "log_sample": True,
    },
    "allenai/Olmo-3-1125-32B": {
        "tokens_per_step": 8_388_608,  # 8192 × 1024
        "tokenizer": "allenai/Olmo-3-1125-32B",
        "checkpoint_source": "stage1_branches",
        "log_sample": True,
    },
}


# =========================
# HELPERS
# =========================

def detect_num_gpus() -> int:
    """
    Respect CUDA_VISIBLE_DEVICES if set.
    Returns the number of GPUs actually visible to this process.
    """
    if not torch.cuda.is_available():
        return 0

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is None or visible.strip() == "":
        return torch.cuda.device_count()

    # CUDA_VISIBLE_DEVICES="0,1,3" → 3 GPUs
    return len([v for v in visible.split(",") if v.strip() != ""])


def verify_repo_is_tagged(repo_id: str, expected_step_multiple: int, min_checkpoints: int = 5) -> Dict[str, Any]:
    api = HfApi()
    refs = api.list_repo_refs(repo_id)

    tag_steps = sorted(
        int(tag.name.split("-")[1])
        for tag in refs.tags
        if tag.name.startswith("step-")
    )

    if not tag_steps:
        raise RuntimeError(f"❌ {repo_id} has no step-* tags")

    if len(tag_steps) < min_checkpoints:
        raise RuntimeError(f"❌ {repo_id} only has {len(tag_steps)} tags")

    misaligned = [s for s in tag_steps if s % expected_step_multiple != 0]
    if misaligned:
        print(
            f"⚠️  {repo_id} has non-multiple steps (likely final save): "
            f"{misaligned[:5]}{'...' if len(misaligned) > 5 else ''}"
        )

    deltas = sorted(set(b - a for a, b in zip(tag_steps, tag_steps[1:])))

    print(
        f"✅ {repo_id} tagging OK:\n"
        f"   • checkpoints: {len(tag_steps)}\n"
        f"   • steps: {tag_steps[0]} → {tag_steps[-1]}\n"
        f"   • expected base step: {expected_step_multiple}\n"
        f"   • observed deltas: {deltas}"
    )

    return {"steps": tag_steps, "deltas": deltas}


def get_model_checkpoints(repo_id: str, tokens_per_step: int) -> List[Dict[str, Any]]:
    api = HfApi()
    refs = api.list_repo_refs(repo_id)

    checkpoints = []
    for tag in refs.tags:
        if not tag.name.startswith("step-"):
            continue
        step = int(tag.name.split("-")[1])
        checkpoints.append({
            "checkpoint": tag.name,
            "tag": tag.name,
            "step": step,
            "tokens": step * tokens_per_step,
        })

    checkpoints.sort(key=lambda x: x["step"])

    if checkpoints:
        print(
            f"📦 {repo_id}: found {len(checkpoints)} checkpoints "
            f"(steps {checkpoints[0]['step']} → {checkpoints[-1]['step']})"
        )
    else:
        print(f"⚠️ No checkpoints found for {repo_id}")

    return checkpoints

def get_checkpoints_from_stage1_branches(repo_id: str, tokens_per_step: int) -> List[Dict[str, Any]]:
    """
    For OLMo3-style repos where stage-1 pretraining checkpoints are git branches
    named "stage1-step{N}". Only stage1 branches are returned.
    """
    PREFIX = "stage1-step"
    api = HfApi()
    refs = api.list_repo_refs(repo_id)

    checkpoints = []
    for branch in refs.branches:
        name = branch.name
        if not name.startswith(PREFIX):
            continue
        try:
            step = int(name[len(PREFIX):])
        except ValueError:
            continue
        checkpoints.append({
            "checkpoint": name,
            "tag": name,
            "step": step,
            "tokens": step * tokens_per_step,
        })

    checkpoints.sort(key=lambda x: x["step"])

    if checkpoints:
        print(
            f"📦 {repo_id}: found {len(checkpoints)} stage1 branch checkpoints "
            f"(steps {checkpoints[0]['step']} → {checkpoints[-1]['step']})"
        )
    else:
        print(f"⚠️ No stage1 branch checkpoints found for {repo_id}")

    return checkpoints


def get_checkpoints_from_step_branches(repo_id: str, tokens_per_step: int) -> List[Dict[str, Any]]:
    """
    For Pythia-style repos where checkpoints are git branches named stepN (no hyphen).
    """
    api = HfApi()
    refs = api.list_repo_refs(repo_id)

    checkpoints = []
    for branch in refs.branches:
        name = branch.name
        if not name.startswith("step"):
            continue
        try:
            step = int(name[4:])  # strip leading "step"
        except ValueError:
            continue
        checkpoints.append({
            "checkpoint": name,
            "tag": name,
            "step": step,
            "tokens": step * tokens_per_step,
        })

    checkpoints.sort(key=lambda x: x["step"])

    if checkpoints:
        print(
            f"📦 {repo_id}: found {len(checkpoints)} branch checkpoints "
            f"(steps {checkpoints[0]['step']} → {checkpoints[-1]['step']})"
        )
    else:
        print(f"⚠️ No branch checkpoints found for {repo_id}")

    return checkpoints


def log_sample_checkpoints(checkpoints: List[Dict[str, Any]], n: int = 20) -> List[Dict[str, Any]]:
    """
    Log-uniformly sample n checkpoints by index.
    Matches the R formula: unique(round(exp(linspace(log(1), log(total), n)))) - 1
    """
    total = len(checkpoints)
    if total <= n:
        return checkpoints

    indices = sorted(set(
        min(int(round(np.exp(x))) - 1, total - 1)
        for x in np.linspace(np.log(1), np.log(total), n)
    ))
    sampled = [checkpoints[i] for i in indices]
    print(f"  📊 Log-sampled {len(sampled)}/{total} checkpoints")
    return sampled


def check_prompts_in_file(filepath: str, expected_prompts: List[str]) -> Dict[str, Any]:
    """
    Determine which (prompt, binom) pairs are missing.
    Never deletes existing data.
    """
    binoms_df = pd.read_csv(BINOMS_CSV)
    expected_binoms = set(f"{r.Word1} and {r.Word2}" for r in binoms_df.itertuples())

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"    ⚠️ Error reading {filepath}: {e}")
        return {
            "missing_pairs": None,  # signal: rerun everything
        }

    missing_pairs = []

    for prompt in expected_prompts:
        prompt_df = df[df["prompt"] == prompt]
        seen_binoms = set(prompt_df["binom"].unique())
        missing = expected_binoms - seen_binoms

        for binom in missing:
            missing_pairs.append((prompt, binom))

    if missing_pairs:
        print(f"    ⚠️ Missing {len(missing_pairs)} (prompt, binom) pairs")
    else:
        print("    ✅ All prompts complete")

    return {
        "missing_pairs": missing_pairs,
    }


@torch.inference_mode()
def to_tokens_and_logprobs(
    model,
    tokenizer,
    input_texts: List[str],
    device: str,
    batch_size: int = 256,
    desc: Optional[str] = None,
    leave: bool = False,
) -> List[float]:
    """
    Returns list of total sequence logprobs for each text in input_texts.

    OOM-safe: halves batch size and retries.
    """
    all_logprobs: List[float] = []
    n = len(input_texts)
    total_batches = (n + batch_size - 1) // batch_size

    i = 0
    pbar = tqdm(total=total_batches, desc=desc, leave=leave)
    while i < n:
        batch_texts = input_texts[i:i + batch_size]
        try:
            enc = tokenizer(
                batch_texts,
                padding="longest",
                return_tensors="pt",
                pad_to_multiple_of=8,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1)

            logprobs = logprobs[:, :-1, :]
            target_ids = input_ids[:, 1:]
            target_mask = attention_mask[:, 1:]

            token_logprobs = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            token_logprobs = token_logprobs * target_mask

            all_logprobs.extend(token_logprobs.sum(dim=-1).tolist())

            i += batch_size
            pbar.update(1)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if batch_size == 1:
                    raise RuntimeError("OOM even at batch_size=1") from e
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                # recompute expected batches isn't super important; keep pbar roughly correct
                pbar.set_description(f"{desc} (OOM -> bs={batch_size})")
            else:
                raise

    pbar.close()
    return all_logprobs


def pick_start_batch_size(model_name: str) -> int:
    # H100 has tons of memory; start aggressively (OOM backoff will handle edge cases).
    name = model_name.lower()
    if "32b" in name:
        return 64
    if "7b" in name or "6.9b" in name:
        return 128
    if "2.8b" in name:
        return 256
    if "1.3b" in name or "1.4b" in name or "1b" in name:
        return 1024
    if "350m" in name or "410m" in name:
        return 2048
    return 4096  # 125m, 160m, 70m, etc.


def get_model_prefs(prompt: str, model_name: str, checkpoint_info: Dict[str, Any], tokenizer, model, device: str) -> pd.DataFrame:
    global _BINOMS_DF
    if _BINOMS_DF is None:
        _BINOMS_DF = pd.read_csv(BINOMS_CSV)
    df = _BINOMS_DF

    df["AandB"] = prompt + df["Word1"] + " and " + df["Word2"]
    df["BandA"] = prompt + df["Word2"] + " and " + df["Word1"]

    alpha_texts = df["AandB"].tolist()
    nonalpha_texts = df["BandA"].tolist()

    # BIG WIN: score both in one pass
    combined_texts = alpha_texts + nonalpha_texts
    start_bs = pick_start_batch_size(model_name)

    scores = to_tokens_and_logprobs(
        model=model,
        tokenizer=tokenizer,
        input_texts=combined_texts,
        device=device,
        batch_size=start_bs,
        desc="      scoring (AandB+BandA)",
        leave=False,
    )

    n = len(alpha_texts)
    alpha_scores = scores[:n]
    nonalpha_scores = scores[n:]

    rows = []
    for i, row in enumerate(df.itertuples()):
        rows.append({
            "model": model_name,
            "checkpoint": checkpoint_info["checkpoint"],
            "revision": checkpoint_info["tag"],
            "step": checkpoint_info["step"],
            "tokens": checkpoint_info["tokens"],
            "prompt": prompt,
            "binom": f"{row.Word1} and {row.Word2}",
            "alpha_logprob": alpha_scores[i],
            "nonalpha_logprob": nonalpha_scores[i],
            "preference": alpha_scores[i] - nonalpha_scores[i],
        })

    return pd.DataFrame(rows)


def atomic_write_csv(df: pd.DataFrame, out_path: str) -> None:
    tmp_path = out_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)


# =========================
# WORKER
# =========================

@dataclass
class WorkItem:
    model_name: str
    tokenizer_id: str
    tokens_per_step: int
    checkpoint: Dict[str, Any]

def run_work_item(item: WorkItem, device: str, out_dir: str) -> None:
    model_name = item.model_name
    ckpt = item.checkpoint
    eval_id = f"{model_name.split('/')[-1]}_{ckpt['checkpoint']}"
    out_path = os.path.join(out_dir, f"{eval_id}.csv")

    tokenizer = AutoTokenizer.from_pretrained(item.tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Resume logic
    if os.path.exists(out_path):
        print(f"\n📄 Found existing file: {eval_id}")
        check_result = check_prompts_in_file(out_path, LIST_OF_PROMPTS)

        if check_result["missing_pairs"] is None:
            prompts_to_run = LIST_OF_PROMPTS
            existing_df = None
        else:
            missing_pairs = check_result["missing_pairs"]
            if not missing_pairs:
                print("  ✅ All prompts complete, skipping")
                return
            prompts_to_run = sorted(set(p for p, _ in missing_pairs))
            existing_df = pd.read_csv(out_path)
    else:
        prompts_to_run = LIST_OF_PROMPTS
        existing_df = None

    print(f"\n🚀 Evaluating {ckpt['checkpoint']} ({ckpt['tokens']:,} tokens) - {len(prompts_to_run)} prompts")
    print("  📥 Loading model...")

    tmp_cache = tempfile.mkdtemp(prefix="hf_ckpt_")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=ckpt["tag"],
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            cache_dir=tmp_cache,
        ).to(device).eval()

        if USE_TORCH_COMPILE:
            try:
                model = torch.compile(model, mode=COMPILE_MODE)
                print("  ⚡ Model compiled with torch.compile")
            except Exception as e:
                print(f"  ⚠️ torch.compile failed (continuing uncompiled): {e}")

        dfs = []
        print("  🔄 Processing prompts...")
        for prompt_idx, prompt in enumerate(
            tqdm(prompts_to_run, desc="  prompts", leave=False), 1
        ):
            df_result = get_model_prefs(
                prompt, model_name, ckpt, tokenizer, model, device
            )
            dfs.append(df_result)
            print(f"    ✅ Completed prompt {prompt_idx}/{len(prompts_to_run)}")

        new_df = pd.concat(dfs, ignore_index=True)

        if existing_df is not None:
            existing_keys = set(zip(existing_df["prompt"], existing_df["binom"]))
            new_df = new_df[
                ~new_df.apply(
                    lambda r: (r["prompt"], r["binom"]) in existing_keys, axis=1
                )
            ]

        final_df = (
            pd.concat([existing_df, new_df], ignore_index=True)
            if existing_df is not None
            else new_df
        )

        print("  💾 Saving results...")
        atomic_write_csv(final_df, out_path)
        print(f"  ✅ Saved {len(final_df)} total rows ({len(new_df)} new) to {out_path}")

    finally:
        # 🔥 HARD CLEANUP (this is what fixes your disk issue)
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(tmp_cache, ignore_errors=True)



def worker_main(rank: int, items: List[WorkItem]) -> None:
    # Pin to one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)
    device = "cuda:0"

    os.makedirs(OUT_DIR, exist_ok=True)
    failures = []

    print(f"\n==============================")
    print(f"🚀 Worker rank={rank} using GPU={rank}")
    print(f"==============================\n")

    for item in items:
        try:
            print("=" * 60)
            print(f"🔍 [{rank}] Model: {item.model_name} | Checkpoint: {item.checkpoint['checkpoint']}")
            print("=" * 60)
            run_work_item(item, device=device, out_dir=OUT_DIR)
        except Exception as e:
            failures.append((item.model_name, item.checkpoint.get("checkpoint"), str(e)))
            print(f"❌ [{rank}] Failure: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    print(f"\n🏁 Worker {rank} complete.")
    if failures:
        print(f"⚠️ Worker {rank} had {len(failures)} failures:")
        for m, c, err in failures:
            print(f" • {m} {c}: {err}")


# =========================
# DRIVER
# =========================

def build_work_items() -> List[WorkItem]:
    items: List[WorkItem] = []

    for model_name, config in MODEL_CONFIGS.items():
        print("=" * 60)
        print(f"🔍 Model: {model_name}")
        print("=" * 60)

        source = config.get("checkpoint_source", "step_tags")
        tps    = config["tokens_per_step"]

        if tps is None:
            print(f"⚠️  tokens_per_step is None for {model_name} — fill in the training config before running.")
            print("⛔ Skipping.\n")
            continue

        try:
            if source == "step_branches":
                checkpoints = get_checkpoints_from_step_branches(model_name, tps)
            elif source == "stage1_branches":
                checkpoints = get_checkpoints_from_stage1_branches(model_name, tps)
            else:
                # step_tags: verify tag structure, then fetch
                verify_repo_is_tagged(model_name, expected_step_multiple=15, min_checkpoints=10)
                checkpoints = get_model_checkpoints(model_name, tps)
        except RuntimeError as e:
            print(str(e))
            print("⛔ Skipping this model.\n")
            continue

        if not checkpoints:
            print(f"⚠️  No checkpoints found for {model_name}, skipping.\n")
            continue

        if config.get("log_sample", False):
            checkpoints = log_sample_checkpoints(checkpoints, n=20)

        for ckpt in checkpoints:
            items.append(WorkItem(
                model_name=model_name,
                tokenizer_id=config["tokenizer"],
                tokens_per_step=tps,
                checkpoint=ckpt,
            ))

    # Sorted within each model already; across models, keep as appended
    print(f"\n🧾 Total work items: {len(items)} checkpoints\n")
    return items


def shard_items(items: List[WorkItem], num_shards: int) -> List[List[WorkItem]]:
    return [items[i::num_shards] for i in range(num_shards)]


def main():
    print("🔧 Building work items (all checkpoints across models)...")
    items = build_work_items()
    if not items:
        print("No checkpoints to run. Exiting.")
        return

    num_gpus = detect_num_gpus()
    print(f"\n🖥️  Detected {num_gpus} GPU(s)")

    os.makedirs(OUT_DIR, exist_ok=True)

    # =========================
    # CPU-only fallback
    # =========================
    if num_gpus == 0:
        print("⚠️ No GPUs detected — running serially on CPU")
        for item in items:
            run_work_item(
                item,
                device="cpu",
                out_dir=OUT_DIR,
            )
        print("\n🏁 RUN COMPLETE (CPU)")
        return

    # =========================
    # Single-GPU path
    # =========================
    if num_gpus == 1:
        print("🚀 Single GPU detected — running serially on the visible GPU")
        torch.cuda.set_device(0)
    
        for item in items:
            run_work_item(
                item,
                device="cuda:0",
                out_dir=OUT_DIR,
            )
        print("\n🏁 RUN COMPLETE (1 GPU)")
        return


    # =========================
    # Multi-GPU path
    # =========================
    print(f"🚀 Multi-GPU mode — sharding across {num_gpus} GPUs")
    shards = shard_items(items, num_gpus)

    for r, sh in enumerate(shards):
        print(f"GPU {r}: {len(sh)} checkpoints")

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(num_gpus):
        p = ctx.Process(target=worker_main, args=(rank, shards[rank]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("\n🏁 RUN COMPLETE (multi-GPU)")



if __name__ == "__main__":
    main()
