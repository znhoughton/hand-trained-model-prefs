# ==========================================================
#  IMPORTS
# ==========================================================
import os
import torch
import pandas as pd
import numpy as np
from torch import cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
from tqdm import tqdm
import glob

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

# Add at the top with other imports
import torch._dynamo

#### confirm tags
def verify_repo_is_tagged(
    repo_id,
    expected_step_multiple,
    min_checkpoints=5,
):
    api = HfApi()
    refs = api.list_repo_refs(repo_id)

    # -----------------------------
    # 1) Collect step-* tags
    # -----------------------------
    tag_steps = sorted(
        int(tag.name.split("-")[1])
        for tag in refs.tags
        if tag.name.startswith("step-")
    )

    if not tag_steps:
        raise RuntimeError(f"❌ {repo_id} has no step-* tags")

    if len(tag_steps) < min_checkpoints:
        raise RuntimeError(
            f"❌ {repo_id} only has {len(tag_steps)} tags"
        )

    # -----------------------------
    # 2) Alignment check (soft)
    # -----------------------------
    misaligned = [s for s in tag_steps if s % expected_step_multiple != 0]

    # DO NOT fail — just warn
    if misaligned:
        print(
            f"⚠️  {repo_id} has non-multiple steps (likely final save): "
            f"{misaligned[:5]}{'...' if len(misaligned) > 5 else ''}"
        )

    # -----------------------------
    # 3) Delta sanity check
    # -----------------------------
    deltas = sorted(set(b - a for a, b in zip(tag_steps, tag_steps[1:])))

    # -----------------------------
    # 4) Report (NO fake missing logic)
    # -----------------------------
    print(
        f"✅ {repo_id} tagging OK:\n"
        f"   • checkpoints: {len(tag_steps)}\n"
        f"   • steps: {tag_steps[0]} → {tag_steps[-1]}\n"
        f"   • expected base step: {expected_step_multiple}\n"
        f"   • observed deltas: {deltas}"
    )

    return {
        "steps": tag_steps,
        "deltas": deltas,
    }


# ==========================================================
#  OPTIMIZED LOGPROB SCORING
# ==========================================================
@torch.no_grad()
@torch.no_grad()
def to_tokens_and_logprobs(
    model,
    tokenizer,
    input_texts,
    batch_size=32,
    desc=None,
    leave=False,
):
    all_logprobs = []

    total_batches = (len(input_texts) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(input_texts), batch_size),
        desc=desc,
        total=total_batches,
        leave=leave,
    ):
        batch_texts = input_texts[i:i + batch_size]

        try:
            enc = tokenizer(batch_texts, padding=True, return_tensors="pt")

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1)

            logprobs = logprobs[:, :-1, :]
            target_ids = input_ids[:, 1:]
            target_mask = attention_mask[:, 1:]

            token_logprobs = logprobs.gather(
                2, target_ids.unsqueeze(-1)
            ).squeeze(-1)
            token_logprobs = token_logprobs * target_mask

            all_logprobs.extend(token_logprobs.sum(dim=-1).tolist())

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if batch_size == 1:
                    raise RuntimeError("OOM even at batch_size=1")
                torch.cuda.empty_cache()
                return to_tokens_and_logprobs(
                    model,
                    tokenizer,
                    input_texts,
                    batch_size // 2,
                    desc=desc,
                    leave=leave,
                )
            else:
                raise

    return all_logprobs



# ==========================================================
#  OPTIMIZED BINOMIAL PREFERENCE SCORING
# ==========================================================
def get_model_prefs(prompt, model_name, checkpoint_info, tokenizer, model):
    df = pd.read_csv("../Data/nonce_and_attested_binoms.csv")

    df["AandB"] = prompt + df["Word1"] + " and " + df["Word2"]
    df["BandA"] = prompt + df["Word2"] + " and " + df["Word1"]

    binomial_alpha = df["AandB"].tolist()
    binomial_nonalpha = df["BandA"].tolist()

    # More conservative batch sizes to avoid OOM
    # For 125M: batch_size=64, 350M: batch_size=48, 1.3B: batch_size=32
    if '125m' in model_name.lower():
        batch_size = 768
    elif '350m' in model_name.lower():
        batch_size = 768
    else:
        batch_size = 256

    print(f"    Processing with batch_size={batch_size}")
    
    # Much simpler - just two calls instead of 100+ batches each
    try:
        alpha_scores = to_tokens_and_logprobs(
            model,
            tokenizer,
            binomial_alpha,
            batch_size,
            desc="      A and B",
            leave=False,
        )

        nonalpha_scores = to_tokens_and_logprobs(
            model,
            tokenizer,
            binomial_nonalpha,
            batch_size,
            desc="      B and A",
            leave=False,
        )

    except Exception as e:
        print(f"    ❌ Error during scoring: {e}")
        raise e

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


# ==========================================================
#  CHECKPOINT DISCOVERY (TAG-BASED, FINAL)
# ==========================================================
def get_model_checkpoints(repo_id, tokens_per_step):
    """
    Discover checkpoints via step-* tags.
    Tokens are computed deterministically as:
        tokens = step * tokens_per_step
    """
    api = HfApi()
    refs = api.list_repo_refs(repo_id)

    checkpoints = []
    for tag in refs.tags:
        if not tag.name.startswith("step-"):
            continue

        step = int(tag.name.split("-")[1])
        checkpoints.append({
            "checkpoint": tag.name,      # step-XXXX
            "tag": tag.name,
            "step": step,
            "tokens": step * tokens_per_step
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


# ==========================================================
#  PROMPT CHECKING FOR EXISTING FILES
# ==========================================================
def check_prompts_in_file(filepath, expected_prompts):
    """
    Check which prompts are present/missing in an existing CSV file,
    and for each present prompt, verify it has complete binomial coverage.
    
    Returns:
        dict with keys:
            - 'missing_prompts': list of prompts not in file at all
            - 'incomplete_prompts': list of prompts with incomplete binomial coverage
            - 'prompts_to_run': combined list of prompts that need to be (re)evaluated
    """
    # Load the expected binomials
    binoms_df = pd.read_csv("../Data/nonce_and_attested_binoms.csv")
    expected_binoms = set(
        f"{row.Word1} and {row.Word2}" 
        for row in binoms_df.itertuples()
    )
    expected_binom_count = len(expected_binoms)
    
    try:
        df = pd.read_csv(filepath)
        existing_prompts = set(df['prompt'].unique())
        expected_set = set(expected_prompts)
        
        # Find prompts that are completely missing
        missing_prompts = sorted(expected_set - existing_prompts)
        
        # For each existing prompt, check if it has all binomials
        incomplete_prompts = []
        for prompt in existing_prompts & expected_set:
            prompt_df = df[df['prompt'] == prompt]
            prompt_binoms = set(prompt_df['binom'].unique())
            
            if len(prompt_binoms) != expected_binom_count:
                incomplete_prompts.append(prompt)
                print(
                    f"    ⚠️ Prompt '{prompt[:30]}...' has {len(prompt_binoms)}/{expected_binom_count} binomials"
                )
        
        # Combine missing and incomplete prompts
        prompts_to_run = sorted(set(missing_prompts + incomplete_prompts))
        
        if missing_prompts:
            print(f"    📝 {len(missing_prompts)} prompts missing entirely")
        if incomplete_prompts:
            print(f"    ⚠️ {len(incomplete_prompts)} prompts have incomplete binomial coverage")
        
        return {
            'missing_prompts': missing_prompts,
            'incomplete_prompts': incomplete_prompts,
            'prompts_to_run': prompts_to_run
        }
        
    except Exception as e:
        print(f"    ⚠️ Error reading {filepath}: {e}")
        print(f"    Will re-run all prompts for this checkpoint")
        return {
            'missing_prompts': expected_prompts,
            'incomplete_prompts': [],
            'prompts_to_run': expected_prompts
        }


# ==========================================================
#  MODEL CONFIGURATIONS
# ==========================================================
MODEL_CONFIGS = {
    # BabyLM
    "znhoughton/opt-babylm-125m-seed42": {
        # tokens/step = 1024 × 320 × 1 × 2
        "tokens_per_step": 655_360,
        "tokenizer": "znhoughton/opt-babylm-125m-seed42",
    },
    "znhoughton/opt-babylm-350m-seed42": {
        # tokens/step = 1024 × 80 × 2 × 2
        "tokens_per_step": 327_680,
        "tokenizer": "znhoughton/opt-babylm-350m-seed42",
    },
    "znhoughton/opt-babylm-1.3b-seed42": {
        # tokens/step = 1024 × 40 × 4 × 2
        "tokens_per_step": 327_680,
        "tokenizer": "znhoughton/opt-babylm-1.3b-seed42",
    },

    # C4
    # "znhoughton/opt-c4-125m-seed42": {
    #     "tokens_per_step": 655_360,
    #     "tokenizer": "znhoughton/opt-pile-125m-seed42",
    # },
    # "znhoughton/opt-c4-350m-seed42": {
    #     "tokens_per_step": 327_680,
    #     "tokenizer": "znhoughton/opt-pile-350m-seed42",
    # },
    # "znhoughton/opt-c4-1.3b-seed42": {
    #     "tokens_per_step": 327_680,
    #     "tokenizer": "znhoughton/opt-pile-1.3b-seed42",
    # },
}


# ==========================================================
#  PROMPTS
# ==========================================================
list_of_prompts = [
    " ",
    "Especially the ",
    "For instance ",
    "In some cases ",
    "Every now and then ",
    "Occasionally you'll find ",
    "There can be examples like ",
    "You might notice things like ",
    "People sometimes mention ",
    "Sometimes just ",
    # "Nothing specific comes to mind except the ",
    # "It reminded me loosely of the ",
    # "There was a vague reference to the ",
    # "Unexpectedly the ",
    # "It's easy to overlook the ",
    # "There used to be talk of ",
    # "Out in the distance was the ",
    # "What puzzled everyone was the ",
    # "At some point I overheard ",
    # "Without warning came ",
    # "A friend once described the ",
    # "The scene shifted toward ",
    # "Nobody expected to hear about ",
    # "Things eventually turned toward ",
    # "The conversation eventually returned to ",
    # "I only remember a hint of the ",
    # "I couldn't quite place the ",
    # "It somehow led back to the ",
    # "What stood out most was the ",
    # "The oddest part involved the ",
    # "Later on, people were discussing ",
    # "There was this fleeting idea about ",
    # "I once heard someone bring up the ",
    # "There was a moment involving the ",
    # "It all started when we noticed the ",
    # "Another example floated around concerning the ",
    # "I came across something about the ",
    # "A situation arose involving the ",
    # "The conversation drifted toward the ",
    # "At one point we ended up discussing the ",
]


# ==========================================================
#  MAIN LOOP
# ==========================================================
def main():
    verified_models = {}

    for model_name, config in MODEL_CONFIGS.items():
        print("=" * 60)
        print(f"🔍 Verifying model: {model_name}")
        print("=" * 60)

        try:
            verify_repo_is_tagged(
                model_name,
                expected_step_multiple=15,
                min_checkpoints=10,
            )
            verified_models[model_name] = config
        except RuntimeError as e:
            print(str(e))
            print("⛔ Skipping this model.\n")


    out_dir = "../Data/checkpoint_results"
    os.makedirs(out_dir, exist_ok=True)
    failed_evals = []

    for model_name, config in verified_models.items():
        print("=" * 60)
        print(f"🔍 Processing model: {model_name}")
        print("=" * 60)

        checkpoints = get_model_checkpoints(
            model_name,
            config["tokens_per_step"]
        )

        if not checkpoints:
            continue

        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        for ckpt in tqdm(checkpoints, desc=model_name.split("/")[-1]):
            eval_id = f"{model_name.split('/')[-1]}_{ckpt['checkpoint']}"
            out_path = os.path.join(out_dir, f"{eval_id}.csv")

            # Check if file exists and what prompts need to be run
            if os.path.exists(out_path):
                print(f"\n📄 Found existing file: {eval_id}")
                check_result = check_prompts_in_file(out_path, list_of_prompts)
                prompts_to_run = check_result['prompts_to_run']
                
                if not prompts_to_run:
                    print(f"  ✅ All prompts complete, skipping")
                    continue
                else:
                    print(f"  🔄 Need to evaluate {len(prompts_to_run)} prompts")
                    # Load existing data to append to
                    existing_df = pd.read_csv(out_path)
                    # Remove any incomplete prompt data
                    existing_df = existing_df[
                        ~existing_df['prompt'].isin(check_result['incomplete_prompts'])
                    ]
            else:
                prompts_to_run = list_of_prompts
                existing_df = None

            print(
                f"\n🚀 Evaluating {ckpt['checkpoint']} "
                f"({ckpt['tokens']:,} tokens) - {len(prompts_to_run)} prompts"
            )

            try:
                print("  📥 Loading model...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    revision=ckpt["tag"],
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32
                ).to(device).eval()
                
                print("  ✅ Model loaded successfully")
                
                if os.name != "nt":  # only compile on Linux
                    try:
                        model = torch.compile(model, mode="reduce-overhead")
                        print("  ⚡ Model compiled with torch.compile")
                    except Exception:
                        pass


                print("  🔄 Processing prompts...")
                dfs = []
                for prompt_idx, prompt in enumerate(tqdm(prompts_to_run,
                                   desc="  prompts",
                                   leave=False), 1):
                    try:
                        df_result = get_model_prefs(
                            prompt,
                            model_name,
                            ckpt,
                            tokenizer,
                            model
                        )
                        dfs.append(df_result)
                        print(f"    ✅ Completed prompt {prompt_idx}/{len(prompts_to_run)}")
                    except Exception as e:
                        print(f"    ❌ Failed on prompt {prompt_idx}: {e}")
                        raise e

                print("  📊 Concatenating results...")
                new_df = pd.concat(dfs, ignore_index=True)
                
                # Combine with existing data if any
                if existing_df is not None:
                    final_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    final_df = new_df

                # Save atomically
                print("  💾 Saving results...")
                tmp_path = out_path + ".tmp"
                final_df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, out_path)
                
                print(f"  ✅ Saved {len(final_df)} total rows ({len(new_df)} new) to {out_path}")

            except Exception as e:
                failed_evals.append(f"{eval_id}: {e}")
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()

            finally:
                print("  🧹 Cleaning up...")
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()
                print("  ✅ Cleanup complete")

    print("\n🏁 RUN COMPLETE")

    if failed_evals:
        print(f"\n⚠️ {len(failed_evals)} failures:")
        for f in failed_evals:
            print(" •", f)


if __name__ == "__main__":
    main()