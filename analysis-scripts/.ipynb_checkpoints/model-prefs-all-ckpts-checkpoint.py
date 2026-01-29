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
import re
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


@torch.no_grad()
def to_tokens_and_logprobs(model, tokenizer, input_texts):

    enc = tokenizer(input_texts, padding=True, return_tensors="pt")
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

    return token_logprobs.sum(dim=-1).tolist()


def get_model_prefs(prompt, model_name, checkpoint_info, tokenizer, model):

    df = pd.read_csv('../Data/nonce_and_attested_binoms.csv')

    df['AandB'] = prompt + df['Word1'] + ' and ' + df['Word2']
    df['BandA'] = prompt + df['Word2'] + ' and ' + df['Word1']

    binomial_alpha = df['AandB'].tolist()
    binomial_nonalpha = df['BandA'].tolist()

    seqs_alpha = np.array_split(binomial_alpha, 100)
    alpha_scores = []
    for batch in tqdm(
        seqs_alpha,
        desc="  alpha batches",
        leave=False
    ):
        alpha_scores.extend(to_tokens_and_logprobs(model, tokenizer, batch.tolist()))

    seqs_nonalpha = np.array_split(binomial_nonalpha, 100)
    nonalpha_scores = []
    for batch in tqdm(
        seqs_nonalpha,
        desc="  nonalpha batches",
        leave=False
    ):
        nonalpha_scores.extend(to_tokens_and_logprobs(model, tokenizer, batch.tolist()))

    rows = []
    for i, row in enumerate(df.itertuples()):
        rows.append({
            "model": model_name,
            "checkpoint": checkpoint_info["checkpoint"],
            "commit_sha": checkpoint_info["commit_sha"],
            "tokens": checkpoint_info["tokens"],
            "prompt": prompt,
            "binom": f"{row.Word1} and {row.Word2}",
            "alpha_logprob": alpha_scores[i],
            "nonalpha_logprob": nonalpha_scores[i],
            "preference": alpha_scores[i] - nonalpha_scores[i]
        })

    return pd.DataFrame(rows)


def extract_tokens_from_commit(commit_title, tokens_per_step):
    """
    Extract checkpoint number from commit title and calculate tokens.
    Handles titles like "Training in progress, step 96" or "checkpoint-96"
    """
    # Try to find step number in commit title
    match = re.search(r'step[- ](\d+)', commit_title, re.IGNORECASE)
    if not match:
        match = re.search(r'checkpoint[- ](\d+)', commit_title, re.IGNORECASE)
    
    if match:
        step = int(match.group(1))
        tokens = step * tokens_per_step
        return step, tokens
    
    return None, None


def get_model_checkpoints(repo_id, tokens_per_step):
    """
    Get all checkpoint commits for a model from HuggingFace.
    Returns list of dicts with checkpoint info.
    """
    api = HfApi()
    
    try:
        commits = api.list_repo_commits(repo_id)
    except Exception as e:
        print(f"⚠️ Could not fetch commits for {repo_id}: {e}")
        return []
    
    checkpoints = []
    for commit in commits:
        step, tokens = extract_tokens_from_commit(commit.title, tokens_per_step)
        
        if step is not None:
            checkpoints.append({
                "checkpoint": f"checkpoint-{step}",
                "commit_sha": commit.commit_id,
                "tokens": tokens,
                "step": step,
                "commit_title": commit.title
            })
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x["step"])
    
    return checkpoints


# ==========================================================
#  MODEL CONFIGURATIONS
# ==========================================================
MODEL_CONFIGS = {
    # BabyLM models - 100M dataset
    "znhoughton/opt-babylm-125m-seed42": {
        "tokens_per_step": 655360,  # 1024 × 320 × 1 × 2
        "tokenizer": "znhoughton/opt-babylm-125m-seed42"  # Use HuggingFace repo ID
    },
    "znhoughton/opt-babylm-350m-seed42": {
        "tokens_per_step": 327680,  # 1024 × 80 × 2 × 2
        "tokenizer": "znhoughton/opt-babylm-350m-seed42"  
    },
    "znhoughton/opt-babylm-1.3b-seed42": {
        "tokens_per_step": 327680,  # 1024 × 40 × 4 × 2
        "tokenizer": "znhoughton/opt-babylm-1.3b-seed42"  
    },
    
    # Pile models - 10B dataset
    "znhoughton/opt-pile-125m-seed42": {
        "tokens_per_step": 655360,  # 1024 × 320 × 1 × 2
        "tokenizer": "znhoughton/opt-pile-125m-seed42"  
    },
    "znhoughton/opt-pile-350m-seed42": {
        "tokens_per_step": 327680,  # 1024 × 80 × 2 × 2
        "tokenizer": "znhoughton/opt-pile-350m-seed42"  
    },
    "znhoughton/opt-pile-1.3b-seed42": {
        "tokens_per_step": 327680,  # 1024 × 40 × 4 × 2
        "tokenizer": "znhoughton/opt-pile-1.3b-seed42"  
    },
}


# ==========================================================
#  MAIN LOOP
# ==========================================================
list_of_prompts = [
    " "
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
    "At one point we ended up discussing the "
]

def main():
    out_dir = "../Data/checkpoint_results"
    os.makedirs(out_dir, exist_ok=True)
    failed_evals = []

    # --------------------------------------------------
    # 🔍 Find already–completed evaluations
    # --------------------------------------------------
    existing_files = glob.glob(os.path.join(out_dir, "*.csv"))
    done_evals = set()
    for f in existing_files:
        # Parse filename like: opt-babylm-125m-seed42_checkpoint-96.csv
        basename = os.path.splitext(os.path.basename(f))[0]
        done_evals.add(basename)

    print("\n==============================================")
    print(f"📦 Found {len(done_evals)} completed checkpoint evaluations")
    print("==============================================\n")

    # --------------------------------------------------
    # 🔄 Loop through models and checkpoints
    # --------------------------------------------------
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"🔍 Processing model: {model_name}")
        print(f"{'='*60}")
        
        tokens_per_step = config["tokens_per_step"]
        tokenizer_path = config["tokenizer"]
        
        # Get all checkpoints for this model
        checkpoints = get_model_checkpoints(model_name, tokens_per_step)
        
        if not checkpoints:
            print(f"⚠️ No checkpoints found for {model_name}")
            continue
        
        print(f"✅ Found {len(checkpoints)} checkpoints")
        
        # Load tokenizer once per model
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        except Exception as e:
            msg = f"FAILED TO LOAD TOKENIZER for {model_name}: {e}"
            print(f"🚨 {msg}")
            failed_evals.append(msg)
            continue
        
        # Evaluate each checkpoint
        for checkpoint_info in tqdm(checkpoints, desc=f"{model_name.split('/')[-1]} checkpoints"):
            safe_model_id = model_name.split("/")[-1]
            safe_checkpoint = checkpoint_info["checkpoint"]
            eval_id = f"{safe_model_id}_{safe_checkpoint}"
            out_path = os.path.join(out_dir, f"{eval_id}.csv")
            
            # Skip if already done
            if eval_id in done_evals:
                continue
            
            print(f"\n  🚀 Evaluating {safe_checkpoint} ({checkpoint_info['tokens']:,} tokens)")
            
            try:
                # Load model from specific commit
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    revision=checkpoint_info["commit_sha"],
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32
                ).to(device).eval()
                
                # Run scoring for all prompts
                per_prompt_results = []
                for prompt in tqdm(list_of_prompts, desc=f"  {safe_checkpoint} prompts", leave=False):
                    df_out = get_model_prefs(prompt, model_name, checkpoint_info, tokenizer, model)
                    per_prompt_results.append(df_out)
                
                final_df = pd.concat(per_prompt_results, ignore_index=True)
                
                # Atomic write
                tmp_path = out_path + ".tmp"
                final_df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, out_path)
                
                print(f"  💾 Saved → {out_path}")
                
            except Exception as e:
                msg = f"FAILED: {eval_id} — {e}"
                print(f"\n  🚨 {msg}")
                failed_evals.append(msg)
            
            finally:
                # Cleanup GPU memory
                try:
                    del model
                    torch.cuda.empty_cache()
                except:
                    pass

    # --------------------------------------------------
    # 📊 Combine all results
    # --------------------------------------------------
    print("\n==============================================")
    print("📊 Combining all checkpoint results...")
    print("==============================================\n")
    
    files = glob.glob(os.path.join(out_dir, "*.csv"))
    if files:
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        combined_path = "../Data/checkpoint_results_combined.csv"
        df.to_csv(combined_path, index=False)
        print(f"✅ Combined results saved to {combined_path}")
        print(f"   Total rows: {len(df):,}")
    
    # --------------------------------------------------
    # 🏁 Summary
    # --------------------------------------------------
    print("\n==============================================")
    print("🏁 RUN COMPLETE")
    print("==============================================")

    if failed_evals:
        print(f"\n⚠️ {len(failed_evals)} evaluations failed:\n")
        for m in failed_evals:
            print("  •", m)
        
        log_path = os.path.join(out_dir, "failed_evals.log")
        with open(log_path, "w", encoding="utf-8") as f:
            for m in failed_evals:
                f.write(m + "\n")
        print(f"\n📝 Wrote failure log → {log_path}")
    else:
        print("\n🎉 No failures — all checkpoints evaluated!\n")


if __name__ == "__main__":
    main()