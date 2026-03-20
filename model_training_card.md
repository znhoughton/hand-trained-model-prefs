# Model Training Card

## Overview

Six autoregressive language models were trained using the OPT architecture across two corpora (BabyLM and C4) and three parameter scales (125M, 350M, 1.3B), for a total of six models. All models share the same tokenizer, context window, and hardware setup. Models are publicly available on HuggingFace Hub under `znhoughton/`.

---

## Tokenizer

A custom Byte-Pair Encoding (BPE) tokenizer was trained from scratch on the **full** BabyLM training corpus (`znhoughton/babylm-150m-v3`, all splits) and used for all six models.

| Property | Value |
|---|---|
| Type | BPE (Byte-Pair Encoding) |
| Vocabulary size | 8,192 tokens |
| Context window | 1,024 tokens |
| Base config | `facebook/opt-125m` tokenizer structure |
| HuggingFace ID | `znhoughton/opt-babylm-100m-bpe` |

The tokenizer was trained once and reused identically for all six models (both BabyLM and C4), ensuring that tokenization is consistent across corpora and model sizes.

---

## Training Infrastructure

| Property | Value |
|---|---|
| GPUs | 2× NVIDIA H100 |
| Parallelism | PyTorch Distributed Data Parallel (DDP) via `torchrun --nproc_per_node=2` |
| CUDA devices | `CUDA_VISIBLE_DEVICES=0,1` |
| Precision | bfloat16 (`--bf16`) |
| Gradient checkpointing | Enabled (`--gradient_checkpointing`) |
| Logging | TensorBoard (`--report_to tensorboard`), every 10 steps |
| Hub sync | HuggingFace Hub, `hub_strategy=checkpoint` |
| `ddp_find_unused_parameters` | False |

---

## Model Architecture

All models use the OPT (Open Pre-trained Transformer) decoder-only architecture (Zhang et al., 2022), initialized from the corresponding `facebook/opt-*` configuration but with weights reinitialized. Architecture configurations:

| Model scale | Parameters | Hidden size | Attention heads | Layers | FFN size | HF base config |
|---|---|---|---|---|---|---|
| 125M | ~125M | 768 | 12 | 12 | 3,072 | `facebook/opt-125m` |
| 350M | ~350M | 1,024 | 16 | 24 | 4,096 | `facebook/opt-350m` |
| 1.3B | ~1.3B | 2,048 | 32 | 24 | 8,192 | `facebook/opt-1.3b` |

All models use the shared BabyLM BPE tokenizer (vocab size 8,192; context window 1,024 tokens).

---

## Shared Hyperparameters (All Six Models)

| Hyperparameter | Value |
|---|---|
| Random seed | 964 (`--seed 964`) |
| Block size | 1,024 tokens |
| Learning rate warmup | 4,000 steps (`--warmup_steps 4000`) |
| Checkpoint save target | Every ~10,000,000 tokens |
| Checkpoints retained on disk | 1 (most recent only; `--save_total_limit 1`) |
| Optimizer | AdamW (HuggingFace Trainer default) |
| LR schedule | Linear decay with warmup |
| Weights saved to Hub | Yes (every checkpoint, `--hub_strategy checkpoint`) |

---

## Corpus 1: BabyLM

### Dataset

| Property | Value |
|---|---|
| HuggingFace dataset | `znhoughton/babylm-150m-v3` |
| Available splits | `train`, `dev` (no `validation` split) |
| Training split used | `train[5%:]` |
| Held-out split | `train[:5%]` (used as validation during training) |
| Approximate training tokens | ~142.5M tokens per epoch (95% of ~150M) |
| Total training tokens | ~9.12B tokens (64 epochs × ~142.5M) |
| Content | Child-directed speech (CHILDES), children's books, simple web text (Warstadt et al., 2023) |

**Why `train[5%:]`?** The training script (`train_autoreg.py`) checks for the exact split key `"validation"` using `if "validation" not in raw_datasets.keys()`. Although `babylm-150m-v3` does include a development set (the `"dev"` split), its key is named `"dev"`, not `"validation"` — so the code does not recognise it and the fallback fires. The fallback creates a validation set by slicing the first 5% of `"train"`, and the remaining 95% (`train[5%:]`) becomes the training split. This is the split that was actually used for training.

### Data Pipeline

Training used the standard HuggingFace `datasets` map-style (non-streaming) pipeline:

1. **Load**: `load_dataset("znhoughton/babylm-150m-v3", split="train[5%:]")`
2. **Tokenize**: Apply the BPE tokenizer to each document (`batched=True`, removing original columns)
3. **Group into blocks**: Concatenate all token IDs and chunk into non-overlapping 1,024-token blocks (`group_texts`). Any remainder tokens at the end of each batch are discarded.
4. **Shuffle per epoch**: At the start of each epoch, `DistributedSampler` shuffles the block indices using `torch.randperm(N, generator=g)` where `g.manual_seed(seed + epoch)` (i.e., seed = 964 + epoch number, starting from epoch 0). This shuffle is identical across both GPUs (rank 0 and rank 1) but the DDP sampler interleaves assignment: rank 0 gets indices `[0, 2, 4, ...]` and rank 1 gets `[1, 3, 5, ...]` from the shuffled sequence, after zero-padding to make N evenly divisible by 2.

### BabyLM Training Configuration

| Parameter | 125M | 350M | 1.3B |
|---|---|---|---|
| Per-device batch size | 400 | 300 | 100 |
| Gradient accumulation steps | 1 | 2 | 4 |
| Effective batch size (sequences) | 800 | 1,200 | 800 |
| Tokens per gradient step | 819,200 | 1,228,800 | 819,200 |
| Learning rate | 3 × 10⁻⁴ | 1 × 10⁻⁴ | 1 × 10⁻⁴ |
| Training epochs | 64 | 64 | 64 |
| Save every N steps | 12 | 8 | 12 |
| Total checkpoints | 912 | 912 | 860 |
| HuggingFace model ID | `znhoughton/opt-babylm-125m-64eps-seed964` | `znhoughton/opt-babylm-350m-64eps-seed964` | `znhoughton/opt-babylm-1.3b-64eps-seed964` |

*Tokens/step = block\_size × per\_device\_batch × grad\_accum\_steps × n\_GPUs = 1024 × batch × accum × 2*

*Save interval: `save_steps = floor(10,000,000 / tokens_per_step)`*

### BabyLM Checkpoint Token Counts (Final Checkpoint per Model)

| Model | Final step | Final tokens seen |
|---|---|---|
| opt-babylm-125m | 10,944 | 8,965,324,800 |
| opt-babylm-350m | 7,296 | 8,965,324,800 |
| opt-babylm-1.3b | 10,944 | 8,965,324,800 |

---

## Corpus 2: C4

### Dataset

| Property | Value |
|---|---|
| HuggingFace dataset | `znhoughton/c4-subset-10B-tokens` |
| Available splits | `train`, `validation` |
| Training split used | `train` (all 17.7M documents) |
| Approximate training tokens | ~10B tokens |
| Total training epochs | 1 (single pass through the corpus) |
| Content | Web text subset of the Colossal Clean Crawled Corpus (Raffel et al., 2020) |

**Why the full `train` split?** The training script checks for the exact split key `"validation"` using `if "validation" not in raw_datasets.keys()`. Because `c4-subset-10B-tokens` has a split named exactly `"validation"`, the condition is `False` and the fallback does not fire. The full `"train"` split is used as-is.

**Dataset composition**: The C4 subset (`znhoughton/c4-subset-10B-tokens`) was constructed by streaming and shuffling documents from the full `allenai/c4` English dataset, sampling by token budget until 10 billion tokens were collected, then writing to sharded Parquet files. The subset is therefore a random shuffled sample of the larger C4 corpus, not sequential first-N documents.

### Data Pipeline

Training used the HuggingFace `datasets` streaming (IterableDataset) pipeline:

1. **Load** (streaming): `load_dataset("znhoughton/c4-subset-10B-tokens", split="train", streaming=True)`
2. **Tokenize**: Apply the BPE tokenizer to each document (`batched=True`, removing original columns)
3. **Group into blocks**: Concatenate token IDs and chunk into non-overlapping 1,024-token blocks (`group_texts`). Remainders at batch boundaries are discarded.
4. **No shuffle**: The HuggingFace `Trainer` does **not** call `.shuffle()` on streaming/`IterableDataset` instances. `IterableDatasetShard` shards the stream across the two GPUs but does not reorder documents. Document order is fixed by the alphabetical reading order of the Parquet shard files on HuggingFace Hub and is fully deterministic across runs.

### C4 Training Configuration

| Parameter | 125M | 350M | 1.3B |
|---|---|---|---|
| Per-device batch size | 400 | 300 | 100 |
| Gradient accumulation steps | 1 | 2 | 4 |
| Effective batch size (sequences) | 800 | 1,200 | 800 |
| Tokens per gradient step | 819,200 | 1,228,800 | 819,200 |
| Learning rate | 3 × 10⁻⁴ | 1 × 10⁻⁴ | 1 × 10⁻⁴ |
| Max steps | 12,207 | 8,138 | 12,207 |
| Total training tokens | ~10B | ~10B | ~10B |
| Save every N steps | 12 | 8 | 12 |
| Total checkpoints | 1,014 | 1,005 | 960 |
| HuggingFace model ID | `znhoughton/opt-c4-125m-seed964` | `znhoughton/opt-c4-350m-seed964` | `znhoughton/opt-c4-1.3b-seed964` |

*C4 uses `--max_steps` (not `--num_train_epochs`) because the dataset is streamed and has no epoch boundary.*

### C4 Checkpoint Token Counts (Final Checkpoint per Model)

| Model | Final step | Final tokens seen |
|---|---|---|
| opt-c4-125m | 12,192 | 9,987,686,400 |
| opt-c4-350m | 8,128 | 9,987,686,400 |
| opt-c4-1.3b | 12,192 | 9,987,686,400 |

---

## Checkpointing Details

Checkpoints were saved approximately every 10,000,000 tokens throughout training. Because `save_steps` is computed as `floor(10,000,000 / tokens_per_step)`, the actual interval is slightly less than 10M tokens for model sizes where `tokens_per_step` does not evenly divide 10M:

| Model scale | Tokens/step | save_steps | Actual tokens/checkpoint |
|---|---|---|---|
| 125M | 819,200 | 12 | 9,830,400 |
| 350M | 1,228,800 | 8 | 9,830,400 |
| 1.3B | 819,200 | 12 | 9,830,400 |

The `--save_total_limit 1` flag meant only the single most recent checkpoint was retained on disk at any time during training. All intermediate checkpoints were pushed to HuggingFace Hub (via `--hub_strategy checkpoint`) before local deletion, making the full checkpoint history available on the Hub.

The `--save_only_model` flag was used for C4 models, meaning optimizer and scheduler states were not saved to Hub (weights only). BabyLM models did not use this flag in the training script, though in practice weights-only checkpoint files were pushed to Hub for all models.

---

## Reproducibility Notes

### BabyLM

- Exact data order is reproducible via `torch.randperm(N, generator=g)` with `g.manual_seed(964 + epoch)` for epoch 0, 1, ..., 63. The same seed formula is used internally by PyTorch's `DistributedSampler`.
- The DDP interleaving assigns blocks to GPUs: rank 0 sees shuffled indices `[0, 2, 4, ...]`, rank 1 sees `[1, 3, 5, ...]`. The effective training sequence seen by the model is the interleaved combination.
- `torch.randperm` output may differ across PyTorch **major** versions even with the same seed. Exact replay requires the same PyTorch major version used during training.
- The `train[5%:]` slice is applied before any shuffling; it is a fixed prefix/suffix split of the dataset's document order.

### C4

- Document order is fixed by the Parquet shard reading order on HuggingFace Hub (alphabetical by shard filename). This order is stable across runs.
- No per-epoch or per-batch shuffle occurs for streaming datasets in the HuggingFace Trainer.
- Exact token block boundaries depend on the `group_texts` batching (default batch size 1,000 documents per `map` call); blocks that would cross the 1,000-document batch boundary are discarded and a new block starts from the next document.

---

## Training Command Template

### BabyLM (example: 125M)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_autoreg.py \
    --model_type opt \
    --config_name models/opt-babylm-125m-64eps \
    --tokenizer_name models/opt-babylm-100m-bpe \
    --dataset_name znhoughton/babylm-150m-v3 \
    --do_train \
    --bf16 \
    --gradient_checkpointing \
    --block_size 1024 \
    --per_device_train_batch_size 400 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --warmup_steps 4000 \
    --save_steps 12 \
    --save_total_limit 1 \
    --logging_steps 10 \
    --report_to tensorboard \
    --num_train_epochs 64 \
    --seed 964 \
    --output_dir runs/opt-babylm-125m-64eps_964-64eps \
    --push_to_hub \
    --hub_model_id znhoughton/opt-babylm-125m-64eps-seed964 \
    --hub_strategy checkpoint \
    --ddp_find_unused_parameters False
```

### C4 (example: 125M)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_autoreg.py \
    --model_type opt \
    --config_name models/opt-c4-125m \
    --tokenizer_name models/opt-babylm-100m-bpe \
    --dataset_name znhoughton/c4-subset-10B-tokens \
    --max_steps 12207 \
    --do_train \
    --streaming \
    --preprocessing_num_workers 8 \
    --bf16 \
    --gradient_checkpointing \
    --block_size 1024 \
    --per_device_train_batch_size 400 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --warmup_steps 4000 \
    --save_steps 12 \
    --save_total_limit 1 \
    --save_only_model \
    --logging_steps 10 \
    --report_to tensorboard \
    --seed 964 \
    --output_dir runs/opt-c4-125m_964 \
    --push_to_hub \
    --hub_model_id znhoughton/opt-c4-125m-seed964 \
    --hub_strategy checkpoint \
    --ddp_find_unused_parameters False
```

---

## Validation Results

Training perplexity (exp of cross-entropy loss) was computed at initialization and at the final checkpoint for all six models using `analysis-scripts/evaluate_training_quality.py`. Initial loss for all models is approximately ln(8,192) ≈ 9.01, consistent with random initialization over the vocabulary. All models converged substantially over training.

### Final-checkpoint perplexity (training set)

| Corpus | Params | Initial loss | Final loss | Initial PPL | Final PPL |
|--------|-------:|-------------:|-----------:|------------:|----------:|
| BabyLM | 125M   | 9.182        | 2.183      | ~9,736      | 8.9       |
| BabyLM | 350M   | 9.156        | 2.513      | ~9,484      | 12.3      |
| BabyLM | 1.3B   | 9.479        | 1.640      | ~13,083     | 5.2       |
| C4     | 125M   | 9.162        | 3.181      | ~9,530      | 24.1      |
| C4     | 350M   | 9.143        | 3.529      | ~9,350      | 34.2      |
| C4     | 1.3B   | 9.415        | 2.998      | ~12,224     | 20.0      |

**Notes:**
- Loss = mean cross-entropy per token; PPL = exp(loss). Values from `trainer_state.json` log history (training set).
- Within each corpus, the 1.3B model achieves the lowest final loss. The 125M outperforms the 350M in both corpora, consistent with the 350M being relatively data-hungry at this training budget (~9–10B tokens).
- Validation perplexity (separate held-out sets) is higher and shows evaluation artefacts for the BabyLM models; training loss above is the authoritative convergence measure.
- Full loss curves saved to `Data/training_quality/loss_curves.{csv,png}`.

---

## Repository Structure & Scripts

### Directory Overview

```
hand-trained-model-prefs/
├── model_training_card.md          # This file
├── model-training-code/            # Training scripts (train_autoreg.py, etc.)
├── Writeup/                        # Paper writeup (writeup.qmd + CoLM extension)
├── Data/
│   ├── checkpoint_results/         # Raw per-checkpoint scoring CSVs (from model-prefs-all-ckpts.py)
│   ├── processed/                  # Aggregated CSVs used by R analyses
│   │   └── checkpoint_results_with_exposures.csv  # ⚠ gitignored (30 GB) — see below
│   └── training_quality/           # Perplexity and loss curve outputs
└── analysis-scripts/
    ├── models/                     # Fitted brms model .rds files
    ├── model-prefs-all-ckpts.py
    ├── count_corpus_exposures.py
    ├── build_checkpoint_exposures.py
    ├── prepare_data.py
    ├── evaluate_training_quality.py
    ├── analysis.Rmd / analysis.html
    └── analysis_cleaned_up.Rmd / analysis_cleaned_up.html
```

### Python Scripts

The Python scripts form a sequential data pipeline. Run them in order to reproduce the full dataset from scratch.

#### 1. `model-prefs-all-ckpts.py`
Runs inference across all model checkpoints (loaded from HuggingFace Hub) to score every binomial pair under every prompt. For each checkpoint, records `alpha_logprob` and `nonalpha_logprob` — the model's log-probabilities for the alpha-first and non-alpha-first orderings of each binomial. Output: per-checkpoint CSV files written to `Data/checkpoint_results/`. Multi-GPU (2× H100), resume-safe, OOM-safe adaptive batch sizing.

#### 2. `count_corpus_exposures.py`
Replays the training data in exact training order to compute cumulative corpus exposure counts (`alpha_seen`, `beta_seen`) for each binomial at each checkpoint step. This is a full replay of ~9–10B tokens per corpus. Output: exposure count CSV.

#### 3. `build_checkpoint_exposures.py`
Joins the raw per-checkpoint model preference data (from step 1) with the cumulative corpus exposure counts (from step 2). Output: `Data/processed/checkpoint_results_with_exposures.csv` — one row per model × checkpoint × prompt × binomial (~209M rows, ~30 GB). **This file is gitignored** due to size; run this script to regenerate it locally.

#### 4. `prepare_data.py`
Aggregates the large joined CSV into smaller, R-readable CSVs in `Data/processed/` (e.g., prompt-averaged preference per binomial × checkpoint, accuracy summaries). These smaller outputs **are** committed to the repo, so collaborators who only want to run the R analyses do not need to run steps 1–4.

#### 5. `evaluate_training_quality.py`
Computes final-checkpoint perplexity for all six models on held-out data, and fetches training loss curves from `trainer_state.json` on HuggingFace Hub. Output: `Data/training_quality/perplexity_results.csv`, `loss_curves.csv`, and corresponding plots. Run this independently of the main pipeline.

### R Analysis Scripts

#### `analysis.Rmd`
Main analysis document. Loads fitted brms models from `analysis-scripts/models/` and produces all coefficient plots, trajectory plots, and summary tables. Reads `Data/processed/checkpoint_results_with_exposures.csv` (via DuckDB) to recover per-checkpoint token counts for trajectory x-axes — this requires the 30 GB file to be present locally. All other data dependencies (the processed CSVs) are committed to the repo.

#### `analysis_cleaned_up.Rmd`
Cleaned-up version of `analysis.Rmd` with the same analyses, formatted for sharing.

#### `analysis-scripts/models/`
Fitted brms model `.rds` files. These are large and may not be committed; if absent, the Rmd files will warn that no fits were found and skip the corresponding plots.

### What Collaborators Need

| Goal | Required files/scripts |
|---|---|
| Read analysis results | Open `analysis_cleaned_up.html` in a browser — no setup needed |
| Re-render the Rmd | R + brms + the `models/` `.rds` files; processed CSVs are in the repo |
| Regenerate trajectory x-axes | Also need `checkpoint_results_with_exposures.csv` (run steps 1–3 above) |
| Reproduce from scratch | All Python scripts + HuggingFace Hub access + ~2× H100 GPUs |

---

## References

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1–67.

Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., ... & Williams, A. (2023). Findings of the BabyLM challenge: Sample-efficient pretraining on a developmentally plausible corpus. In *Proceedings of the BabyLM Challenge at CoNLL 2023*.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., ... & Zettlemoyer, L. (2022). OPT: Open pre-trained transformer language models. *arXiv preprint arXiv:2205.01068*.
