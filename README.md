# Hand-Trained Model Preferences

This repository contains all code, data, and writing for a study of binomial ordering preferences in transformer language models across training. The project trains six autoregressive language models from scratch, scores their ordering preferences on a set of binomial expressions at every checkpoint throughout training, and analyses how those preferences relate to corpus frequency statistics and abstract phonological/semantic constraints.

---

## Research Overview

**Binomials** are fixed multi-word expressions with a conventional ordering (e.g., *bread and butter*, *salt and pepper*). Human ordering preferences are shaped by both abstract phonological and semantic constraints (e.g., shorter words first, more animate words first) and item-specific corpus frequency. This project asks:

1. Do language models trained on developmentally plausible amounts of data develop binomial ordering preferences?
2. To what extent are those preferences driven by corpus frequency versus abstract constraints?
3. How do these effects evolve over the course of training?

The study replicates and extends Houghton, Sagae, & Morgan (2025, ACL) using models trained from scratch (rather than pre-trained LLMs), enabling full access to the training data seen at every checkpoint and allowing a direct examination of learning trajectories.

---

## Repository Structure

```
hand-trained-model-prefs/
│
├── README.md
├── model_training_card.md          # Detailed model training documentation
│
├── model-training-code/            # Scripts for training the six models
│   ├── train_autoreg.py            # Main HuggingFace Trainer training script
│   ├── tokenizer_and_config.py     # BPE tokenizer training and model config setup
│   ├── tokenization.py             # Tokenization utilities
│   ├── train_all_opt_babylm.sh     # Shell script: trains all three BabyLM models
│   ├── train_all_opt_c4.sh         # Shell script: trains all three C4 models
│   ├── retro_tag_all_models.py     # Tags checkpoints on HuggingFace Hub by step
│   ├── upload-babylm.py            # Uploads BabyLM dataset to Hub
│   └── create_config_only.py       # Creates model config without training
│
├── analysis-scripts/               # Data collection and analysis pipeline
│   ├── model-prefs-all-ckpts.py    # Step 1: Score all checkpoints on all binomials
│   ├── count_corpus_exposures.py   # Step 2: Replay training data, count exposures
│   ├── build_checkpoint_exposures.py # Step 3: Join scores with exposure counts
│   ├── prepare_data.py             # Step 4: Aggregate into R-ready CSVs
│   ├── evaluate_training_quality.py # Training quality (perplexity, loss curves)
│   ├── prepare_results.R           # Extracts brms model results into CSVs
│   ├── analysis.Rmd                # Main analysis document
│   └── analysis_cleaned_up.Rmd     # Cleaned-up version for sharing
│
├── Data/
│   ├── nonce_and_attested_binoms.csv    # Binomial stimuli (nonce + attested)
│   ├── all_human_data.csv               # Human response data + corpus counts + constraint scores
│   ├── binomial_corpus_counts.csv       # Per-binomial corpus occurrence counts
│   ├── babylm_step_exposures.csv        # Cumulative exposure counts per step (BabyLM)
│   ├── c4_step_exposures.csv            # Cumulative exposure counts per step (C4)
│   ├── checkpoint_results/              # Raw per-checkpoint CSV files (one per model×checkpoint)
│   ├── processed/                       # Aggregated CSVs used by R (committed to repo)
│   │   └── checkpoint_results_with_exposures.csv  # ⚠ gitignored (~30 GB); regenerate locally
│   └── training_quality/                # Perplexity results and loss curves
│
└── Writeup/
    ├── writeup.qmd                 # Quarto paper (CoLM 2026 format)
    ├── references.bib              # Bibliography (Zotero/BetterBibTeX)
    ├── r-references.bib            # Auto-generated R package citations
    ├── colm2026_conference.sty     # CoLM style file
    └── _extensions/colm/           # Quarto extension for CoLM template
```

---

## Models

Six OPT-architecture decoder-only language models were trained from scratch across two corpora and three parameter scales. All models share a custom BPE tokenizer (vocabulary size 8,192; context window 1,024 tokens) trained on the full BabyLM corpus.

| Corpus | Parameters | HuggingFace ID | Training tokens | Checkpoints |
|--------|-----------|----------------|-----------------|-------------|
| BabyLM | 125M | `znhoughton/opt-babylm-125m-64eps-seed964` | ~9.1B (64 epochs) | ~912 |
| BabyLM | 350M | `znhoughton/opt-babylm-350m-64eps-seed964` | ~9.1B (64 epochs) | ~912 |
| BabyLM | 1.3B | `znhoughton/opt-babylm-1.3b-64eps-seed964` | ~9.1B (64 epochs) | ~860 |
| C4     | 125M | `znhoughton/opt-c4-125m-seed964`           | ~10B (1 epoch)    | ~1,014 |
| C4     | 350M | `znhoughton/opt-c4-350m-seed964`           | ~10B (1 epoch)    | ~1,005 |
| C4     | 1.3B | `znhoughton/opt-c4-1.3b-seed964`           | ~10B (1 epoch)    | ~960 |

Checkpoints were saved approximately every 10 million training tokens and pushed to HuggingFace Hub throughout training. See [`model_training_card.md`](model_training_card.md) for full architecture, hyperparameter, and reproducibility details.

**Corpora:**
- **BabyLM** (`znhoughton/babylm-150m-v3`): ~150M tokens of child-directed speech, children's books, and simple web text (Warstadt et al., 2023). Trained for 64 epochs.
- **C4** (`znhoughton/c4-subset-10B-tokens`): ~10B tokens of filtered web text, a subset of the Colossal Clean Crawled Corpus (Raffel et al., 2020). Trained for 1 epoch.

---

## Key Concepts

### Preference

For each binomial (e.g., *bread and butter* / *butter and bread*), the model's **preference** is the difference in log-probability between the alphabetically ordered form and the non-alphabetically ordered form:

```
preference = log P(alpha order) − log P(non-alpha order)
```

Positive values indicate the model prefers the alphabetically ordered form; negative values indicate preference for the reverse. Scores are averaged across 50 diverse prompt prefixes to reduce context sensitivity.

### RelFreq (Relative Frequency)

**RelFreq** is the proportion of corpus occurrences in the alphabetically ordered form, centered:

```
RelFreq = P(alpha) − 0.5,  where  P(alpha) = count(alpha) / (count(alpha) + count(non-alpha))
```

This is a fixed property of each binomial derived from the full training corpus. It is constant across checkpoints — it does not reflect how much of each ordering the model had seen up to a given training stage.

### AbsPref (Abstract Preference)

**AbsPref** is the fitted value from a logistic regression of human ordering preferences on phonological and semantic constraints (Morgan & Levy, 2016): stress, iconicity, formality, cultural salience, etc. It captures the degree to which abstract structural biases favour one ordering, independently of corpus frequency.

### Preference Deviation

**Preference deviation** is the difference between a model's preference score and the log relative frequency ratio:

```
preference deviation = preference − ln(count(alpha) / count(non-alpha))
```

A value of zero means the model's preference exactly matches what corpus frequency would predict. Positive values indicate a stronger preference for the alpha-ordered form than corpus statistics alone would suggest. This measure allows us to ask whether models develop preferences beyond simple frequency memorisation.

---

## Analysis Pipeline

The pipeline runs in five sequential steps. Steps 1–4 are Python; step 5 is R.

### Step 1 — Score checkpoints (`model-prefs-all-ckpts.py`)

Downloads each checkpoint from HuggingFace Hub and scores every binomial pair under every prompt prefix, computing `alpha_logprob` and `nonalpha_logprob`. Outputs one CSV per model×checkpoint to `Data/checkpoint_results/`.

- Multi-GPU (2× H100), OOM-safe with adaptive batch sizing
- Resume-safe: skips fully scored checkpoints, re-runs incomplete ones
- ~5,600 checkpoint files × ~36,975 rows each ≈ 209 million rows total

### Step 2 — Count corpus exposures (`count_corpus_exposures.py`)

Replays the training data in exact training order to compute cumulative corpus exposure counts (`alpha_seen`, `beta_seen`) for each binomial at each checkpoint. This is a full replay of ~9–10B tokens per corpus, using the same data pipeline as training.

Output: `Data/babylm_step_exposures.csv` and `Data/c4_step_exposures.csv`.

### Step 3 — Join scores and exposures (`build_checkpoint_exposures.py`)

Merges the raw model preference data (Step 1) with the cumulative exposure counts (Step 2). Output: `Data/processed/checkpoint_results_with_exposures.csv` (~209M rows, ~30 GB).

> **Note:** This file is gitignored due to its size. Run this step locally to regenerate it. All downstream R analyses that do not require per-checkpoint token trajectory axes can be run without it, using the smaller committed CSVs from Step 4.

### Step 4 — Aggregate for R (`prepare_data.py`)

Aggregates the large joined CSV into smaller R-readable CSVs in `Data/processed/`:

| Output file | Contents |
|-------------|----------|
| `attested_agg.csv` | Prompt-averaged preference per binomial × checkpoint (attested binomials only) |
| `nonce_agg.csv` | Same for nonce binomials |
| `final_attested.csv` | Final checkpoint only — used for brms cross-sectional models |
| `final_nonce.csv` | Same for nonce binomials |
| `training_attested.csv` | 20 log-sampled checkpoints — used for brms trajectory models |
| `training_nonce.csv` | Same for nonce binomials |
| `attested_correlations.csv` | Pearson r(preference, human pref) at every checkpoint |
| `attested_accuracy.csv` | Ordering accuracy at every checkpoint |

These files **are** committed to the repo; collaborators who only want to run R analyses do not need Steps 1–3.

### Step 5 — Fit models and extract results (R)

Bayesian mixed-effects regression models are fit in R using the `brms` package. Three analyses are conducted on the attested binomials:

1. **Preference model**: Predicts raw preference from AbsPref, RelFreq, log overall frequency, and their interactions — at the final checkpoint and at 20 log-sampled training checkpoints.
2. **Preference deviation model**: Same predictors, with preference deviation as the outcome — tests whether preferences go beyond corpus frequency predictions.
3. **Absolute preference deviation model**: Predicts the mean absolute preference deviation from training stage × log overall frequency — tracks how corpus-likeness evolves over training.

Results are extracted from the fitted `brm` objects via `prepare_results.R`, which saves coefficient tables and summary statistics to `Data/processed/results/` for use in the Quarto writeup.

---

## Writeup

The paper is written in Quarto (`Writeup/writeup.qmd`) targeting the CoLM 2026 format. To render:

```bash
cd Writeup
quarto render writeup.qmd
```

**Requirements:**
- Quarto ≥ 1.4
- R with packages: `brms`, `tidyverse`, `knitr`, `patchwork`
- LaTeX distribution (for PDF output)
- Fitted brms model `.rds` files in `analysis-scripts/models/`
- Processed CSVs in `Data/processed/results/` (from `prepare_results.R`)

The bibliography is managed via Zotero + BetterBibTeX (`references.bib`). R package citations are auto-generated into `r-references.bib` at render time via `knitr::write_bib()`.

---

## Reproducing from Scratch

| Goal | What you need |
|------|---------------|
| Read results | Open `analysis-scripts/analysis_cleaned_up.html` in a browser |
| Re-render the Quarto paper | R, Quarto, LaTeX, fitted `.rds` files, processed CSVs |
| Re-run R analyses | R + brms + processed CSVs from `Data/processed/` (committed) |
| Regenerate processed CSVs | Python ≥ 3.10, `pandas`, `numpy`, `scipy`; run `prepare_data.py` |
| Re-score all checkpoints | 2× H100 GPUs, HuggingFace Hub access; run `model-prefs-all-ckpts.py` |
| Retrain models from scratch | 2× H100 GPUs, HuggingFace Hub account; run `train_all_opt_babylm.sh` and `train_all_opt_c4.sh` |

### Python dependencies

```bash
pip install torch transformers datasets pandas numpy scipy tqdm huggingface_hub
```

### R dependencies

```r
install.packages(c("brms", "tidyverse", "knitr", "patchwork"))
```

---

## Data Files

| File | Description |
|------|-------------|
| `Data/nonce_and_attested_binoms.csv` | Full binomial stimulus list with `Word1`, `Word2`, and attested/nonce label |
| `Data/all_human_data.csv` | Human preference responses, corpus counts (`OverallFreq`, `RelFreq`), and constraint scores (`Form`, `Percept`, `Culture`, `Power`, `Intense`, `Icon`, `Freq`, `Len`, `Lapse`, `*BStress`) from Morgan & Levy (2016) |
| `Data/binomial_corpus_counts.csv` | Raw alpha/non-alpha counts per binomial in each corpus |
| `Data/babylm_step_exposures.csv` | Cumulative alpha/non-alpha exposures per binomial at each BabyLM training step |
| `Data/c4_step_exposures.csv` | Same for C4 |
| `Data/training_quality/` | Perplexity results and training loss curves for all six models |

---

## References

- Houghton, Z. N., Sagae, K., & Morgan, E. (2025). The role of abstract representations and observed preferences in the ordering of binomials in large language models. *Proceedings of ACL 2025*.
- Morgan, E., & Levy, R. (2016). Abstract knowledge versus direct experience in processing of binomial expressions. *Cognition*, 157, 384–402.
- Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1–67.
- Warstadt, A., et al. (2023). Findings of the BabyLM challenge. *Proceedings of the BabyLM Challenge at CoNLL 2023*.
- Zhang, S., et al. (2022). OPT: Open pre-trained transformer language models. *arXiv:2205.01068*.
