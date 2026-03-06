# Methods

## Model Training

We trained six language models spanning two corpora and three parameter scales
using the OPT architecture (Zhang et al., 2022). All models used a shared BPE
tokenizer with a vocabulary of 8,192 tokens and a context window of 1,024 tokens,
trained on the BabyLM corpus.

### Corpora

**BabyLM corpus.** Three models were trained on the BabyLM corpus
(`znhoughton/babylm-150m-v3`; Warstadt et al., 2023), a developmentally
plausible collection of approximately 150 million tokens drawn from child-directed
speech, children's books, and simple web text. To approximate a full-training
token budget comparable to the C4 models, each model was trained for 64 epochs,
yielding approximately 9.6 billion total tokens seen.

**C4 corpus.** Three matched models were trained on a 10-billion-token subset of
the Colossal Clean Crawled Corpus (`znhoughton/c4-subset-10B-tokens`; Raffel
et al., 2020), representing large-scale naturalistic web text.

### Architecture and Hyperparameters

For each corpus, we trained models at three scales following the standard OPT
model family configurations:

| Model | Parameters | Hidden size | Heads | Layers | FFN size |
|-------|-----------|-------------|-------|--------|----------|
| OPT-125M | 125M | 768 | 12 | 12 | 3,072 |
| OPT-350M | 350M | 1,024 | 16 | 24 | 4,096 |
| OPT-1.3B | 1.3B | 2,048 | 32 | 24 | 8,192 |

All models were trained with bfloat16 precision and gradient checkpointing on
two NVIDIA H100 GPUs using PyTorch DDP (`torchrun --nproc_per_node=2`). Additional
shared hyperparameters: learning rate warmup over 4,000 steps, random seed 964.

Per-model batch configuration and effective tokens per gradient step:

| Model scale | Per-device batch | Grad. accum. steps | Learning rate | Tokens/step |
|-------------|-----------------|-------------------|---------------|-------------|
| 125M | 400 | 1 | 3 × 10⁻⁴ | 819,200 |
| 350M | 300 | 2 | 1 × 10⁻⁴ | 1,228,800 |
| 1.3B | 100 | 4 | 1 × 10⁻⁴ | 819,200 |

*(Tokens/step = block\_size × per\_device\_batch × grad\_accum\_steps × n\_GPUs)*

### Checkpoint Cadence

Model weights were saved every 10 million tokens throughout training. This
produced between 860 and 1,014 checkpoints per model across the full training
run (see per-model details in the checkpoint tables below).

---

## Stimuli

Stimuli were English binomial expressions of the form *X and Y* (e.g., *salt and
pepper*, *bread and butter*), drawn from a larger norming study. The full set
comprised both **attested** binomials (those observed with non-zero corpus
frequency) and **nonce** binomials (Overall Frequency = 0), which were used to
isolate gradient grammatical sensitivity from lexical memorization.

---

## Analysis

### Model Preference Score

At each checkpoint, each model was queried with multiple prompt templates for
every binomial. The model's preference for the *alpha* ordering (Word1 before
Word2) was computed as the proportion of prompts on which the model assigned
higher probability to the alpha form. This raw preference score was then averaged
across prompt variants to yield a single **preference** value per binomial per
checkpoint.

### Analyses Run at Every Checkpoint

The following analyses were computed at every saved checkpoint for all six models
(860–1,014 checkpoints per model):

1. **Pearson correlation** between model preference and mean human preference
   (averaged across participants per binomial), computed separately for nonce and
   attested binomials.

2. **Accuracy** — the proportion of binomials on which the model's binary
   preference (preference > 0.5 = alpha ordering preferred) matched the majority
   human binary preference, computed separately for nonce and attested binomials.

### Analyses Run at ~20 Log-Sampled Checkpoints

To reduce computational cost for the regression-based analyses, for each model we
selected approximately 20 checkpoints sampled at logarithmically equal spacing
across the full set of checkpoints. This log-spacing ensures dense coverage of
early training (where preferences change most rapidly) while still capturing the
later plateau. The sampling formula was:

```
indices = unique(round(exp(linspace(log(1), log(n), 20))) − 1)
```

where *n* is the total number of checkpoints for that model. This typically
yielded 19 distinct checkpoints (some indices collide after rounding).

The following analyses were computed at these ~20 sampled checkpoints:

3. **Constraint regression** — for each checkpoint, an OLS regression of model
   preference on ten linguistic constraint scores: Form (phonotactics), Percept
   (perceptual salience), Culture (cultural primacy), Power (power/status),
   Intense (word intensity), Icon (iconicity), Freq (lexical frequency), Len
   (word length), Lapse (temporal lapse), and BStress (final stress). Regression
   coefficients, standard errors, and p-values were retained for each constraint
   at each sampled checkpoint.

4. **GenPref regression** — for each checkpoint, a simple linear regression of
   model preference on GenPref, a composite human-preference score derived from a
   logistic model fit to the human norming data:

   GenPref = logistic(0.022 + 0.239·Form + 0.249·Percept + 0.418·Culture
             + 0.260·Power + 0.019·Intense + 1.304·Icon + 0.086·Freq
             + 0.152·Len − 0.194·Lapse + 0.360·BStress) − 0.5

   The regression slope on GenPref captures the degree to which the model's
   preferences are explained by the composite human constraint weighting.

### Final-Checkpoint Bayesian Models

At the final checkpoint of each model (the last saved checkpoint), Bayesian
mixed-effects regression models were fit using `brms` (Bürkner, 2017) in R.
Separate models were fit for nonce and attested binomials, with model preference
as the outcome and GenPref (plus, for attested binomials, log corpus frequency
and relative frequency) as predictors. These models included random intercepts
and slopes by binomial and by model.

---

## Sampled Checkpoints by Model

The tables below list the exact checkpoints selected by the log-sampling procedure
for the trajectory analyses (analyses 3 and 4 above). Token counts are taken
directly from the checkpoint metadata.

### opt-babylm-125m-64eps-seed964 — 912 total checkpoints

| # | Training step | Tokens seen |
|---|--------------|-------------|
| 1 | 12 | 9,830,400 |
| 2 | 24 | 19,660,800 |
| 3 | 36 | 29,491,200 |
| 4 | 48 | 39,321,600 |
| 5 | 72 | 58,982,400 |
| 6 | 108 | 88,473,600 |
| 7 | 144 | 117,964,800 |
| 8 | 216 | 176,947,200 |
| 9 | 300 | 245,760,000 |
| 10 | 432 | 353,894,400 |
| 11 | 624 | 511,180,800 |
| 12 | 888 | 727,449,600 |
| 13 | 1,272 | 1,042,022,400 |
| 14 | 1,824 | 1,494,220,800 |
| 15 | 2,604 | 2,133,196,800 |
| 16 | 3,732 | 3,057,254,400 |
| 17 | 5,340 | 4,374,528,000 |
| 18 | 7,644 | 6,261,964,800 |
| 19 | 10,944 | 8,965,324,800 |

### opt-babylm-350m-64eps-seed964 — 912 total checkpoints

| # | Training step | Tokens seen |
|---|--------------|-------------|
| 1 | 8 | 9,830,400 |
| 2 | 16 | 19,660,800 |
| 3 | 24 | 29,491,200 |
| 4 | 32 | 39,321,600 |
| 5 | 48 | 58,982,400 |
| 6 | 72 | 88,473,600 |
| 7 | 96 | 117,964,800 |
| 8 | 144 | 176,947,200 |
| 9 | 200 | 245,760,000 |
| 10 | 288 | 353,894,400 |
| 11 | 416 | 511,180,800 |
| 12 | 592 | 727,449,600 |
| 13 | 848 | 1,042,022,400 |
| 14 | 1,216 | 1,494,220,800 |
| 15 | 1,736 | 2,133,196,800 |
| 16 | 2,488 | 3,057,254,400 |
| 17 | 3,560 | 4,374,528,000 |
| 18 | 5,096 | 6,261,964,800 |
| 19 | 7,296 | 8,965,324,800 |

### opt-babylm-1.3b-64eps-seed964 — 860 total checkpoints

| # | Training step | Tokens seen |
|---|--------------|-------------|
| 1 | 12 | 9,830,400 |
| 2 | 24 | 19,660,800 |
| 3 | 36 | 29,491,200 |
| 4 | 48 | 39,321,600 |
| 5 | 72 | 58,982,400 |
| 6 | 96 | 78,643,200 |
| 7 | 144 | 117,964,800 |
| 8 | 204 | 167,116,800 |
| 9 | 300 | 245,760,000 |
| 10 | 420 | 344,064,000 |
| 11 | 600 | 491,520,000 |
| 12 | 852 | 697,958,400 |
| 13 | 1,224 | 1,002,700,800 |
| 14 | 2,208 | 1,808,793,600 |
| 15 | 2,952 | 2,418,278,400 |
| 16 | 4,020 | 3,293,184,000 |
| 17 | 5,532 | 4,531,814,400 |
| 18 | 7,728 | 6,330,777,600 |
| 19 | 10,944 | 8,965,324,800 |

### opt-c4-125m-seed964 — 1,014 total checkpoints

| # | Training step | Tokens seen |
|---|--------------|-------------|
| 1 | 36 | 29,491,200 |
| 2 | 48 | 39,321,600 |
| 3 | 60 | 49,152,000 |
| 4 | 72 | 58,982,400 |
| 5 | 96 | 78,643,200 |
| 6 | 132 | 108,134,400 |
| 7 | 180 | 147,456,000 |
| 8 | 240 | 196,608,000 |
| 9 | 348 | 285,081,600 |
| 10 | 480 | 393,216,000 |
| 11 | 684 | 560,332,800 |
| 12 | 972 | 796,262,400 |
| 13 | 1,392 | 1,140,326,400 |
| 14 | 1,992 | 1,631,846,400 |
| 15 | 2,856 | 2,339,635,200 |
| 16 | 4,104 | 3,361,996,800 |
| 17 | 5,892 | 4,826,726,400 |
| 18 | 8,472 | 6,940,262,400 |
| 19 | 12,192 | 9,987,686,400 |

### opt-c4-350m-seed964 — 1,005 total checkpoints

| # | Training step | Tokens seen |
|---|--------------|-------------|
| 1 | 16 | 19,660,800 |
| 2 | 24 | 29,491,200 |
| 3 | 32 | 39,321,600 |
| 4 | 40 | 49,152,000 |
| 5 | 56 | 68,812,800 |
| 6 | 80 | 98,304,000 |
| 7 | 112 | 137,625,600 |
| 8 | 152 | 186,777,600 |
| 9 | 216 | 265,420,800 |
| 10 | 312 | 383,385,600 |
| 11 | 448 | 550,502,400 |
| 12 | 640 | 786,432,000 |
| 13 | 912 | 1,120,665,600 |
| 14 | 1,320 | 1,622,016,000 |
| 15 | 1,888 | 2,319,974,400 |
| 16 | 2,720 | 3,342,336,000 |
| 17 | 3,920 | 4,816,896,000 |
| 18 | 5,672 | 6,969,753,600 |
| 19 | 8,128 | 9,987,686,400 |

### opt-c4-1.3b-seed964 — 960 total checkpoints

| # | Training step | Tokens seen |
|---|--------------|-------------|
| 1 | 60 | 49,152,000 |
| 2 | 72 | 58,982,400 |
| 3 | 84 | 68,812,800 |
| 4 | 96 | 78,643,200 |
| 5 | 120 | 98,304,000 |
| 6 | 156 | 127,795,200 |
| 7 | 204 | 167,116,800 |
| 8 | 264 | 216,268,800 |
| 9 | 360 | 294,912,000 |
| 10 | 492 | 403,046,400 |
| 11 | 684 | 560,332,800 |
| 12 | 960 | 786,432,000 |
| 13 | 1,368 | 1,120,665,600 |
| 14 | 1,944 | 1,592,524,800 |
| 15 | 2,760 | 2,260,992,000 |
| 16 | 3,948 | 3,234,201,600 |
| 17 | 5,640 | 4,620,288,000 |
| 18 | 8,076 | 6,615,859,200 |
| 19 | 12,192 | 9,987,686,400 |

---

## References

Bürkner, P.-C. (2017). brms: An R package for Bayesian multilevel models using
Stan. *Journal of Statistical Software*, 80(1), 1–28.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... &
Liu, P. J. (2020). Exploring the limits of transfer learning with a unified
text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1–67.

Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., ... &
Williams, A. (2023). Findings of the BabyLM challenge: Sample-efficient
pretraining on a developmentally plausible corpus. In *Proceedings of the
BabyLM Challenge at CoNLL 2023*.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., ... &
Zettlemoyer, L. (2022). OPT: Open pre-trained transformer language models.
*arXiv preprint arXiv:2205.01068*.
