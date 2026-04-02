#!/usr/bin/env Rscript
# reading_time_delta_ll.R
# =======================
# Replication of Oh & Schuler (2023, TACL) Figure 1 — Natural Stories SPR panel
# extended with BabyLM and C4 models.
#
# ── Methodology (Section 3.3 of the paper) ────────────────────────────────────
# Response variable : log(RT)   [natural log, as in lme4 default]
# Baseline predictors: word_length (nchar), word_position (zone within story)
#                      — both centred and scaled (z-scored)
# Random effects    : (1 + word_length_s + word_position_s | subject)   ← by-subject
#                   + (1 | word_type)                                    ← by-word-type
#                   + (1 | subject:item)                                 ← subject×sentence
#   "The LME models included by-subject random slopes for all fixed effects
#    as well as random intercepts for each subject and each word type.
#    Additionally, for self-paced reading times … a random intercept for each
#    subject-sentence interaction was included."  (Oh & Schuler 2023, p.339)
# Full model: baseline + surprisal_s (also with by-subject random slope)
# ΔLL = logLik(full) − logLik(baseline)   [REML = FALSE throughout]
# Package: lme4::lmer()
#
# ── Filtering (Section 3.1) ───────────────────────────────────────────────────
# - Remove subjects with correct ≤ 3 comprehension answers
# - Remove sentence-initial (zone == 1) and sentence-final (zone == max per item) words
# - Remove RTs < 100 ms or > 3000 ms
# - All observations log-transformed before fitting
#
# ── Split (Section 3.1 footnote 3) ──────────────────────────────────────────
# Partition based on (subject_num + item) %% 2:
#   each subject-by-sentence (item) combination stays intact in one partition.
#   Exploratory set = remainder 0; held-out set = remainder 1.
# We fit only on the exploratory set.
#
# ── Inputs ────────────────────────────────────────────────────────────────────
# natural_stories/processed_RTs.tsv   Natural Stories reading times
#   Columns: WorkerId  WorkTimeInSeconds  correct  item  zone  RT  word  ...
#   Download: github.com/languageMIT/naturalstories → naturalstories_RTs/
# ns_surprisal/<model>.csv            per-word surprisal from get_ns_surprisal.py
# ns_surprisal/ns_perplexity.csv      corpus perplexity per model
#
# ── Outputs ───────────────────────────────────────────────────────────────────
# ns_delta_ll.csv          ΔLL + perplexity per model
# ns_delta_ll_plot.pdf/png Figure 1 replication

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR <- dirname(rstudioapi::getSourceEditorContext()$path)
NS_DIR     <- file.path(SCRIPT_DIR, "natural_stories")
SURP_DIR   <- file.path(SCRIPT_DIR, "ns_surprisal")
PPL_CSV    <- file.path(SURP_DIR, "ns_perplexity.csv")
RT_TSV     <- file.path(NS_DIR, "processed_RTs.tsv")
OUT_CSV    <- file.path(SCRIPT_DIR, "ns_delta_ll.csv")
OUT_PDF    <- file.path(SCRIPT_DIR, "ns_delta_ll_plot.pdf")
OUT_PNG    <- file.path(SCRIPT_DIR, "ns_delta_ll_plot.png")

# ── Check inputs ──────────────────────────────────────────────────────────────
if (!file.exists(RT_TSV)) {
  stop(
    "Reading time file not found: ", RT_TSV, "\n",
    "Download processed_RTs.tsv from:\n",
    "  https://github.com/languageMIT/naturalstories/tree/master/naturalstories_RTs\n",
    "and place it in: ", NS_DIR
  )
}
if (!file.exists(PPL_CSV)) {
  stop("Perplexity file not found: ", PPL_CSV,
       "\nRun get_ns_surprisal.py first.")
}

# ── Load reading times ────────────────────────────────────────────────────────
message("Loading reading times …")

rt_raw <- read_tsv(RT_TSV, show_col_types = FALSE)
message(sprintf("  Raw columns: %s", paste(names(rt_raw), collapse = ", ")))

# Normalise to canonical names
rt_raw <- rt_raw |>
  rename(
    subject = WorkerId,
    RT      = RT,
    item    = item,
    zone    = zone,
    word    = word,
    correct = correct
  )

message(sprintf("  Raw: %d rows, %d subjects",
                nrow(rt_raw), n_distinct(rt_raw$subject)))

# ── Filter subjects: remove those who answered ≤3 comprehension Qs correctly ─
# 'correct' is a per-subject count repeated on every row for that subject.
good_subjects <- rt_raw |>
  distinct(subject, correct) |>
  filter(correct > 3) |>
  pull(subject)

message(sprintf("  Subjects with correct > 3: %d of %d",
                length(good_subjects), n_distinct(rt_raw$subject)))

# ── Build analysis dataset ────────────────────────────────────────────────────
rt <- rt_raw |>
  filter(subject %in% good_subjects,
         !is.na(RT), !is.na(zone), !is.na(word)) |>
  mutate(
    RT      = as.numeric(RT),
    subject = as.character(subject),
    item    = as.integer(item),
    zone    = as.integer(zone)
  )

# Word-level covariates (Oh & Schuler 2023: "word length measured in characters
# and index of word position within each sentence")
rt <- rt |>
  mutate(
    word_length   = nchar(as.character(word)),
    word_position = zone          # position within story (proxy for sentence position)
  )

# Remove sentence-initial (zone 1) and sentence-final (last zone per item) words,
# and RTs outside [100, 3000] ms
last_zones <- rt |>
  group_by(item) |>
  summarise(last_zone = max(zone), .groups = "drop")

rt <- rt |>
  left_join(last_zones, by = "item") |>
  filter(
    zone != 1,
    zone != last_zone,
    RT >= 100,
    RT <= 3000
  ) |>
  select(-last_zone)

message(sprintf("  After filtering: %d rows, %d subjects",
                nrow(rt), n_distinct(rt$subject)))

# Log-transform RT (natural log, as is lme4 convention and what the paper uses)
rt <- rt |> mutate(log_rt = log(RT))

# ── Exploratory / held-out split (Section 3.1, footnote 3) ──────────────────
# "partitioning was conducted based on the sum of subject ID and sentence ID,
#  resulting in each subject-by-sentence combination remaining intact"
# Natural Stories has no sentence boundaries in the RT file; item (story) is
# used as the sentence unit for this split.
rt <- rt |>
  mutate(
    subject_num = as.integer(factor(subject)),
    split_key   = (subject_num + item) %% 2,
    set         = if_else(split_key == 0L, "exploratory", "held_out")
  )

n_exp  <- sum(rt$set == "exploratory")
n_held <- sum(rt$set == "held_out")
message(sprintf("  Exploratory: %d rows  |  Held-out: %d rows", n_exp, n_held))

exploratory <- filter(rt, set == "exploratory")

# ── Centre and scale predictors (z-score, applied to exploratory set) ────────
# "All predictors were centered and scaled prior to model fitting" (p.339)
z <- function(x) as.numeric(scale(x))

exploratory <- exploratory |>
  mutate(
    word_length_s   = z(word_length),
    word_position_s = z(word_position),
    # word_type for by-word-type random intercept
    word_type        = tolower(as.character(word)),
    # subject:item interaction for the subject-sentence random intercept
    subj_item        = paste(subject, item, sep = ":")
  )

# ── Fit baseline model ────────────────────────────────────────────────────────
# "by-subject random slopes for all fixed effects as well as random intercepts
#  for each subject and each word type. Additionally, for self-paced reading
#  times … a random intercept for each subject-sentence interaction"
#
# lmer notation:
#   (1 + word_length_s + word_position_s | subject)  → by-subject intercept +
#                                                        slopes for both predictors
#   (1 | word_type)                                  → by-word-type intercept
#   (1 | subj_item)                                  → subject × sentence intercept
message("\nFitting baseline LME …")

baseline_formula <- log_rt ~
  word_length_s + word_position_s +
  (1 + word_length_s + word_position_s | subject) +
  (1 | word_type) +
  (1 | subj_item)

baseline_mod <- lmer(
  baseline_formula,
  data    = exploratory,
  REML    = FALSE,
  control = lmerControl(optimizer = "bobyqa",
                        optCtrl   = list(maxfun = 2e5))
)
baseline_ll <- as.numeric(logLik(baseline_mod))
message(sprintf("  Baseline log-likelihood: %.4f", baseline_ll))

# ── Load perplexity table ─────────────────────────────────────────────────────
ppl_df <- read_csv(PPL_CSV, show_col_types = FALSE)

# Build reverse-lookup: safe_name → model_id
# get_ns_surprisal.py uses model_id.replace("/", "_") as safe_name
safe_to_model <- setNames(
  ppl_df$model,
  gsub("/", "_", ppl_df$model)
)

# ── Compute ΔLL for each surprisal file ───────────────────────────────────────
surp_files <- list.files(SURP_DIR, pattern = "\\.csv$", full.names = TRUE)
surp_files <- surp_files[basename(surp_files) != "ns_perplexity.csv"]

results <- vector("list", length(surp_files))

for (i in seq_along(surp_files)) {
  csv_path  <- surp_files[[i]]
  safe_name <- tools::file_path_sans_ext(basename(csv_path))
  model_id  <- safe_to_model[safe_name]

  if (is.na(model_id)) {
    message(sprintf("  [SKIP] %s — not found in perplexity table", safe_name))
    next
  }

  ppl_row <- filter(ppl_df, model == model_id)
  message(sprintf("\n[%d/%d] %s  (ppl = %.2f)",
                  i, length(surp_files), model_id, ppl_row$perplexity[1]))

  # Load surprisal
  surp <- tryCatch(
    read_csv(csv_path, show_col_types = FALSE),
    error = function(e) { message("  ERROR reading CSV: ", e$message); NULL }
  )
  if (is.null(surp) || nrow(surp) == 0) next

  # Join surprisal onto the exploratory set (match by item + zone)
  d <- exploratory |>
    left_join(surp |> select(item, zone, surprisal),
              by = c("item", "zone")) |>
    filter(!is.na(surprisal), is.finite(surprisal))

  if (nrow(d) < 500) {
    message(sprintf("  [SKIP] Only %d matched rows after join — check item/zone alignment",
                    nrow(d)))
    next
  }
  message(sprintf("  Matched rows: %d", nrow(d)))

  # Scale surprisal (z-score within the exploratory set, same as other predictors)
  d <- d |> mutate(surprisal_s = z(surprisal))

  # Full model: add surprisal as fixed effect and by-subject random slope.
  # The paper says "by-subject random slopes for ALL fixed effects", so surprisal_s
  # is added to the existing by-subject grouping.
  full_formula <- log_rt ~
    word_length_s + word_position_s + surprisal_s +
    (1 + word_length_s + word_position_s + surprisal_s | subject) +
    (1 | word_type) +
    (1 | subj_item)

  full_mod <- tryCatch(
    lmer(full_formula, data = d, REML = FALSE,
         control = lmerControl(optimizer = "bobyqa",
                               optCtrl   = list(maxfun = 2e5))),
    error = function(e) {
      message("  lmer ERROR: ", e$message)
      NULL
    },
    warning = function(w) {
      message("  lmer WARNING (proceeding): ", w$message)
      suppressWarnings(
        lmer(full_formula, data = d, REML = FALSE,
             control = lmerControl(optimizer = "bobyqa",
                                   optCtrl   = list(maxfun = 2e5)))
      )
    }
  )
  if (is.null(full_mod)) next

  full_ll  <- as.numeric(logLik(full_mod))
  delta_ll <- full_ll - baseline_ll

  message(sprintf("  ΔLL = %.4f", delta_ll))

  results[[i]] <- tibble(
    model      = model_id,
    family     = ppl_row$family[1],
    params     = ppl_row$params[1],
    perplexity = ppl_row$perplexity[1],
    delta_ll   = delta_ll
  )
}

results_df <- bind_rows(results)
write_csv(results_df, OUT_CSV)
message(sprintf("\nΔLL results saved → %s", OUT_CSV))
print(results_df |> arrange(family, perplexity), n = Inf)

# ── Plot ──────────────────────────────────────────────────────────────────────
message("\nPlotting …")

FAMILY_LEVELS  <- c("GPT-2", "GPT-Neo", "OPT",
                    "BabyLM", "BabyLM (early)", "BabyLM (mid)",
                    "C4",     "C4 (early)",     "C4 (mid)")
FAMILY_COLOURS <- c(
  "GPT-2"          = "#2166ac",
  "GPT-Neo"        = "#4dac26",
  "OPT"            = "#b2182b",
  "BabyLM"         = "#238443",
  "BabyLM (early)" = "#d9f0a3",
  "BabyLM (mid)"   = "#78c679",
  "C4"             = "#980043",
  "C4 (early)"     = "#d4b9da",
  "C4 (mid)"       = "#df65b0"
)
FAMILY_SHAPES <- c(
  "GPT-2"          = 16,
  "GPT-Neo"        = 15,
  "OPT"            = 17,
  "BabyLM"         = 18,
  "BabyLM (early)" = 1,
  "BabyLM (mid)"   = 19,
  "C4"             = 8,
  "C4 (early)"     = 2,
  "C4 (mid)"       = 6
)

plot_df <- results_df |>
  filter(!is.na(delta_ll), !is.na(perplexity)) |>
  mutate(
    family     = factor(family, levels = FAMILY_LEVELS),
    params_num = as.numeric(gsub("M$", "", params))
  ) |>
  arrange(family, perplexity)

# y-axis limits: pad 10% above and below the data range
y_range  <- range(plot_df$delta_ll, na.rm = TRUE)
y_pad    <- diff(y_range) * 0.1
y_limits <- c(y_range[1] - y_pad, y_range[2] + y_pad)

p <- ggplot(plot_df,
            aes(x = perplexity, y = delta_ll, colour = family, shape = family)) +
  # Dashed lines connecting models within each family (sorted by perplexity)
  geom_line(
    aes(group = family),
    linetype = "dashed", linewidth = 0.7, alpha = 0.7
  ) +
  geom_point(size = 3, alpha = 0.9) +
  scale_x_continuous(
    "Perplexity",
    trans  = "log2",
    labels = scales::trans_format("log2", scales::math_format(2^.x))
  ) +
  scale_y_continuous("\u0394LL", limits = y_limits) +
  scale_colour_manual(values = FAMILY_COLOURS, name = "Model family") +
  scale_shape_manual(values  = FAMILY_SHAPES,  name = "Model family") +
  ggtitle("Natural Stories SPR") +
  theme_classic(base_size = 13) +
  theme(
    panel.grid.major = element_line(colour = "grey92", linewidth = 0.4),
    axis.line        = element_line(colour = "grey40"),
    axis.ticks       = element_line(colour = "grey40"),
    legend.position  = "right",
    plot.title       = element_text(face = "bold", hjust = 0.5)
  )

print(p)
ggsave(OUT_PDF, p, width = 7, height = 5)
ggsave(OUT_PNG, p, width = 7, height = 5, dpi = 150)
message(sprintf("Saved:\n  %s\n  %s", OUT_PDF, OUT_PNG))
