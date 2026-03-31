#!/usr/bin/env Rscript
# reading_time_delta_ll.R
# =======================
# Replication of Oh & Schuler (2023) Figure 1 (Natural Stories SPR panel)
# extended with BabyLM and C4 models.
#
# Method (following Oh & Schuler 2023, Section 3.3):
#   Baseline LME: log(RT) ~ word_length + word_position
#                           + (1 + word_length + word_position | subject)
#                           + (1 | word_type)
#                           + (1 | subject_sentence)
#   Full LME: same + surprisal + (surprisal | subject) [added to random slopes]
#   ΔLL = logLik(full) - logLik(baseline)
#   Split: exploratory set = rows where (subject_id + sentence_id) %% 2 == 0
#
# Inputs:
#   natural_stories/processed_RTs.tsv  — Natural Stories reading times
#     (download from github.com/languageMIT/naturalstories)
#   ns_surprisal/<model>.csv           — per-word surprisal (from get_ns_surprisal.py)
#   ns_surprisal/ns_perplexity.csv     — corpus perplexity per model
#
# Output:
#   ns_delta_ll.csv          — ΔLL and perplexity per model
#   ns_delta_ll_plot.pdf/png — Figure 1 replication

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
  library(ggrepel)
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
  stop("Perplexity file not found. Run get_ns_surprisal.py first.")
}

# ── Load reading times ────────────────────────────────────────────────────────
message("Loading reading times …")

rt_raw <- read_tsv(RT_TSV, show_col_types = FALSE)

# Normalize column names (the RT file uses various conventions across releases)
# Try to map to canonical names: subject, item, zone, word, RT, sentence
rt_raw <- rt_raw |>
  rename_with(tolower) |>
  rename(
    subject  = any_of(c("workerid", "subject", "subj", "participant")),
    item     = any_of(c("item", "story")),
    zone     = any_of(c("zone", "word_num", "position")),
    word     = any_of(c("word")),
    RT       = any_of(c("rt", "reading_time")),
    sentence = any_of(c("sentence", "sent", "sentence_num"))
  )

# Verify required columns exist
req_cols <- c("subject", "item", "zone", "word", "RT")
missing  <- setdiff(req_cols, names(rt_raw))
if (length(missing) > 0) {
  message("Available columns: ", paste(names(rt_raw), collapse = ", "))
  stop("Missing required columns: ", paste(missing, collapse = ", "))
}

message(sprintf("  Raw: %d rows, %d subjects",
                nrow(rt_raw), n_distinct(rt_raw$subject)))

# ── Filter subjects: remove those who answered ≤3 comprehension Qs correctly ─
# Comprehension question accuracy is in columns like correct_1, correct_2, …
# or a summary column. Try both approaches.
if ("correct" %in% names(rt_raw)) {
  # Summary column already present
  good_subjects <- rt_raw |>
    group_by(subject) |>
    summarise(n_correct = sum(correct, na.rm = TRUE)) |>
    filter(n_correct > 3) |>
    pull(subject)
} else {
  correct_cols <- grep("^correct", names(rt_raw), value = TRUE)
  if (length(correct_cols) > 0) {
    good_subjects <- rt_raw |>
      group_by(subject) |>
      summarise(n_correct = sum(across(all_of(correct_cols)), na.rm = TRUE)) |>
      filter(n_correct > 3) |>
      pull(subject)
  } else {
    message("  WARNING: No comprehension question columns found; skipping accuracy filter.")
    good_subjects <- unique(rt_raw$subject)
  }
}

# ── Build analysis dataset ────────────────────────────────────────────────────
# Keep only RT rows (drop comprehension question rows if any)
rt <- rt_raw |>
  filter(subject %in% good_subjects,
         !is.na(RT), !is.na(zone)) |>
  mutate(
    RT      = as.numeric(RT),
    subject = as.character(subject),
    item    = as.integer(item),
    zone    = as.integer(zone)
  )

# Per-story sentence index: sentence is needed for (subject:sentence) random intercept.
# If 'sentence' column exists, use it; otherwise derive from item
if (!"sentence" %in% names(rt)) {
  # Each story (item) is treated as one sentence for the random effect
  rt <- rt |> mutate(sentence = item)
}

# Word-level covariates
rt <- rt |>
  mutate(
    word_length   = nchar(word),
    word_position = zone
  )

# Filter reading times: remove sentence-initial (zone == 1) and sentence-final words,
# RTs < 100 ms or > 3000 ms
# Sentence-final: last zone per item
last_zones <- rt |> group_by(item) |> summarise(last_zone = max(zone))
rt <- rt |>
  left_join(last_zones, by = "item") |>
  filter(
    zone != 1,           # remove sentence-initial
    zone != last_zone,   # remove sentence-final
    RT >= 100,
    RT <= 3000
  ) |>
  select(-last_zone)

message(sprintf("  After filtering: %d rows, %d subjects",
                nrow(rt), n_distinct(rt$subject)))

# Log-transform RT
rt <- rt |> mutate(log_rt = log(RT))

# ── Exploratory / held-out split ──────────────────────────────────────────────
# Split based on sum of (numeric subject_id + sentence_id) %% 2
# Following Oh & Schuler (2023): each subject-by-sentence pair stays intact
rt <- rt |>
  mutate(
    subject_num = as.integer(factor(subject)),
    split_key   = (subject_num + as.integer(sentence)) %% 2,
    set         = if_else(split_key == 0, "exploratory", "held_out")
  )

exploratory <- filter(rt, set == "exploratory")
message(sprintf("  Exploratory set: %d rows", nrow(exploratory)))

# ── Centre and scale all predictors ──────────────────────────────────────────
scale_vec <- function(x) as.numeric(scale(x))
exploratory <- exploratory |>
  mutate(
    word_length_s   = scale_vec(word_length),
    word_position_s = scale_vec(word_position)
  )

# ── Unique word type factor (for by-word random intercept) ───────────────────
exploratory <- exploratory |>
  mutate(word_type      = tolower(word),
         subject_sentence = paste(subject, sentence, sep = "_"))

# ── Fit baseline model (once, reused for all models) ─────────────────────────
message("\nFitting baseline LME …")
baseline_formula <- log_rt ~
  word_length_s + word_position_s +
  (1 + word_length_s + word_position_s | subject) +
  (1 | word_type) +
  (1 | subject_sentence)

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

# ── Compute ΔLL for each model ────────────────────────────────────────────────
surp_files <- list.files(SURP_DIR, pattern = "\\.csv$", full.names = TRUE)
surp_files <- surp_files[basename(surp_files) != "ns_perplexity.csv"]

results <- vector("list", length(surp_files))

for (i in seq_along(surp_files)) {
  csv_path <- surp_files[[i]]
  safe_name <- tools::file_path_sans_ext(basename(csv_path))

  # Recover model ID from safe_name (replace first _ in org/repo with /)
  model_id <- sub("^([^_]+)_(.+)$", "\\1/\\2", safe_name)
  # For simple model IDs (gpt2, gpt2-medium), no slash is expected
  if (!model_id %in% ppl_df$model) model_id <- safe_name

  # Match to perplexity table
  ppl_row <- filter(ppl_df, model == model_id)
  if (nrow(ppl_row) == 0) {
    message(sprintf("  [SKIP] %s — not in perplexity table", model_id))
    next
  }

  message(sprintf("\n[%d/%d] %s", i, length(surp_files), model_id))

  # Load surprisal
  surp <- tryCatch(
    read_csv(csv_path, show_col_types = FALSE),
    error = function(e) { message("  ERROR reading CSV: ", e$message); NULL }
  )
  if (is.null(surp) || nrow(surp) == 0) next

  # Join surprisal to exploratory set
  d <- exploratory |>
    left_join(surp |> select(item, zone, surprisal),
              by = c("item", "zone")) |>
    filter(!is.na(surprisal), is.finite(surprisal))

  if (nrow(d) < 500) {
    message(sprintf("  [SKIP] Only %d matched rows after join", nrow(d)))
    next
  }

  d <- d |> mutate(surprisal_s = scale_vec(surprisal))

  # Fit full model (surprisal added as fixed effect + random slope by subject)
  full_formula <- update(
    baseline_formula,
    . ~ . + surprisal_s + (surprisal_s || subject)
  )

  full_mod <- tryCatch(
    lmer(full_formula, data = d, REML = FALSE,
         control = lmerControl(optimizer = "bobyqa",
                               optCtrl   = list(maxfun = 2e5))),
    error   = function(e) { message("  lmer ERROR: ", e$message); NULL },
    warning = function(w) {
      message("  lmer WARNING: ", w$message)
      suppressWarnings(
        lmer(full_formula, data = d, REML = FALSE,
             control = lmerControl(optimizer = "bobyqa",
                                   optCtrl   = list(maxfun = 2e5)))
      )
    }
  )
  if (is.null(full_mod)) next

  full_ll <- as.numeric(logLik(full_mod))
  delta_ll <- full_ll - baseline_ll

  message(sprintf("  ΔLL = %.2f  (ppl = %.2f)", delta_ll, ppl_row$perplexity[1]))

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

FAMILY_LEVELS  <- c("GPT-2", "GPT-Neo", "OPT", "BabyLM", "C4")
FAMILY_COLOURS <- c(
  "GPT-2"   = "#2166ac",
  "GPT-Neo" = "#4dac26",
  "OPT"     = "#b2182b",
  "BabyLM"  = "#e08214",
  "C4"      = "#35978f"
)
FAMILY_SHAPES <- c(
  "GPT-2"   = 16,
  "GPT-Neo" = 15,
  "OPT"     = 17,
  "BabyLM"  = 18,
  "C4"      = 8
)

plot_df <- results_df |>
  filter(!is.na(delta_ll), !is.na(perplexity)) |>
  mutate(
    family = factor(family, levels = FAMILY_LEVELS),
    # Params label: strip trailing M, convert large values to B notation
    params_num = as.numeric(gsub("M$", "", params)),
    label = case_when(
      params_num >= 1000 ~ paste0(params_num / 1000, "B"),
      TRUE               ~ paste0(params_num, "M")
    )
  )

# Per-family regression lines (log-linear, as in Oh & Schuler Figure 1)
family_lines <- plot_df |>
  group_by(family) |>
  filter(n() >= 2) |>
  group_modify(function(d, k) {
    fit <- lm(delta_ll ~ log(perplexity), data = d)
    x_seq <- seq(min(d$perplexity), max(d$perplexity), length.out = 200)
    tibble(perplexity = x_seq,
           delta_ll   = predict(fit, newdata = data.frame(perplexity = x_seq)))
  }) |>
  ungroup() |>
  mutate(family = factor(family, levels = FAMILY_LEVELS))

p <- ggplot(plot_df, aes(x = perplexity, y = delta_ll,
                         colour = family, shape = family)) +
  # Least-squares regression lines per family (matching Oh & Schuler style)
  geom_line(
    data = family_lines,
    aes(x = perplexity, y = delta_ll, colour = family, group = family),
    linetype = "dotted", linewidth = 0.8, inherit.aes = FALSE
  ) +
  geom_point(size = 3, alpha = 0.9) +
  ggrepel::geom_text_repel(
    aes(label = label),
    size = 2.8, fontface = "bold",
    show.legend    = FALSE,
    min.segment.length = 0.2,
    box.padding    = 0.3,
    point.padding  = 0.2,
    max.overlaps   = 20
  ) +
  scale_x_continuous(
    "Perplexity",
    trans  = "log2",
    labels = scales::trans_format("log2", scales::math_format(2^.x))
  ) +
  scale_y_continuous("\u0394LL") +
  scale_colour_manual(values = FAMILY_COLOURS, name = "Model family") +
  scale_shape_manual(values = FAMILY_SHAPES,  name = "Model family") +
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
