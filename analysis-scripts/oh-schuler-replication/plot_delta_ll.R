# plot_delta_ll.R
#
# Oh & Schuler (2023) replication with binomial ordering preferences.
#
# Left panel  — ΔLL × Perplexity:
#   outcome  = resp_alpha (binary: 1 = human chose α ordering)
#   baseline = glmer(resp_alpha ~ RelFreq + OverallFreq + RelFreq:OverallFreq + (1|binom))
#   full     = glmer(resp_alpha ~ RelFreq + OverallFreq + RelFreq:OverallFreq + preference + (1|binom))
#   ΔLL      = logLik(full) − logLik(baseline)
#
# Middle panel — model preference log-odds coefficient × Perplexity:
#   Coefficient of preference from the full model above.
#
# Right panel — AbsPref coefficient × Perplexity:
#   Fits lm(preference ~ AbsPref + RelFreq + OverallFreq +
#                        AbsPref:OverallFreq + RelFreq:OverallFreq)
#   per model; plots the AbsPref estimate against perplexity.
#
# RelFreq / OverallFreq: uses pile_corpus_freq.csv when available
# (computed by compute_pile_freq.py); falls back to Google Books RelFreq.
#
# Perplexity = validation perplexity on WikiText-2 test set (get_model_prefs.py)
#
# Output: delta_ll_plot.pdf / .png

library(tidyverse)
library(patchwork)
library(lme4)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR <- tryCatch(
  normalizePath(dirname(rstudioapi::getSourceEditorContext()$path), mustWork = FALSE),
  error = function(e) tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE),
    error = function(e2) normalizePath(".")
  )
)
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, "..", ".."), mustWork = FALSE)
DATA_DIR   <- file.path(BASE_DIR, "Data")

PREFS_CSV           <- file.path(SCRIPT_DIR, "oh_schuler_prefs.csv")
PREFS_BY_PROMPT_CSV <- file.path(SCRIPT_DIR, "oh_schuler_prefs_by_prompt.csv")
PPL_CSV             <- file.path(SCRIPT_DIR, "oh_schuler_perplexity.csv")
HUMAN_CSV           <- file.path(DATA_DIR, "all_human_data.csv")
BINOMS_CSV          <- file.path(DATA_DIR, "nonce_and_attested_binoms.csv")
PILE_FREQ_CSV       <- file.path(SCRIPT_DIR, "pile_corpus_freq.csv")
TRAIN_CSV           <- file.path(BASE_DIR, "Data", "processed", "training_attested.csv")
PYTHIA_PPL_CSV      <- file.path(BASE_DIR, "Data", "training_quality", "perplexity_results.csv")
OUT_PDF       <- file.path(SCRIPT_DIR, "delta_ll_plot.pdf")
OUT_PNG       <- file.path(SCRIPT_DIR, "delta_ll_plot.png")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH <- file.path(SCRIPT_DIR, "plot_delta_ll.log")
log_line <- function(...) {
  msg <- paste0(...)
  message(msg)
  cat(msg, "\n", file = LOG_PATH, append = TRUE)
}
cat(sprintf("\n%s\nRun: %s\n", strrep("=", 60), format(Sys.time())),
    file = LOG_PATH, append = TRUE)

# ── Load data ─────────────────────────────────────────────────────────────────
message("Loading data ...")

# Attested binomials only
all_binoms_df   <- read_csv(BINOMS_CSV, show_col_types = FALSE)
attested_binoms <- all_binoms_df |> filter(Attested == 1) |> pull(Alpha)

human_trials_all <- read_csv(HUMAN_CSV, show_col_types = FALSE) |>
  mutate(binom = Alpha, resp_alpha = as.integer(resp == "alpha")) |>
  select(binom, resp_alpha, RelFreq, GenPref, OverallFreq)

human_trials <- human_trials_all |> filter(binom %in% attested_binoms)

model_prefs_all <- read_csv(PREFS_CSV, show_col_types = FALSE)

# ── Fill model metadata if missing from CSV (older exports may lack these columns)
# The Python script adds model_family / model_params / model_label for newly scored
# models, but rows from earlier runs may have NAs or the columns may be absent.
parse_family <- function(m) {
  case_when(
    grepl("gpt-neo",              m, TRUE) ~ "GPT-Neo",
    grepl("gpt2",                 m, TRUE) ~ "GPT-2",
    grepl("olmo-2|olmo_2",        m, TRUE) ~ "OLMo-2",
    grepl("olmo-3|olmo_3|olmo3",  m, TRUE) ~ "OLMo-3",
    grepl("olmo",                 m, TRUE) ~ "OLMo-1",
    grepl("opt-|/opt",            m, TRUE) ~ "OPT",
    grepl("pythia",               m, TRUE) ~ "Pythia",
    grepl("babylm|baby.lm",       m, TRUE) ~ "BabyLM",
    grepl("znhoughton.*c4|c4.*znhoughton|-c4$|/c4$|_c4$", m, TRUE) ~ "C4",
    TRUE ~ NA_character_
  )
}
parse_params <- function(m) {
  case_when(
    grepl("^gpt2$",             m, TRUE) ~ "124M",
    grepl("gpt2-medium",        m, TRUE) ~ "355M",
    grepl("gpt2-large",         m, TRUE) ~ "774M",
    grepl("gpt2-xl",            m, TRUE) ~ "1542M",
    grepl("160m",               m, TRUE) ~ "160M",
    grepl("125m",               m, TRUE) ~ "125M",
    grepl("350m",               m, TRUE) ~ "350M",
    grepl("410m",               m, TRUE) ~ "410M",
    grepl("1[._-]3b|1300m",     m, TRUE) ~ "1300M",
    grepl("1[._-]4b|1400m",     m, TRUE) ~ "1400M",
    grepl("2[._-]7b|2700m",     m, TRUE) ~ "2700M",
    grepl("2[._-]8b|2800m",     m, TRUE) ~ "2800M",
    grepl("6[._-]7b|6700m",     m, TRUE) ~ "6700M",
    grepl("\\b13b\\b|13000m",   m, TRUE) ~ "13000M",
    grepl("\\b30b\\b|30000m",   m, TRUE) ~ "30000M",
    grepl("\\b66b\\b|66000m",   m, TRUE) ~ "66000M",
    grepl("\\b1b\\b|1000m",     m, TRUE) ~ "1000M",
    grepl("\\b7b\\b|7000m",     m, TRUE) ~ "7000M",
    grepl("\\b32b\\b|32000m",   m, TRUE) ~ "32000M",
    TRUE ~ NA_character_
  )
}
fill_model_meta <- function(df) {
  if (!"model_family" %in% names(df)) df$model_family <- NA_character_
  if (!"model_params" %in% names(df)) df$model_params <- NA_character_
  if (!"model_label"  %in% names(df)) df$model_label  <- NA_character_
  df |> mutate(
    # First pass: parse from model name
    model_family = coalesce(model_family, parse_family(model)),
    model_params = coalesce(model_params, parse_params(model)),
    # Second pass: if still NA, fall back to parsing the existing model_label
    # (handles rows where the model identifier doesn't match any regex but the
    # label like "OLMo-1B" was already stored in the CSV by the Python script)
    model_family = coalesce(model_family, parse_family(model_label)),
    model_params = coalesce(model_params, parse_params(model_label)),
    # Construct label only when not already present
    model_label  = coalesce(model_label, paste0(model_family, "-", model_params))
  )
}

model_prefs_all <- fill_model_meta(model_prefs_all)
model_prefs     <- model_prefs_all |> filter(binom %in% attested_binoms)

# Diagnostic: log any models whose family could not be parsed
na_family_models <- model_prefs_all |> filter(is.na(model_family)) |> pull(model) |> unique()
if (length(na_family_models) > 0) {
  log_line("WARNING: models with NA model_family (not plotted): ",
           paste(na_family_models, collapse = ", "))
} else {
  log_line("All models in oh_schuler_prefs.csv have a recognised model_family.")
}

# Per-prompt preferences for mixed model (if available)
has_by_prompt <- file.exists(PREFS_BY_PROMPT_CSV)
if (has_by_prompt) {
  model_prefs_by_prompt <- read_csv(PREFS_BY_PROMPT_CSV, show_col_types = FALSE) |>
    fill_model_meta() |>
    filter(binom %in% attested_binoms)
  # Index of models that have per-prompt data — used in abspref_coef below
  models_with_prompt <- unique(model_prefs_by_prompt$model)
  log_line(sprintf(
    "Per-prompt prefs: %d models with by-prompt data; %d models in averaged prefs only",
    length(models_with_prompt),
    n_distinct(model_prefs$model) - length(intersect(unique(model_prefs$model), models_with_prompt))
  ))
} else {
  models_with_prompt <- character(0)
  message("WARNING: Per-prompt preferences not found — right panel will use averaged data with lm().")
}

ppl <- read_csv(PPL_CSV, show_col_types = FALSE)

# ── Log drop counts ───────────────────────────────────────────────────────────
n_human_before  <- n_distinct(human_trials_all$binom)
n_human_after   <- n_distinct(human_trials$binom)
n_prefs_before  <- n_distinct(model_prefs_all$binom)
n_prefs_after   <- n_distinct(model_prefs$binom)
dropped_human   <- setdiff(unique(human_trials_all$binom), attested_binoms)
dropped_prefs   <- setdiff(unique(model_prefs_all$binom),  attested_binoms)

log_line(sprintf(
  "Human trials:  %d → %d unique binomials retained  (%d dropped as non-attested)",
  n_human_before, n_human_after, n_human_before - n_human_after
))
if (length(dropped_human) > 0)
  log_line("  Dropped from human_trials: ", paste(dropped_human, collapse = ", "))

log_line(sprintf(
  "Model prefs:   %d → %d unique binomials retained  (%d dropped as non-attested)",
  n_prefs_before, n_prefs_after, n_prefs_before - n_prefs_after
))
if (length(dropped_prefs) > 0)
  log_line("  Dropped from model_prefs: ", paste(dropped_prefs, collapse = ", "))

log_line(sprintf(
  "Final: %d human trials, %d unique attested binomials, %d models",
  nrow(human_trials), n_distinct(human_trials$binom), n_distinct(model_prefs$model)
))
message(sprintf("  (full log → %s)", LOG_PATH))

# ── Pile corpus frequencies (if available) ────────────────────────────────────
use_pile_freq <- file.exists(PILE_FREQ_CSV)
if (use_pile_freq) {
  pile_freq <- read_csv(PILE_FREQ_CSV, show_col_types = FALSE) |>
    select(binom, freq_prob_c, log_total) |>
    filter(!is.na(freq_prob_c), !is.na(log_total))
  log_line(sprintf(
    "Pile corpus frequencies loaded: %d binomials with valid freq_prob_c",
    nrow(pile_freq)
  ))
} else {
  message(sprintf(
    "WARNING: Pile corpus freq not found (%s) — falling back to Google Books RelFreq.\n   Run compute_pile_freq.py to generate it.",
    PILE_FREQ_CSV
  ))
}

# ── Per-binom item table (for right-panel regression) ─────────────────────────
# AbsPref  = GenPref (human absolute ordering preference, constant per binom)
# RelFreq / OverallFreq: Pile when available, else Google Books
binom_items <- human_trials |>
  group_by(binom) |>
  summarise(
    AbsPref           = first(GenPref),
    RelFreq_google    = first(RelFreq),
    OverallFreq_google = log(pmax(first(OverallFreq), 1)),   # log(count + 1) guard
    .groups = "drop"
  )

if (use_pile_freq) {
  binom_items <- binom_items |>
    left_join(pile_freq, by = "binom") |>
    rename(RelFreq_pile = freq_prob_c, OverallFreq_pile = log_total)
}

# ── Join human trials with model preferences ──────────────────────────────────
dat <- human_trials |>
  inner_join(model_prefs, by = "binom")

# Attach Pile freq to trial-level data if available (used in ΔLL baseline glm)
if (use_pile_freq) {
  dat <- dat |>
    left_join(pile_freq |> rename(freq_prob_c_pile = freq_prob_c), by = "binom")
}

# ── Helper: choose RelFreq predictor per model family ─────────────────────────
# OPT  → Pile freq_prob_c (if available)
# GPT-2 / GPT-Neo → Google Books RelFreq (no comparable training-data proxy)
# OPT and OLMo were both trained on large web corpora → use Pile freq when available.
# GPT-2 / GPT-Neo → fall back to Google Books RelFreq.
PILE_FREQ_FAMILIES <- c("OPT", "OLMo-1", "OLMo-2", "OLMo-3")

choose_freq <- function(d, family) {
  if (use_pile_freq && family %in% PILE_FREQ_FAMILIES && "freq_prob_c_pile" %in% names(d)) {
    d$freq_use <- d$freq_prob_c_pile
  } else {
    d$freq_use <- d$RelFreq
  }
  d
}

# ── Fit logistic regressions for ΔLL ─────────────────────────────────────────
# Baseline and full models include a random intercept for binomial to account
# for item-level variation in human response rate.  glmer() is attempted first;
# if it fails to converge, falls back to glm().
message("Fitting ΔLL logistic regressions ...")
delta_ll <- dat |>
  group_by(model, model_family, model_params, model_label) |>
  group_modify(function(d, k) {
    d <- choose_freq(d, k$model_family)
    if (any(is.na(d$freq_use))) d <- filter(d, !is.na(freq_use))
    if (nrow(d) < 5) return(tibble(delta_ll = NA_real_, pref_coef = NA_real_,
                                   n_trials = nrow(d), n_items = NA_integer_))

    # Log-transform OverallFreq for the baseline (same as binom_items treatment)
    d <- d |> mutate(ovf = log(pmax(OverallFreq, 1)))

    # Try mixed models with binom random intercept; fall back to glm on failure
    baseline_mod <- tryCatch(
      glmer(resp_alpha ~ freq_use + ovf + freq_use:ovf + (1 | binom),
            family = binomial, data = d,
            control = glmerControl(optimizer = "bobyqa")),
      error = function(e) glm(resp_alpha ~ freq_use + ovf + freq_use:ovf,
                              family = binomial, data = d)
    )
    full_mod <- tryCatch(
      glmer(resp_alpha ~ freq_use + ovf + freq_use:ovf + preference + (1 | binom),
            family = binomial, data = d,
            control = glmerControl(optimizer = "bobyqa")),
      error = function(e) glm(resp_alpha ~ freq_use + ovf + freq_use:ovf + preference,
                              family = binomial, data = d)
    )

    # Extract the preference coefficient from the full model
    cf <- if (inherits(full_mod, "merMod")) fixef(full_mod) else coef(full_mod)
    pref_coef <- if ("preference" %in% names(cf)) cf[["preference"]] else NA_real_

    tibble(
      delta_ll  = as.numeric(logLik(full_mod)) - as.numeric(logLik(baseline_mod)),
      pref_coef = pref_coef,
      n_trials  = nrow(d),
      n_items   = n_distinct(d$binom)
    )
  }) |>
  ungroup()

# ── Fit item-level mixed model for AbsPref coefficient ────────────────────────
# Uses per-prompt data (one row per binom×prompt) with random intercepts for
# binom and prompt. Falls back to lm() on averaged data if by-prompt CSV absent.
message("Fitting AbsPref mixed models ...")

# Iterate over ALL models in model_prefs (so we never drop a model that predates
# oh_schuler_prefs_by_prompt.csv).  If a model also has per-prompt data, we use
# lmer with random intercepts for binom and prompt; otherwise we fall back to lm.
abspref_coef <- model_prefs |>
  group_by(model, model_family, model_params, model_label) |>
  group_modify(function(d, k) {
    this_model <- k$model

    # Use per-prompt data when available for this specific model
    if (has_by_prompt && this_model %in% models_with_prompt) {
      d_fit <- model_prefs_by_prompt |>
        filter(model == this_model) |>
        left_join(binom_items, by = "binom")
      use_lmer <- TRUE
    } else {
      d_fit <- d |> left_join(binom_items, by = "binom")
      use_lmer <- FALSE
    }

    # Choose RelFreq and OverallFreq based on model family & availability
    if (use_pile_freq && k$model_family %in% PILE_FREQ_FAMILIES &&
        "RelFreq_pile" %in% names(d_fit) && "OverallFreq_pile" %in% names(d_fit)) {
      d_fit$rf  <- d_fit$RelFreq_pile
      d_fit$ovf <- d_fit$OverallFreq_pile
    } else {
      d_fit$rf  <- d_fit$RelFreq_google
      d_fit$ovf <- d_fit$OverallFreq_google
    }

    d_fit <- d_fit |> filter(!is.na(AbsPref), !is.na(rf), !is.na(ovf), !is.na(preference))
    if (nrow(d_fit) < 5) return(tibble(estimate = NA_real_, se = NA_real_, t = NA_real_, p = NA_real_, n = nrow(d_fit)))

    # Center all predictors
    d_fit <- d_fit |> mutate(
      AbsPref_c = as.numeric(scale(AbsPref, scale = FALSE)),
      rf_c      = as.numeric(scale(rf,      scale = FALSE)),
      ovf_c     = as.numeric(scale(ovf,     scale = FALSE))
    )

    if (use_lmer && "prompt" %in% names(d_fit)) {
      fit <- tryCatch(
        lmer(preference ~ AbsPref_c + rf_c + ovf_c +
               AbsPref_c:ovf_c + rf_c:ovf_c +
               (1 | binom) + (1 | prompt),
             data = d_fit, REML = TRUE),
        error = function(e) NULL
      )
      if (is.null(fit)) {
        fit <- lm(preference ~ AbsPref_c + rf_c + ovf_c +
                    AbsPref_c:ovf_c + rf_c:ovf_c, data = d_fit)
      }
      cf <- summary(fit)$coefficients
    } else {
      fit <- lm(preference ~ AbsPref_c + rf_c + ovf_c +
                  AbsPref_c:ovf_c + rf_c:ovf_c, data = d_fit)
      cf  <- summary(fit)$coefficients
    }

    if (!"AbsPref_c" %in% rownames(cf)) return(tibble(estimate = NA_real_, se = NA_real_, t = NA_real_, p = NA_real_, n = nrow(d_fit)))
    tibble(
      estimate = cf["AbsPref_c", "Estimate"],
      se       = cf["AbsPref_c", "Std. Error"],
      t        = cf["AbsPref_c", "t value"],
      p        = if ("Pr(>|t|)" %in% colnames(cf)) cf["AbsPref_c", "Pr(>|t|)"] else NA_real_,
      n        = nrow(d_fit)
    )
  }) |>
  ungroup()

# ── Pythia final-checkpoint ΔLL ───────────────────────────────────────────────
# Pythia data lives in training_attested.csv (already processed). Dynamic
# per-checkpoint exposures are NOT available for Pythia, so we use static
# RelFreq (Google Books) as the baseline predictor.
pythia_results <- NULL
if (file.exists(TRAIN_CSV)) {
  message("Loading Pythia final-checkpoint data from training_attested.csv ...")

  parse_pythia_size <- function(x) dplyr::case_when(
    grepl("160m",    x, TRUE) ~ "160M",
    grepl("410m",    x, TRUE) ~ "410M",
    grepl("1[._]4b", x, TRUE) ~ "1400M",
    grepl("2[._]8b", x, TRUE) ~ "2800M",
    TRUE ~ NA_character_
  )

  train_all <- read_csv(TRAIN_CSV, show_col_types = FALSE)

  pythia_final <- train_all |>
    filter(grepl("pythia", model, TRUE)) |>
    group_by(model) |>
    filter(step == max(step)) |>
    ungroup() |>
    filter(binom %in% attested_binoms) |>
    mutate(
      model_family = "Pythia",
      model_params = parse_pythia_size(model),
      model_label  = paste0("Pythia-", parse_pythia_size(model))
    ) |>
    filter(!is.na(model_params))

  # Join static RelFreq and OverallFreq from human trials
  pythia_dat <- human_trials |>
    select(binom, resp_alpha, RelFreq, OverallFreq) |>
    inner_join(pythia_final |> select(model, model_family, model_params, model_label,
                                      binom, preference),
               by = "binom")

  pythia_delta_ll <- pythia_dat |>
    group_by(model, model_family, model_params, model_label) |>
    group_modify(function(d, k) {
      d <- filter(d, !is.na(RelFreq), !is.na(OverallFreq), !is.na(preference))
      d <- d |> mutate(ovf = log(pmax(OverallFreq, 1)))
      if (nrow(d) < 5) return(tibble(delta_ll = NA_real_, pref_coef = NA_real_,
                                     n_trials = nrow(d), n_items = 0L))
      baseline <- tryCatch(
        glmer(resp_alpha ~ RelFreq + ovf + RelFreq:ovf + (1 | binom),
              family = binomial, data = d,
              control = glmerControl(optimizer = "bobyqa")),
        error = function(e) glm(resp_alpha ~ RelFreq + ovf + RelFreq:ovf,
                                family = binomial, data = d)
      )
      full <- tryCatch(
        glmer(resp_alpha ~ RelFreq + ovf + RelFreq:ovf + preference + (1 | binom),
              family = binomial, data = d,
              control = glmerControl(optimizer = "bobyqa")),
        error = function(e) glm(resp_alpha ~ RelFreq + ovf + RelFreq:ovf + preference,
                                family = binomial, data = d)
      )
      cf        <- if (inherits(full, "merMod")) fixef(full) else coef(full)
      pref_coef <- if ("preference" %in% names(cf)) cf[["preference"]] else NA_real_
      tibble(delta_ll  = as.numeric(logLik(full)) - as.numeric(logLik(baseline)),
             pref_coef = pref_coef,
             n_trials  = nrow(d), n_items = n_distinct(d$binom))
    }) |>
    ungroup()

  # Pythia AbsPref coefficient — same formula as the main abspref_coef block
  pythia_abspref_coef <- pythia_final |>
    left_join(binom_items, by = "binom") |>
    group_by(model, model_family, model_params, model_label) |>
    group_modify(function(d, k) {
      if (use_pile_freq && "RelFreq_pile" %in% names(d) && "OverallFreq_pile" %in% names(d)) {
        d$rf  <- coalesce(d$RelFreq_pile,     d$RelFreq_google)
        d$ovf <- coalesce(d$OverallFreq_pile, d$OverallFreq_google)
      } else {
        d$rf  <- d$RelFreq_google
        d$ovf <- d$OverallFreq_google
      }
      d <- d |> filter(!is.na(AbsPref), !is.na(rf), !is.na(ovf), !is.na(preference))
      if (nrow(d) < 5) return(tibble(estimate = NA_real_, se = NA_real_,
                                     t = NA_real_, p = NA_real_, n = nrow(d)))
      d <- d |> mutate(
        AbsPref_c = as.numeric(scale(AbsPref, scale = FALSE)),
        rf_c      = as.numeric(scale(rf,      scale = FALSE)),
        ovf_c     = as.numeric(scale(ovf,     scale = FALSE))
      )
      fit <- lm(preference ~ AbsPref_c + rf_c + ovf_c +
                  AbsPref_c:ovf_c + rf_c:ovf_c, data = d)
      cf  <- summary(fit)$coefficients
      if (!"AbsPref_c" %in% rownames(cf)) return(tibble(estimate = NA_real_, se = NA_real_,
                                                         t = NA_real_, p = NA_real_, n = nrow(d)))
      tibble(estimate = cf["AbsPref_c", "Estimate"],
             se       = cf["AbsPref_c", "Std. Error"],
             t        = cf["AbsPref_c", "t value"],
             p        = if ("Pr(>|t|)" %in% colnames(cf)) cf["AbsPref_c", "Pr(>|t|)"] else NA_real_,
             n        = nrow(d))
    }) |>
    ungroup()

  # Pythia perplexity — from oh_schuler_perplexity.csv (WikiText-2, same benchmark
  # as all other models in this plot). Computed by get_model_prefs.py with ppl_only=True.
  pythia_results <- pythia_delta_ll |>
    left_join(pythia_abspref_coef |> select(model, estimate, se, t, p), by = "model") |>
    left_join(ppl |> select(model, perplexity), by = "model") |>
    mutate(params_M = as.numeric(sub("M$", "", model_params)))

  message(sprintf("  Pythia: %d models (%d with AbsPref estimate)",
                  nrow(pythia_results), sum(!is.na(pythia_results$estimate))))
} else {
  message("training_attested.csv not found — Pythia points omitted.")
}

# ── Combine and join perplexity ───────────────────────────────────────────────
results <- delta_ll |>
  left_join(abspref_coef, by = c("model", "model_family", "model_params", "model_label")) |>
  left_join(ppl |> select(model, perplexity), by = "model") |>
  filter(!is.na(perplexity)) |>
  mutate(params_M = as.numeric(sub("M$", "", model_params)))

if (!is.null(pythia_results) && nrow(pythia_results) > 0) {
  results <- bind_rows(results, pythia_results |> filter(!is.na(perplexity)))
}

results <- results |>
  mutate(
    model_family = factor(model_family,
                          levels = c("GPT-2", "GPT-Neo", "OPT",
                                     "OLMo-1", "OLMo-2", "OLMo-3",
                                     "Pythia", "BabyLM", "C4"))
  ) |>
  arrange(model_family, params_M)

message("\nResults:")
print(results |> select(model_family, model_label, model_params, perplexity,
                        delta_ll, pref_coef, estimate),
      n = Inf)

# ── Plot aesthetics ───────────────────────────────────────────────────────────
family_colours <- c(
  "GPT-2"  = "#2166ac",
  "GPT-Neo"= "#4dac26",
  "OPT"    = "#b2182b",
  "OLMo-1" = "#d6604d",
  "OLMo-2" = "#f4a582",
  "OLMo-3" = "#92c5de",
  "Pythia" = "#762a83",
  "BabyLM" = "#e08214",
  "C4"     = "#35978f"
)

shared_layers <- list(
  scale_x_continuous(
    "Validation perplexity on WikiText-2 (lower \u2192 better)",
    trans  = "log2",
    labels = scales::trans_format("log2", scales::math_format(2^.x))
  ),
  scale_colour_manual(values = family_colours, name = "Model family"),
  scale_fill_manual(values = family_colours, name = "Model family"),
  guides(colour = guide_legend(override.aes = list(size = 3))),
  theme_classic(base_size = 13),
  theme(
    panel.grid.major = element_line(colour = "grey92", linewidth = 0.4),
    axis.line        = element_line(colour = "grey40"),
    axis.ticks       = element_line(colour = "grey40"),
    legend.position  = "right"
  )
)

# ── Significance tests ────────────────────────────────────────────────────────
slope_tests <- results |>
  filter(!is.na(delta_ll)) |>
  group_by(model_family) |>
  group_modify(function(d, k) {
    if (nrow(d) < 3) return(tibble(r = NA_real_, p = NA_real_, df = NA_integer_))
    ct <- cor.test(log2(d$perplexity), d$delta_ll)
    tibble(r = ct$estimate, p = ct$p.value, df = ct$parameter)
  }) |>
  ungroup() |>
  mutate(lbl = sprintf("r = %.2f, p %s %.3f",
                       r,
                       ifelse(p < .001, "<", "="),
                       pmax(p, 0.001)))

message("\nSlope tests (log2 perplexity × ΔLL) by family:")
print(slope_tests)

# ── Pre-compute per-family model counts for smooth vs. line decision ──────────
# Families with ≥ 3 models get a polynomial smooth (+ CI ribbon).
# Families with exactly 2 models get a plain dashed line (poly() needs ≥ 3 pts).
left_data  <- filter(results, !is.na(delta_ll))
right_data <- filter(results, !is.na(estimate))

fam_n_left  <- left_data  |> count(model_family)
fam_n_right <- right_data |> count(model_family)

poly_fams_left  <- fam_n_left  |> filter(n >= 3) |> pull(model_family)
line_fams_left  <- fam_n_left  |> filter(n == 2) |> pull(model_family)
poly_fams_right <- fam_n_right |> filter(n >= 3) |> pull(model_family)
line_fams_right <- fam_n_right |> filter(n == 2) |> pull(model_family)

log_line(sprintf(
  "Left panel  — poly smooth: [%s] | plain line: [%s]",
  paste(poly_fams_left,  collapse = ", "),
  paste(line_fams_left,  collapse = ", ")
))
log_line(sprintf(
  "Right panel — poly smooth: [%s] | plain line: [%s]",
  paste(poly_fams_right, collapse = ", "),
  paste(line_fams_right, collapse = ", ")
))

# ── Pre-compute polynomial CIs in log2-perplexity space ──────────────────────
# geom_smooth with scale_x_continuous(trans="log2") can fit on the raw scale
# rather than the transformed scale in some ggplot2 versions, misplacing the CI
# ribbon.  We fit explicitly in log2 space and plot with geom_ribbon + geom_line
# to guarantee the ribbon appears at the correct y positions.
poly_smooth_ci <- function(data, y_col, families, n_points = 200) {
  data |>
    filter(model_family %in% families) |>
    group_by(model_family) |>
    group_modify(function(d, k) {
      x_raw <- d$perplexity
      y_raw <- d[[y_col]]
      ok    <- !is.na(x_raw) & !is.na(y_raw)
      if (sum(ok) < 3) return(tibble())
      x_log <- log2(x_raw[ok])
      x_seq <- seq(min(x_log), max(x_log), length.out = n_points)
      fit   <- lm(y ~ poly(x, 2), data = data.frame(x = x_log, y = y_raw[ok]))
      pred  <- predict(fit, newdata = data.frame(x = x_seq), interval = "confidence")
      tibble(perplexity = 2^x_seq,
             fit = pred[, "fit"],
             lwr = pred[, "lwr"],
             upr = pred[, "upr"])
    }) |>
    ungroup()
}

smooth_left  <- poly_smooth_ci(left_data,  "delta_ll", poly_fams_left)
smooth_right <- poly_smooth_ci(right_data, "estimate", poly_fams_right)

# ── Left panel: ΔLL ──────────────────────────────────────────────────────────
p_left <- ggplot(left_data,
                 aes(x = perplexity, y = delta_ll, colour = model_family)) +
  # CI ribbon — drawn first so points appear on top
  geom_ribbon(
    data = smooth_left,
    aes(x = perplexity, ymin = lwr, ymax = upr, fill = model_family),
    alpha = 0.15, colour = NA, inherit.aes = FALSE
  ) +
  # Polynomial trend line for families with ≥ 3 models
  geom_line(
    data = smooth_left,
    aes(x = perplexity, y = fit, colour = model_family, group = model_family),
    linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE
  ) +
  # Plain dashed line for families with exactly 2 models
  geom_line(
    data     = ~ arrange(filter(.x, model_family %in% line_fams_left),
                         model_family, perplexity),
    aes(group = model_family),
    linetype = "dashed", linewidth = 0.9
  ) +
  geom_point(size = 3, alpha = 0.9) +
  geom_text(aes(label = model_params),
            hjust = -0.12, vjust = 0.4,
            size = 3, fontface = "bold", show.legend = FALSE) +
  scale_y_continuous("\u0394LL") +
  labs(
    title    = "\u0394LL vs. validation perplexity",
    subtitle = paste0(
      "\u0394LL = logLik(full) \u2212 logLik(baseline). Baseline: ",
      ifelse(use_pile_freq, "Pile freq [OPT/OLMo] / Google freq [GPT]", "Google Books RelFreq"),
      " + OverallFreq + interaction + (1|binom). Full adds model pref."
    )
  ) +
  shared_layers

# ── Right panel: AbsPref coefficient ─────────────────────────────────────────
p_right <- ggplot(right_data,
                  aes(x = perplexity, y = estimate, colour = model_family)) +
  geom_hline(yintercept = 0, linetype = "dotted", colour = "grey50") +
  # CI ribbon — drawn first so points appear on top
  geom_ribbon(
    data = smooth_right,
    aes(x = perplexity, ymin = lwr, ymax = upr, fill = model_family),
    alpha = 0.15, colour = NA, inherit.aes = FALSE
  ) +
  # Polynomial trend line for families with ≥ 3 models
  geom_line(
    data = smooth_right,
    aes(x = perplexity, y = fit, colour = model_family, group = model_family),
    linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE
  ) +
  # Plain dashed line for families with exactly 2 models
  geom_line(
    data     = ~ arrange(filter(.x, model_family %in% line_fams_right),
                         model_family, perplexity),
    aes(group = model_family),
    linetype = "dashed", linewidth = 0.9
  ) +
  geom_point(size = 3, alpha = 0.9) +
  geom_text(aes(label = model_params),
            hjust = -0.12, vjust = 0.4,
            size = 3, fontface = "bold", show.legend = FALSE) +
  scale_y_continuous("AbsPref coefficient") +
  labs(
    title    = "AbsPref \u03b2 vs. validation perplexity",
    subtitle = paste0(
      "From lm(model pref \u223c AbsPref + RelFreq + OverallFreq + ",
      "AbsPref\u00d7OverallFreq + RelFreq\u00d7OverallFreq). ",
      "Dashed = quadratic fit per family."
    )
  ) +
  shared_layers

# ── Middle panel: logistic preference coefficient (resp_alpha ~ model pref) ───
# Shows the raw log-odds coefficient of model preference predicting human resp_alpha
# (from the full mixed logistic regression above), plotted vs. perplexity.
middle_data <- filter(results, !is.na(pref_coef))

fam_n_middle  <- middle_data |> count(model_family)
poly_fams_mid <- fam_n_middle |> filter(n >= 3) |> pull(model_family)
line_fams_mid <- fam_n_middle |> filter(n == 2) |> pull(model_family)

smooth_middle <- poly_smooth_ci(middle_data, "pref_coef", poly_fams_mid)

p_middle <- ggplot(middle_data,
                   aes(x = perplexity, y = pref_coef, colour = model_family)) +
  geom_hline(yintercept = 0, linetype = "dotted", colour = "grey50") +
  geom_ribbon(
    data = smooth_middle,
    aes(x = perplexity, ymin = lwr, ymax = upr, fill = model_family),
    alpha = 0.15, colour = NA, inherit.aes = FALSE
  ) +
  geom_line(
    data = smooth_middle,
    aes(x = perplexity, y = fit, colour = model_family, group = model_family),
    linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE
  ) +
  geom_line(
    data     = ~ arrange(filter(.x, model_family %in% line_fams_mid),
                         model_family, perplexity),
    aes(group = model_family),
    linetype = "dashed", linewidth = 0.9
  ) +
  geom_point(size = 3, alpha = 0.9) +
  geom_text(aes(label = model_params),
            hjust = -0.12, vjust = 0.4,
            size = 3, fontface = "bold", show.legend = FALSE) +
  scale_y_continuous("Model pref \u03b2 (log-odds)") +
  labs(
    title    = "Model pref \u03b2 vs. validation perplexity",
    subtitle = paste0(
      "Log-odds coefficient of model preference in ",
      "glmer(resp_alpha \u223c RelFreq + OverallFreq + RelFreq\u00d7OverallFreq + pref + (1|binom)). ",
      "Dashed = quadratic fit per family."
    )
  ) +
  shared_layers

# ── Combine and save ──────────────────────────────────────────────────────────
p_combined <- p_left + p_middle + p_right +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

print(p_combined)

ggsave(OUT_PDF, p_combined, width = 20, height = 5.5)
ggsave(OUT_PNG, p_combined, width = 20, height = 5.5, dpi = 150)
message(sprintf("\nSaved:\n  %s\n  %s", OUT_PDF, OUT_PNG))
