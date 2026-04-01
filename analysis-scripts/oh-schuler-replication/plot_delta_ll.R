# plot_delta_ll.R
#
# Oh & Schuler (2023) replication with binomial ordering preferences.
#
# Left panel  — ΔLL × Perplexity
# Middle panel — model preference log-odds coefficient × Perplexity
# Right panel  — AbsPref coefficient × Perplexity
#
# Output: delta_ll_plot.pdf / .png

library(tidyverse)
library(ggh4x)
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
NS_PPL_CSV          <- file.path(SCRIPT_DIR, "ns_surprisal", "ns_perplexity.csv")
HUMAN_CSV           <- file.path(DATA_DIR, "all_human_data.csv")
BINOMS_CSV          <- file.path(DATA_DIR, "nonce_and_attested_binoms.csv")
PILE_FREQ_CSV       <- file.path(SCRIPT_DIR, "pile_corpus_freq.csv")
TRAIN_CSV           <- file.path(BASE_DIR, "Data", "processed", "training_attested.csv")
OUT_PDF             <- file.path(SCRIPT_DIR, "delta_ll_plot.pdf")
OUT_PNG             <- file.path(SCRIPT_DIR, "delta_ll_plot.png")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH <- file.path(SCRIPT_DIR, "plot_delta_ll.log")
log_line <- function(...) {
  msg <- paste0(...)
  message(msg)
  cat(msg, "\n", file = LOG_PATH, append = TRUE)
}
cat(sprintf("\n%s\nRun: %s\n", strrep("=", 60), format(Sys.time())),
    file = LOG_PATH, append = TRUE)

# ── Model metadata helpers ────────────────────────────────────────────────────
# NOTE: parse_family() and parse_params() are used only as a fallback for rows
# in CSVs that pre-date the model_family/model_params columns being written by
# get_model_prefs.py.  New rows always have these filled already.
#
# Key fixes vs. previous version:
#   - BabyLM and C4 families added (znhoughton/opt-babylm-* and opt-c4-*)
#   - OLMo stale rows ("OLMo" family, no "-hf" suffix) are remapped to "OLMo-1"
#   - parse_params now matches "1b", "1.3b", "7b" etc. via a shared regex that
#     handles both label strings ("BabyLM-1.3B") and HF model IDs ("opt-1.3b")
#     without \\b word-boundary failures on hyphen-adjacent tokens.

parse_family <- function(m) {
  case_when(
    # znhoughton custom models — must come before generic OPT match
    grepl("znhoughton.*babylm|babylm.*znhoughton", m, TRUE) ~ "BabyLM",
    grepl("znhoughton.*c4|c4.*znhoughton|-c4[-_]|/c4[-_]|_c4[-_]|opt-c4", m, TRUE) ~ "C4",
    grepl("gpt-neo",              m, TRUE) ~ "GPT-Neo",
    grepl("gpt2",                 m, TRUE) ~ "GPT-2",
    # OLMo: check gen-2 and gen-3 before gen-1 so the more specific wins
    grepl("olmo-2|olmo_2",        m, TRUE) ~ "OLMo-2",
    grepl("olmo-3|olmo_3|olmo3",  m, TRUE) ~ "OLMo-3",
    grepl("olmo",                 m, TRUE) ~ "OLMo-1",   # covers OLMo-1 and stale "OLMo" rows
    grepl("opt-|/opt",            m, TRUE) ~ "OPT",
    grepl("pythia",               m, TRUE) ~ "Pythia",
    grepl("babylm|baby.lm",       m, TRUE) ~ "BabyLM",
    TRUE ~ NA_character_
  )
}

parse_params <- function(m) {
  # Normalise: lower-case, collapse separators so "1.3B", "1.3b", "1_3b" all hit
  mn <- tolower(gsub("[\\s_]", "-", m, perl = TRUE))
  case_when(
    # Exact GPT-2 base (no size suffix)
    grepl("^gpt2$",             mn) ~ "124M",
    grepl("gpt2-medium",        mn) ~ "355M",
    grepl("gpt2-large",         mn) ~ "774M",
    grepl("gpt2-xl",            mn) ~ "1542M",
    # Named sizes — order matters: check larger first to avoid "1b" matching inside "1.3b"
    grepl("175b",               mn) ~ "175000M",
    grepl("66b",                mn) ~ "66000M",
    grepl("32b",                mn) ~ "32000M",
    grepl("30b",                mn) ~ "30000M",
    grepl("13b",                mn) ~ "13000M",
    grepl("6\\.7b",             mn) ~ "6700M",
    grepl("2\\.8b",             mn) ~ "2800M",
    grepl("2\\.7b",             mn) ~ "2700M",
    grepl("1\\.4b",             mn) ~ "1400M",
    grepl("1\\.3b",             mn) ~ "1300M",
    grepl("7b",                 mn) ~ "7000M",
    grepl("1b",                 mn) ~ "1000M",   # must come AFTER all x.yb patterns
    grepl("410m",               mn) ~ "410M",
    grepl("350m",               mn) ~ "350M",
    grepl("160m",               mn) ~ "160M",
    grepl("125m",               mn) ~ "125M",
    grepl("124m",               mn) ~ "124M",
    TRUE ~ NA_character_
  )
}

fill_model_meta <- function(df) {
  if (!"model_family" %in% names(df)) df$model_family <- NA_character_
  if (!"model_params" %in% names(df)) df$model_params <- NA_character_
  if (!"model_label"  %in% names(df)) df$model_label  <- NA_character_
  df |> mutate(
    # Remap the stale "OLMo" family written by earlier script versions
    model_family = if_else(model_family == "OLMo", "OLMo-1", model_family),
    # First-pass: parse from model ID
    model_family = coalesce(model_family, parse_family(model)),
    model_params = coalesce(model_params, parse_params(model)),
    # Second-pass: if still NA, try parsing from stored label
    model_family = coalesce(model_family, parse_family(model_label)),
    model_params = coalesce(model_params, parse_params(model_label)),
    model_label  = coalesce(model_label, paste0(model_family, "-", model_params))
  )
}

# ── Load data ─────────────────────────────────────────────────────────────────
message("Loading data ...")

all_binoms_df   <- read_csv(BINOMS_CSV, show_col_types = FALSE)
attested_binoms <- all_binoms_df |> filter(Attested == 1) |> pull(Alpha)

human_trials_all <- read_csv(HUMAN_CSV, show_col_types = FALSE) |>
  mutate(binom = Alpha, resp_alpha = as.integer(resp == "alpha")) |>
  select(binom, participant, resp_alpha, RelFreq, GenPref, OverallFreq)

human_trials <- human_trials_all |> filter(binom %in% attested_binoms)

model_prefs_all <- read_csv(PREFS_CSV, show_col_types = FALSE) |>
  fill_model_meta()

# De-duplicate: if a model appears under both an old and new ID, keep the one
# whose model_family is not NA (new rows always have it).
model_prefs_all <- model_prefs_all |>
  group_by(binom) |>
  # Within each binom, drop rows that are pure duplicates of another row
  distinct() |>
  ungroup()

model_prefs <- model_prefs_all |> filter(binom %in% attested_binoms)

# Log any models that still can't be parsed
na_family_models <- model_prefs_all |> filter(is.na(model_family)) |> pull(model) |> unique()
if (length(na_family_models) > 0) {
  log_line("WARNING: models with NA model_family (will be dropped from plots): ",
           paste(na_family_models, collapse = ", "))
} else {
  log_line("All models in oh_schuler_prefs.csv have a recognised model_family.")
}

# Per-prompt preferences
has_by_prompt <- file.exists(PREFS_BY_PROMPT_CSV)
if (has_by_prompt) {
  model_prefs_by_prompt <- read_csv(PREFS_BY_PROMPT_CSV, show_col_types = FALSE) |>
    fill_model_meta() |>
    filter(binom %in% attested_binoms)
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

ppl <- read_csv(PPL_CSV, show_col_types = FALSE) |>
  fill_model_meta() |>
  # Drop stale duplicate OLMo rows (non-hf IDs) — keep -hf variants as canonical
  filter(!model %in% c("allenai/OLMo-1B", "allenai/OLMo-7B"))

# Merge in checkpoint perplexities from ns_perplexity.csv (model@revision keys)
if (file.exists(NS_PPL_CSV)) {
  ns_ppl <- read_csv(NS_PPL_CSV, show_col_types = FALSE) |>
    filter(grepl("@", model))   # only checkpoint entries have @revision
  if (nrow(ns_ppl) > 0) {
    ppl <- bind_rows(ppl, ns_ppl |> select(model, family, params, perplexity))
    log_line(sprintf("Merged %d checkpoint perplexity rows from ns_perplexity.csv", nrow(ns_ppl)))
  }
}

# ── Log drop counts ───────────────────────────────────────────────────────────
n_human_before  <- n_distinct(human_trials_all$binom)
n_human_after   <- n_distinct(human_trials$binom)
n_prefs_before  <- n_distinct(model_prefs_all$binom)
n_prefs_after   <- n_distinct(model_prefs$binom)

log_line(sprintf(
  "Human trials:  %d → %d unique binomials retained  (%d dropped as non-attested)",
  n_human_before, n_human_after, n_human_before - n_human_after
))
log_line(sprintf(
  "Model prefs:   %d → %d unique binomials retained  (%d dropped as non-attested)",
  n_prefs_before, n_prefs_after, n_prefs_before - n_prefs_after
))
log_line(sprintf(
  "Final: %d human trials, %d unique attested binomials, %d models",
  nrow(human_trials), n_distinct(human_trials$binom), n_distinct(model_prefs$model)
))
message(sprintf("  (full log → %s)", LOG_PATH))

# ── Pile corpus frequencies ───────────────────────────────────────────────────
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
    "WARNING: Pile corpus freq not found (%s) — falling back to Google Books RelFreq.",
    PILE_FREQ_CSV
  ))
}

# ── Per-binom item table ──────────────────────────────────────────────────────
binom_items <- human_trials |>
  group_by(binom) |>
  summarise(
    AbsPref            = first(GenPref),
    RelFreq_google     = first(RelFreq),
    OverallFreq_google = log(pmax(first(OverallFreq), 1)),
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

if (use_pile_freq) {
  dat <- dat |>
    left_join(pile_freq |> rename(freq_prob_c_pile = freq_prob_c), by = "binom")
}

# ── Frequency predictor selection ─────────────────────────────────────────────
# OPT, OLMo, BabyLM, and C4 are all trained on large web corpora → use Pile
# freq when available. GPT-2 / GPT-Neo → Google Books RelFreq.
PILE_FREQ_FAMILIES <- c("OPT", "OLMo-1", "OLMo-2", "OLMo-3", "BabyLM", "C4")

choose_freq <- function(d, family) {
  if (use_pile_freq && family %in% PILE_FREQ_FAMILIES && "freq_prob_c_pile" %in% names(d)) {
    d$freq_use <- d$freq_prob_c_pile
  } else {
    d$freq_use <- d$RelFreq
  }
  d
}

# ── Fit logistic regressions for ΔLL ─────────────────────────────────────────
message("Fitting ΔLL logistic regressions ...")
delta_ll <- dat |>
  group_by(model, model_family, model_params, model_label) |>
  group_modify(function(d, k) {
    d <- choose_freq(d, k$model_family)
    if (any(is.na(d$freq_use))) d <- filter(d, !is.na(freq_use))
    if (nrow(d) < 5) return(tibble(delta_ll = NA_real_, pref_coef = NA_real_,
                                   n_trials = nrow(d), n_items = NA_integer_))
    d <- d |> mutate(ovf = log(pmax(OverallFreq, 1)))

    baseline_mod <- glmer(
      resp_alpha ~ freq_use + (1 | binom) + (1 | participant),
      family = binomial, data = d,
      control = glmerControl(optimizer = "bobyqa")
    )
    full_mod <- glmer(
      resp_alpha ~ freq_use + preference +
        (1 | binom) + (1 | participant),
      family = binomial, data = d,
      control = glmerControl(optimizer = "bobyqa")
    )
    tibble(
      delta_ll = as.numeric(logLik(full_mod)) - as.numeric(logLik(baseline_mod)),
      n_trials = nrow(d),
      n_items  = n_distinct(d$binom)
    )
  }) |>
  ungroup()

# ── Fit separate GLM for model-preference coefficient ─────────────────────────
message("Fitting model-preference GLMs ...")
pref_coef_results <- dat |>
  group_by(model, model_family, model_params, model_label) |>
  group_modify(function(d, k) {
    d <- filter(d, !is.na(preference))
    if (nrow(d) < 5) return(tibble(pref_coef = NA_real_))
    fit <- tryCatch(
      glmer(
        resp_alpha ~ preference + (1 | binom) + (1 | participant),
        family  = binomial, data = d,
        control = glmerControl(optimizer = "bobyqa")
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) return(tibble(pref_coef = NA_real_))
    tibble(pref_coef = fixef(fit)[["preference"]])
  }) |>
  ungroup()

# ── Fit item-level mixed model for AbsPref coefficient ────────────────────────
message("Fitting AbsPref mixed models ...")

abspref_coef <- model_prefs |>
  group_by(model, model_family, model_params, model_label) |>
  group_modify(function(d, k) {
    this_model <- k$model

    if (has_by_prompt && this_model %in% models_with_prompt) {
      d_fit <- model_prefs_by_prompt |>
        filter(model == this_model) |>
        left_join(binom_items, by = "binom")
      use_lmer <- TRUE
    } else {
      d_fit <- d |> left_join(binom_items, by = "binom")
      use_lmer <- FALSE
    }

    if (use_pile_freq && k$model_family %in% PILE_FREQ_FAMILIES &&
        "RelFreq_pile" %in% names(d_fit) && "OverallFreq_pile" %in% names(d_fit)) {
      d_fit$rf  <- d_fit$RelFreq_pile
      d_fit$ovf <- d_fit$OverallFreq_pile
    } else {
      d_fit$rf  <- d_fit$RelFreq_google
      d_fit$ovf <- d_fit$OverallFreq_google
    }

    d_fit <- d_fit |> filter(!is.na(AbsPref), !is.na(rf), !is.na(ovf), !is.na(preference))
    if (nrow(d_fit) < 5) return(tibble(estimate = NA_real_, se = NA_real_,
                                       t = NA_real_, p = NA_real_, n = nrow(d_fit)))
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
    } else {
      fit <- NULL
    }
    if (is.null(fit)) {
      fit <- tryCatch(
        lmer(preference ~ AbsPref_c + rf_c + ovf_c +
               AbsPref_c:ovf_c + rf_c:ovf_c +
               (1 | binom),
             data = d_fit, REML = TRUE),
        error = function(e) lm(preference ~ AbsPref_c + rf_c + ovf_c +
                                 AbsPref_c:ovf_c + rf_c:ovf_c, data = d_fit)
      )
    }
    cf <- summary(fit)$coefficients
    if (!"AbsPref_c" %in% rownames(cf)) return(tibble(estimate = NA_real_, se = NA_real_,
                                                       t = NA_real_, p = NA_real_, n = nrow(d_fit)))
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

  pythia_dat <- human_trials |>
    select(binom, participant, resp_alpha, RelFreq, OverallFreq) |>
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
      baseline <- glmer(
        resp_alpha ~ RelFreq + ovf + RelFreq:ovf + (1 | binom) + (1 | participant),
        family = binomial, data = d,
        control = glmerControl(optimizer = "bobyqa")
      )
      full <- glmer(
        resp_alpha ~ RelFreq + ovf + RelFreq:ovf + preference +
          (1 | binom) + (1 | participant),
        family = binomial, data = d,
        control = glmerControl(optimizer = "bobyqa")
      )
      tibble(delta_ll = as.numeric(logLik(full)) - as.numeric(logLik(baseline)),
             n_trials = nrow(d), n_items = n_distinct(d$binom))
    }) |>
    ungroup()

  pythia_pref_coef <- pythia_dat |>
    group_by(model, model_family, model_params, model_label) |>
    group_modify(function(d, k) {
      d <- filter(d, !is.na(preference))
      if (nrow(d) < 5) return(tibble(pref_coef = NA_real_))
      fit <- tryCatch(
        glmer(
          resp_alpha ~ preference + (1 | binom) + (1 | participant),
          family  = binomial, data = d,
          control = glmerControl(optimizer = "bobyqa")
        ),
        error = function(e) NULL
      )
      if (is.null(fit)) return(tibble(pref_coef = NA_real_))
      tibble(pref_coef = fixef(fit)[["preference"]])
    }) |>
    ungroup()

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
      fit <- tryCatch(
        lmer(preference ~ AbsPref_c + rf_c + ovf_c +
               AbsPref_c:ovf_c + rf_c:ovf_c + (1 | binom),
             data = d, REML = TRUE),
        error = function(e) lm(preference ~ AbsPref_c + rf_c + ovf_c +
                                 AbsPref_c:ovf_c + rf_c:ovf_c, data = d)
      )
      cf <- summary(fit)$coefficients
      if (!"AbsPref_c" %in% rownames(cf)) return(tibble(estimate = NA_real_, se = NA_real_,
                                                         t = NA_real_, p = NA_real_, n = nrow(d)))
      tibble(estimate = cf["AbsPref_c", "Estimate"],
             se       = cf["AbsPref_c", "Std. Error"],
             t        = cf["AbsPref_c", "t value"],
             p        = if ("Pr(>|t|)" %in% colnames(cf)) cf["AbsPref_c", "Pr(>|t|)"] else NA_real_,
             n        = nrow(d))
    }) |>
    ungroup()

  pythia_results <- pythia_delta_ll |>
    left_join(pythia_pref_coef |> select(model, pref_coef), by = "model") |>
    left_join(pythia_abspref_coef |> select(model, estimate, se, t, p), by = "model") |>
    left_join(ppl |> select(model, perplexity), by = "model") |>
    mutate(params_M = as.numeric(sub("M$", "", model_params)))

  message(sprintf("  Pythia: %d models (%d with AbsPref estimate)",
                  nrow(pythia_results), sum(!is.na(pythia_results$estimate))))
} else {
  message("training_attested.csv not found — Pythia points omitted.")
}

# ── BabyLM / C4 training-checkpoint families ─────────────────────────────────
# For each model, pick the step at index floor(n/3) as "early" and floor(n/2)
# as "mid" (skipping the first checkpoint in both cases).
ck_results <- NULL
if (file.exists(TRAIN_CSV)) {
  message("Building BabyLM/C4 checkpoint families ...")

  train_ck <- read_csv(TRAIN_CSV, show_col_types = FALSE) |>
    filter(grepl("babylm|opt-c4", model, ignore.case = TRUE)) |>
    filter(binom %in% attested_binoms)

  # Per model, identify the early and mid step numbers
  ck_map <- train_ck |>
    distinct(model, step, checkpoint) |>
    group_by(model) |>
    arrange(step) |>
    mutate(
      n     = n(),
      idx   = row_number() - 1L,        # 0-based index
      phase = case_when(
        idx == floor(n[1] / 3) ~ "early",
        idx == floor(n[1] / 2) ~ "mid",
        TRUE                   ~ NA_character_
      )
    ) |>
    filter(!is.na(phase)) |>
    select(model, step, checkpoint, phase) |>
    ungroup()

  # Base corpus tag (BabyLM / C4) and params for each model
  ck_map <- ck_map |>
    mutate(
      base_family  = if_else(grepl("babylm", model, TRUE), "BabyLM", "C4"),
      model_family = paste0(base_family, " (", phase, ")"),
      model_params = parse_params(model),
      model_label  = paste0(model_family, "-", model_params),
      # Key used to look up perplexity: model@checkpoint
      ppl_key      = paste0(model, "@", checkpoint)
    )

  ck_dat <- human_trials |>
    select(binom, participant, resp_alpha, RelFreq, OverallFreq) |>
    inner_join(
      train_ck |>
        inner_join(ck_map |> select(model, step, model_family, model_params,
                                    model_label, ppl_key),
                   by = c("model", "step")) |>
        select(model, model_family, model_params, model_label, ppl_key,
               binom, preference),
      by = "binom"
    )

  if (use_pile_freq) {
    ck_dat <- ck_dat |>
      left_join(pile_freq |> rename(freq_prob_c_pile = freq_prob_c), by = "binom")
  }

  ck_delta_ll <- ck_dat |>
    group_by(model, model_family, model_params, model_label, ppl_key) |>
    group_modify(function(d, k) {
      d <- choose_freq(d, k$model_family)
      d <- filter(d, !is.na(freq_use), !is.na(preference))
      if (nrow(d) < 5) return(tibble(delta_ll = NA_real_, n_trials = nrow(d), n_items = 0L))
      d <- d |> mutate(ovf = log(pmax(OverallFreq, 1)))
      baseline <- glmer(resp_alpha ~ freq_use + (1 | binom) + (1 | participant),
                        family = binomial, data = d,
                        control = glmerControl(optimizer = "bobyqa"))
      full     <- glmer(resp_alpha ~ freq_use + preference + (1 | binom) + (1 | participant),
                        family = binomial, data = d,
                        control = glmerControl(optimizer = "bobyqa"))
      tibble(delta_ll = as.numeric(logLik(full)) - as.numeric(logLik(baseline)),
             n_trials = nrow(d), n_items = n_distinct(d$binom))
    }) |>
    ungroup()

  ck_pref_coef <- ck_dat |>
    group_by(model, model_family, model_params, model_label, ppl_key) |>
    group_modify(function(d, k) {
      d <- filter(d, !is.na(preference))
      if (nrow(d) < 5) return(tibble(pref_coef = NA_real_))
      fit <- tryCatch(
        glmer(resp_alpha ~ preference + (1 | binom) + (1 | participant),
              family = binomial, data = d,
              control = glmerControl(optimizer = "bobyqa")),
        error = function(e) NULL
      )
      if (is.null(fit)) return(tibble(pref_coef = NA_real_))
      tibble(pref_coef = fixef(fit)[["preference"]])
    }) |>
    ungroup()

  ck_abspref <- train_ck |>
    inner_join(ck_map |> select(model, step, model_family, model_params, model_label, ppl_key),
               by = c("model", "step")) |>
    left_join(binom_items, by = "binom") |>
    group_by(model, model_family, model_params, model_label, ppl_key) |>
    group_modify(function(d, k) {
      if (use_pile_freq && "RelFreq_pile" %in% names(d)) {
        d$rf  <- coalesce(d$RelFreq_pile,     d$RelFreq_google)
        d$ovf <- coalesce(d$OverallFreq_pile, d$OverallFreq_google)
      } else {
        d$rf  <- d$RelFreq_google
        d$ovf <- d$OverallFreq_google
      }
      d <- filter(d, !is.na(AbsPref), !is.na(rf), !is.na(ovf), !is.na(preference))
      if (nrow(d) < 5) return(tibble(estimate = NA_real_, se = NA_real_,
                                     t = NA_real_, p = NA_real_, n = nrow(d)))
      d <- d |> mutate(
        AbsPref_c = as.numeric(scale(AbsPref, scale = FALSE)),
        rf_c      = as.numeric(scale(rf,      scale = FALSE)),
        ovf_c     = as.numeric(scale(ovf,     scale = FALSE))
      )
      fit <- tryCatch(
        lmer(preference ~ AbsPref_c + rf_c + ovf_c +
               AbsPref_c:ovf_c + rf_c:ovf_c + (1 | binom),
             data = d, REML = TRUE),
        error = function(e) lm(preference ~ AbsPref_c + rf_c + ovf_c +
                                 AbsPref_c:ovf_c + rf_c:ovf_c, data = d)
      )
      cf <- summary(fit)$coefficients
      if (!"AbsPref_c" %in% rownames(cf)) return(tibble(estimate = NA_real_, se = NA_real_,
                                                         t = NA_real_, p = NA_real_, n = nrow(d)))
      tibble(estimate = cf["AbsPref_c", "Estimate"],
             se       = cf["AbsPref_c", "Std. Error"],
             t        = cf["AbsPref_c", "t value"],
             p        = if ("Pr(>|t|)" %in% colnames(cf)) cf["AbsPref_c", "Pr(>|t|)"] else NA_real_,
             n        = nrow(d))
    }) |>
    ungroup()

  ck_results <- ck_delta_ll |>
    left_join(ck_pref_coef  |> select(model, model_family, ppl_key, pref_coef),
              by = c("model", "model_family", "ppl_key")) |>
    left_join(ck_abspref    |> select(model, model_family, ppl_key, estimate, se, t, p),
              by = c("model", "model_family", "ppl_key")) |>
    left_join(ppl |> select(model, perplexity), by = c("ppl_key" = "model")) |>
    filter(!is.na(perplexity)) |>
    mutate(params_M = as.numeric(sub("M$", "", model_params)))

  message(sprintf("  BabyLM/C4 checkpoints: %d rows retained with perplexity", nrow(ck_results)))
}

# ── Combine and join perplexity ───────────────────────────────────────────────
results <- delta_ll |>
  left_join(pref_coef_results |> select(model, pref_coef),
            by = "model") |>
  left_join(abspref_coef, by = c("model", "model_family", "model_params", "model_label")) |>
  left_join(ppl |> select(model, perplexity), by = "model") |>
  filter(!is.na(perplexity)) |>
  mutate(params_M = as.numeric(sub("M$", "", model_params)))

if (!is.null(pythia_results) && nrow(pythia_results) > 0) {
  results <- bind_rows(results, pythia_results |> filter(!is.na(perplexity)))
}
if (!is.null(ck_results) && nrow(ck_results) > 0) {
  results <- bind_rows(results, ck_results |> filter(!is.na(perplexity)))
}

# Warn about any remaining NA labels (should be 0 after fixes above)
na_label_rows <- results |> filter(is.na(params_M))
if (nrow(na_label_rows) > 0) {
  log_line("WARNING: rows with NA params_M (will show as NA on plot): ",
           paste(unique(na_label_rows$model), collapse = ", "))
}

results <- results |>
  mutate(
    model_family = factor(model_family,
                          levels = c("GPT-2", "GPT-Neo", "OPT",
                                     "OLMo-1", "OLMo-2", "OLMo-3",
                                     "Pythia", "BabyLM", "C4",
                                     "BabyLM (early)", "BabyLM (mid)",
                                     "C4 (early)", "C4 (mid)"))
  ) |>
  arrange(model_family, params_M)

message("\nResults:")
print(results |> select(model_family, model_label, model_params, perplexity,
                        delta_ll, pref_coef, estimate),
      n = Inf)

# ── Plot aesthetics ───────────────────────────────────────────────────────────
family_colours <- c(
  "GPT-2"          = "#2166ac",
  "GPT-Neo"        = "#4dac26",
  "OPT"            = "#b2182b",
  "OLMo-1"         = "#d6604d",
  "OLMo-2"         = "#f4a582",
  "OLMo-3"         = "#92c5de",
  "Pythia"         = "#762a83",
  "BabyLM"         = "#e08214",
  "BabyLM (early)" = "#fdbf6f",   # light orange
  "BabyLM (mid)"   = "#c87a0d",   # darker orange
  "C4"             = "#35978f",
  "C4 (early)"     = "#80cdc1",   # light teal
  "C4 (mid)"       = "#01665e"    # dark teal
)

shared_layers <- list(
  scale_x_continuous(
    "Validation perplexity on WikiText-2 (lower \u2192 better)",
    trans  = "log2",
    labels = scales::trans_format("log2", scales::math_format(2^.x))
  ),
  scale_colour_manual(values = family_colours, name = "Model family"),
  guides(colour = guide_legend(override.aes = list(size = 3))),
  theme_classic(base_size = 13),
  theme(
    panel.grid.major = element_line(colour = "grey92", linewidth = 0.4),
    axis.line        = element_line(colour = "grey40"),
    axis.ticks       = element_line(colour = "grey40"),
    legend.position  = "right"
  )
)

# ── Slope tests ───────────────────────────────────────────────────────────────
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

# ── Long-format results for faceting ─────────────────────────────────────────
results_long <- results |>
  pivot_longer(
    cols      = c(delta_ll, pref_coef, estimate),
    names_to  = "metric",
    values_to = "value"
  ) |>
  filter(!is.na(value)) |>
  mutate(
    metric = factor(metric,
                    levels = c("delta_ll", "pref_coef", "estimate"),
                    labels = c("\u0394LL", "Model Pref \u03b2", "AbsPref \u03b2"))
  )

# ── Polynomial smooth lines (no CI) for all families × metrics ───────────────
poly_smooth_all <- function(data_long, n_points = 200) {
  data_long |>
    group_by(metric, model_family) |>
    group_modify(function(d, k) {
      x_raw <- d$perplexity
      y_raw <- d$value
      ok    <- !is.na(x_raw) & !is.na(y_raw)
      n_ok  <- sum(ok)
      if (n_ok < 2) return(tibble())
      x_log <- log2(x_raw[ok])
      x_seq <- seq(min(x_log), max(x_log), length.out = n_points)
      deg     <- max(1L, min(2L, n_ok - 2L))
      fit_obj <- lm(y ~ poly(x, deg), data = data.frame(x = x_log, y = y_raw[ok]))
      pred    <- predict(fit_obj, newdata = data.frame(x = x_seq))
      tibble(perplexity = 2^x_seq, fit = pred)
    }) |>
    ungroup()
}

smooth_long <- poly_smooth_all(results_long)

# ── y-axis limits: data range only (no CI inflation) ─────────────────────────
y_limits <- results_long |>
  group_by(metric) |>
  summarise(
    ymin = min(value, na.rm = TRUE),
    ymax = max(value, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    pad  = (ymax - ymin) * 0.1,
    ymin = ymin - pad,
    ymax = ymax + pad
  )

# ── Reference lines (y = 0) only for pref / estimate rows ────────────────────
hline_df <- tibble(
  metric = factor(c("Model Pref \u03b2", "AbsPref \u03b2"),
                  levels = levels(results_long$metric))
)

# ── Size labels for all rows ──────────────────────────────────────────────────
labels_df <- results_long |>
  mutate(
    params_num = as.numeric(gsub("M$", "", model_params)),
    size_label = case_when(
      params_num >= 1000 ~ paste0(round(params_num / 1000, 1), "B"),
      TRUE               ~ paste0(params_num, "M")
    )
  )

# ── Faceted plot ──────────────────────────────────────────────────────────────
p_combined <- ggplot(results_long,
                     aes(x = perplexity, y = value, colour = model_family)) +
  # y = 0 reference
  geom_hline(data = hline_df, aes(yintercept = 0),
             linetype = "dotted", colour = "grey50", inherit.aes = FALSE) +
  # polynomial smooth lines (no CI ribbon)
  geom_line(
    data = smooth_long,
    aes(x = perplexity, y = fit, colour = model_family, group = model_family),
    linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE
  ) +
  # data points
  geom_point(size = 2.5, alpha = 0.9) +
  # size labels in ΔLL row only
  geom_text(
    data = labels_df,
    aes(label = size_label, colour = model_family),
    vjust = -0.7, hjust = 0.5,
    size = 2.6, fontface = "bold",
    show.legend = FALSE
  ) +
  # per-panel y limits based on data range only
  ggh4x::facetted_pos_scales(
    y = map2(
      y_limits$ymin, y_limits$ymax,
      ~ scale_y_continuous(limits = c(.x, .y))
    )
  ) +
  scale_x_continuous(
    "Validation perplexity (log\u2082 scale)",
    trans  = "log2",
    labels = scales::trans_format("log2", scales::math_format(2^.x))
  ) +
  scale_y_continuous(NULL) +
  scale_colour_manual(values = family_colours, name = "Model family") +
  guides(colour = guide_legend(override.aes = list(size = 3))) +
  facet_grid(metric ~ model_family, scales = "free_y") +
  theme_classic(base_size = 12) +
  theme(
    panel.grid.major  = element_line(colour = "grey92", linewidth = 0.4),
    axis.line         = element_line(colour = "grey40"),
    axis.ticks        = element_line(colour = "grey40"),
    strip.background  = element_rect(fill = "grey95", colour = "grey70"),
    strip.text        = element_text(face = "bold", size = 11),
    strip.text.y      = element_text(angle = 0),
    legend.position   = "bottom",
    panel.spacing.x   = unit(0.6, "lines"),
    panel.spacing.y   = unit(0.5, "lines")
  )

print(p_combined)

ggsave(OUT_PDF, p_combined, width = 22, height = 8)
ggsave(OUT_PNG, p_combined, width = 22, height = 8, dpi = 150)
message(sprintf("\nSaved:\n  %s\n  %s", OUT_PDF, OUT_PNG))