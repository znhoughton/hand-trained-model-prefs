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
library(duckdb)
library(DBI)

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
EXPOSURES_CSV       <- file.path(BASE_DIR, "Data", "processed", "checkpoint_results_with_exposures.csv")
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

# ── Dynamic (checkpoint-specific) freq for final BabyLM / C4 models ──────────
# For each final BabyLM/C4 model, look up exposures at its final training step.
# This gives freq_prob_c_dyn = P(alpha) - 0.5 and log_total_dyn = log(total seen)
# computed from the actual corpus exposure counts at that checkpoint.
dyn_freq_final <- NULL
if (file.exists(EXPOSURES_CSV)) {
  babylm_c4_models <- model_prefs |>
    filter(grepl("babylm|opt-c4", model, ignore.case = TRUE)) |>
    distinct(model)

  if (nrow(babylm_c4_models) > 0) {
    con_dyn <- dbConnect(duckdb::duckdb())
    on.exit(dbDisconnect(con_dyn, shutdown = TRUE), add = TRUE)

    exp_fwd_dyn <- gsub("\\\\", "/",
                        if (file.exists(sub("\\.csv$", ".parquet", EXPOSURES_CSV)))
                          sub("\\.csv$", ".parquet", EXPOSURES_CSV)
                        else EXPOSURES_CSV)
    fmt_dyn <- if (grepl("\\.parquet$", exp_fwd_dyn)) "read_parquet" else "read_csv_auto"
    dbExecute(con_dyn, sprintf("CREATE VIEW exp_dyn AS SELECT * FROM %s('%s')",
                               fmt_dyn, exp_fwd_dyn))
    dbWriteTable(con_dyn, "bc_models", babylm_c4_models, overwrite = TRUE)

    # Find final step per model (max step in exposures file)
    final_steps <- dbGetQuery(con_dyn, "
      SELECT model, MAX(step) AS step
      FROM exp_dyn
      INNER JOIN bc_models USING (model)
      GROUP BY model
    ")
    dbWriteTable(con_dyn, "final_steps", final_steps, overwrite = TRUE)

    dyn_freq_final <- dbGetQuery(con_dyn, "
      SELECT e.model, e.binom,
             ANY_VALUE(e.alpha_seen) AS alpha_seen,
             ANY_VALUE(e.beta_seen)  AS beta_seen
      FROM exp_dyn e
      INNER JOIN final_steps f ON e.model = f.model AND e.step = f.step
      GROUP BY e.model, e.binom
    ") |>
      as_tibble() |>
      mutate(
        freq_prob_c_dyn = if_else(alpha_seen > 0 & beta_seen > 0,
                                  plogis(log(alpha_seen / beta_seen)) - 0.5,
                                  NA_real_),
        log_total_dyn   = if_else(alpha_seen + beta_seen > 0,
                                  log(alpha_seen + beta_seen),
                                  NA_real_)
      ) |>
      select(model, binom, freq_prob_c_dyn, log_total_dyn)

    message(sprintf("  Dynamic freq for final BabyLM/C4: %d model-binom rows", nrow(dyn_freq_final)))
    dbDisconnect(con_dyn, shutdown = TRUE)
  }
}

if (!is.null(dyn_freq_final)) {
  dat <- dat |>
    left_join(dyn_freq_final, by = c("model", "binom"))
}

# ── Frequency predictor selection ─────────────────────────────────────────────
# Use Pile RelFreq everywhere when available; fall back to Google Books RelFreq.
PILE_FREQ_FAMILIES <- c("OPT", "OLMo-1", "OLMo-2", "OLMo-3", "BabyLM", "C4")  # kept for reference

choose_freq <- function(d) {
  # Priority: dynamic checkpoint-specific > Pile > Google
  # Also sets ovf_raw (already log-transformed, will be z-scored later)
  if ("freq_prob_c_dyn" %in% names(d) && !all(is.na(d$freq_prob_c_dyn))) {
    d$freq_use <- d$freq_prob_c_dyn
    d$ovf_raw  <- if ("log_total_dyn" %in% names(d)) d$log_total_dyn
                  else log(pmax(d$OverallFreq, 1))
  } else if ("freq_prob_c_ck" %in% names(d) && !all(is.na(d$freq_prob_c_ck))) {
    d$freq_use <- d$freq_prob_c_ck
    d$ovf_raw  <- if ("log_total_ck" %in% names(d)) d$log_total_ck
                  else log(pmax(d$OverallFreq, 1))
  } else if (use_pile_freq && "freq_prob_c_pile" %in% names(d) &&
             !all(is.na(d$freq_prob_c_pile))) {
    d$freq_use <- d$freq_prob_c_pile
    d$ovf_raw  <- if ("log_total" %in% names(d) && !all(is.na(d$log_total)))
                    d$log_total
                  else log(pmax(d$OverallFreq, 1))
  } else {
    d$freq_use <- d$RelFreq
    d$ovf_raw  <- log(pmax(d$OverallFreq, 1))
  }
  d
}

# For abspref blocks: set rf/ovf columns (rf = RelFreq on -0.5..0.5 scale, ovf = log overall freq)
choose_rf <- function(d) {
  # Priority: dynamic checkpoint-specific > Pile > Google
  if ("freq_prob_c_dyn" %in% names(d) && !all(is.na(d$freq_prob_c_dyn))) {
    d$rf  <- d$freq_prob_c_dyn
    d$ovf <- if ("log_total_dyn" %in% names(d)) d$log_total_dyn
             else d$OverallFreq_google
  } else if ("freq_prob_c_ck" %in% names(d) && !all(is.na(d$freq_prob_c_ck))) {
    d$rf  <- d$freq_prob_c_ck
    d$ovf <- if ("log_total_ck" %in% names(d)) d$log_total_ck
             else d$OverallFreq_google
  } else if (use_pile_freq && "RelFreq_pile" %in% names(d) &&
             !all(is.na(d$RelFreq_pile))) {
    d$rf  <- d$RelFreq_pile
    d$ovf <- if ("OverallFreq_pile" %in% names(d)) d$OverallFreq_pile
             else d$OverallFreq_google
  } else {
    d$rf  <- d$RelFreq_google
    d$ovf <- d$OverallFreq_google
  }
  d
}

# ── Fit logistic regressions for ΔLL ─────────────────────────────────────────
message("Fitting ΔLL logistic regressions ...")
delta_ll <- dat |>
  group_by(model, model_family, model_params, model_label) |>
  group_modify(function(d, k) {
    d <- choose_freq(d)
    if (any(is.na(d$freq_use))) d <- filter(d, !is.na(freq_use))
    if (nrow(d) < 5) return(tibble(delta_ll = NA_real_, pref_coef = NA_real_,
                                   n_trials = nrow(d), n_items = NA_integer_))
    d <- d |> mutate(ovf_c = as.numeric(scale(ovf_raw)),
                     preference_s = as.numeric(scale(preference)))

    baseline_mod <- glmer(
      resp_alpha ~ freq_use + ovf_c + freq_use:ovf_c + (1 | binom) + (1 | participant),
      family = binomial, data = d,
      control = glmerControl(optimizer = "bobyqa")
    )
    full_mod <- glmer(
      resp_alpha ~ freq_use + ovf_c + freq_use:ovf_c + preference_s +
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
    d <- d |> mutate(preference_s = as.numeric(scale(preference)))
    fit <- tryCatch(
      glmer(
        resp_alpha ~ preference_s + (1 | binom) + (1 | participant),
        family  = binomial, data = d,
        control = glmerControl(optimizer = "bobyqa")
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) return(tibble(pref_coef = NA_real_))
    tibble(pref_coef = fixef(fit)[["preference_s"]])
  }) |>
  ungroup()

# ── Fit item-level mixed model for AbsPref coefficient ────────────────────────
message("Fitting AbsPref mixed models ...")

abspref_coef <- model_prefs |>
  group_by(model, model_family, model_params, model_label) |>
  group_modify(function(d, k) {
    this_model <- k$model
    na_row <- tibble(estimate = NA_real_, se = NA_real_, t = NA_real_,
                     p = NA_real_, relfreq_coef = NA_real_, n = NA_integer_)

    if (!has_by_prompt || !this_model %in% models_with_prompt)
      return(na_row)

    d_fit <- model_prefs_by_prompt |>
      filter(model == this_model) |>
      left_join(binom_items, by = "binom")
    # Join dynamic freq for BabyLM/C4 final models
    if (!is.null(dyn_freq_final) && this_model %in% dyn_freq_final$model) {
      d_fit <- d_fit |>
        left_join(dyn_freq_final |>
                    filter(model == this_model) |>
                    select(binom, freq_prob_c_dyn, log_total_dyn),
                  by = "binom")
    }
    d_fit <- choose_rf(d_fit)

    d_fit <- d_fit |> filter(!is.na(AbsPref), !is.na(rf), !is.na(ovf), !is.na(preference))
    if (nrow(d_fit) < 5) return(na_row |> mutate(n = nrow(d_fit)))
    d_fit <- d_fit |> mutate(
      AbsPref_c = AbsPref - 0.5,
      rf_c      = rf,
      ovf_c     = as.numeric(scale(ovf))
    )

    fit <- tryCatch(
      lmer(preference ~ AbsPref_c + rf_c + ovf_c +
             AbsPref_c:ovf_c + rf_c:ovf_c +
             (1 | binom) + (1 | prompt),
           data = d_fit, REML = TRUE),
      error = function(e) NULL
    )
    if (is.null(fit)) return(na_row)

    cf <- summary(fit)$coefficients
    if (!"AbsPref_c" %in% rownames(cf)) return(na_row)
    tibble(
      estimate     = cf["AbsPref_c", "Estimate"],
      se           = cf["AbsPref_c", "Std. Error"],
      t            = cf["AbsPref_c", "t value"],
      p            = if ("Pr(>|t|)" %in% colnames(cf)) cf["AbsPref_c", "Pr(>|t|)"] else NA_real_,
      relfreq_coef = if ("rf_c" %in% rownames(cf)) cf["rf_c", "Estimate"] else NA_real_,
      n            = nrow(d_fit)
    )
  }) |>
  ungroup()


# ── BabyLM / C4 training-checkpoint families ─────────────────────────────────
# For each model, pick the step at index floor(n/3) as "early" and floor(n/2)
# as "mid" (skipping the first checkpoint in both cases).
ck_results <- NULL
if (file.exists(TRAIN_CSV)) {
  message("Building BabyLM/C4 checkpoint families ...")

  con <- dbConnect(duckdb::duckdb())
  on.exit(dbDisconnect(con, shutdown = TRUE), add = TRUE)

  train_csv_fwd <- gsub("\\\\", "/", TRAIN_CSV)
  dbExecute(con, sprintf(
    "CREATE VIEW train_csv AS SELECT * FROM read_csv_auto('%s')", train_csv_fwd
  ))

  # Read only metadata columns to identify early/mid steps — tiny result
  ck_meta <- dbGetQuery(con, "
    SELECT DISTINCT model, step, checkpoint, tokens
    FROM train_csv
    WHERE regexp_matches(lower(model), 'babylm|opt-c4')
  ")

  # Per model, identify the early and mid step numbers by token count
  # (closest log-sampled checkpoint to 1/3 and 1/2 of total training tokens)
  ck_map <- ck_meta |>
    group_by(model) |>
    arrange(step) |>
    mutate(
      total_tokens = max(tokens),
      phase = case_when(
        tokens == tokens[which.min(abs(tokens - total_tokens / 3))] ~ "early",
        tokens == tokens[which.min(abs(tokens - total_tokens / 2))] ~ "mid",
        TRUE ~ NA_character_
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
      ppl_key      = paste0(model, "@", checkpoint)
    )

  # Push needed_steps and attested_binoms into DuckDB for server-side filtering
  needed_steps <- ck_map |> distinct(model, step)
  dbWriteTable(con, "needed_steps",   needed_steps,                         overwrite = TRUE)
  dbWriteTable(con, "attested_binoms_tbl", data.frame(binom = attested_binoms), overwrite = TRUE)

  # Pull only the rows we need — all heavy filtering happens inside DuckDB
  train_ck <- dbGetQuery(con, "
    SELECT t.*
    FROM train_csv t
    INNER JOIN needed_steps n    ON t.model = n.model AND t.step = n.step
    INNER JOIN attested_binoms_tbl b ON t.binom = b.binom
  ") |> as_tibble()
  message(sprintf("  train_ck: %d rows loaded", nrow(train_ck)))

  # Load checkpoint-specific relative frequency via DuckDB (uses parquet if available)
  ck_freq <- NULL
  if (file.exists(EXPOSURES_CSV)) {
    exp_fwd <- gsub("\\\\", "/",
                    if (file.exists(sub("\\.csv$", ".parquet", EXPOSURES_CSV)))
                      sub("\\.csv$", ".parquet", EXPOSURES_CSV)
                    else EXPOSURES_CSV)
    fmt <- if (grepl("\\.parquet$", exp_fwd)) "read_parquet" else "read_csv_auto"
    dbExecute(con, sprintf("CREATE VIEW exposures AS SELECT * FROM %s('%s')", fmt, exp_fwd))

    ck_freq <- dbGetQuery(con, "
      SELECT e.model, e.step, e.binom,
             ANY_VALUE(e.alpha_seen) AS alpha_seen,
             ANY_VALUE(e.beta_seen)  AS beta_seen
      FROM exposures e
      INNER JOIN needed_steps n ON e.model = n.model AND e.step = n.step
      WHERE e.alpha_seen IS NOT NULL AND e.beta_seen IS NOT NULL
      GROUP BY e.model, e.step, e.binom
    ") |>
      as_tibble() |>
      mutate(
        freq_prob_c_ck = if_else(alpha_seen > 0 & beta_seen > 0,
                                 plogis(log(alpha_seen / beta_seen)) - 0.5,
                                 NA_real_),
        log_total_ck   = if_else(alpha_seen + beta_seen > 0,
                                 log(alpha_seen + beta_seen),
                                 NA_real_)
      ) |>
      select(model, step, binom, freq_prob_c_ck, log_total_ck)
    message(sprintf("  Loaded checkpoint-specific freq: %d model-step-binom rows", nrow(ck_freq)))
  }

  ck_dat <- human_trials |>
    select(binom, participant, resp_alpha, RelFreq, OverallFreq) |>
    inner_join(
      train_ck |>
        inner_join(ck_map |> select(model, step, model_family, model_params,
                                    model_label, ppl_key),
                   by = c("model", "step")) |>
        select(model, step, model_family, model_params, model_label, ppl_key,
               binom, preference),
      by = "binom"
    )

  if (use_pile_freq) {
    ck_dat <- ck_dat |>
      left_join(pile_freq |> rename(freq_prob_c_pile = freq_prob_c), by = "binom")
  }

  if (!is.null(ck_freq)) {
    ck_dat <- ck_dat |>
      left_join(ck_freq, by = c("model", "step", "binom"))
  }

  ck_delta_ll <- ck_dat |>
    group_by(model, model_family, model_params, model_label, ppl_key) |>
    group_modify(function(d, k) {
      d <- choose_freq(d)
      d <- filter(d, !is.na(freq_use), !is.na(preference))
      if (nrow(d) < 5) return(tibble(delta_ll = NA_real_, n_trials = nrow(d), n_items = 0L))
      d <- d |> mutate(ovf_c = as.numeric(scale(ovf_raw)),
                       preference_s = as.numeric(scale(preference)))
      baseline <- glmer(resp_alpha ~ freq_use + ovf_c + freq_use:ovf_c + (1 | binom) + (1 | participant),
                        family = binomial, data = d,
                        control = glmerControl(optimizer = "bobyqa"))
      full     <- glmer(resp_alpha ~ freq_use + ovf_c + freq_use:ovf_c + preference_s + (1 | binom) + (1 | participant),
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
      d <- d |> mutate(preference_s = as.numeric(scale(preference)))
      fit <- tryCatch(
        glmer(resp_alpha ~ preference_s + (1 | binom) + (1 | participant),
              family = binomial, data = d,
              control = glmerControl(optimizer = "bobyqa")),
        error = function(e) NULL
      )
      if (is.null(fit)) return(tibble(pref_coef = NA_real_))
      tibble(pref_coef = fixef(fit)[["preference_s"]])
    }) |>
    ungroup()

  ck_abspref_base <- train_ck |>
    inner_join(ck_map |> select(model, step, model_family, model_params, model_label, ppl_key),
               by = c("model", "step")) |>
    left_join(binom_items, by = "binom")
  if (!is.null(ck_freq)) {
    ck_abspref_base <- ck_abspref_base |>
      left_join(ck_freq, by = c("model", "step", "binom"))
  }
  ck_abspref <- ck_abspref_base |>
    group_by(model, model_family, model_params, model_label, ppl_key) |>
    group_modify(function(d, k) {
      d <- choose_rf(d)
      d <- filter(d, !is.na(AbsPref), !is.na(rf), !is.na(ovf), !is.na(preference))
      if (nrow(d) < 5) return(tibble(estimate = NA_real_, se = NA_real_,
                                     t = NA_real_, p = NA_real_,
                                     relfreq_coef = NA_real_, n = nrow(d)))
      d <- d |> mutate(
        AbsPref_c = AbsPref - 0.5,
        rf_c      = rf,
        ovf_c     = as.numeric(scale(ovf))
      )
      fit <- tryCatch(
        lmer(preference ~ AbsPref_c + rf_c + ovf_c +
               AbsPref_c:ovf_c + rf_c:ovf_c + (1 | binom),
             data = d, REML = TRUE),
        error = function(e) NULL
      )
      if (is.null(fit)) return(tibble(estimate = NA_real_, se = NA_real_, t = NA_real_,
                                      p = NA_real_, relfreq_coef = NA_real_, n = nrow(d)))
      cf <- summary(fit)$coefficients
      if (!"AbsPref_c" %in% rownames(cf)) return(tibble(estimate = NA_real_, se = NA_real_,
                                                         t = NA_real_, p = NA_real_,
                                                         relfreq_coef = NA_real_, n = nrow(d)))
      tibble(estimate     = cf["AbsPref_c", "Estimate"],
             se           = cf["AbsPref_c", "Std. Error"],
             t            = cf["AbsPref_c", "t value"],
             p            = if ("Pr(>|t|)" %in% colnames(cf)) cf["AbsPref_c", "Pr(>|t|)"] else NA_real_,
             relfreq_coef = if ("rf_c" %in% rownames(cf)) cf["rf_c", "Estimate"] else NA_real_,
             n            = nrow(d))
    }) |>
    ungroup()

  ck_results <- ck_delta_ll |>
    left_join(ck_pref_coef  |> select(model, model_family, ppl_key, pref_coef),
              by = c("model", "model_family", "ppl_key")) |>
    left_join(ck_abspref    |> select(model, model_family, ppl_key, estimate, se, t, p, relfreq_coef),
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
                                     "OLMo-1", "OLMo-1 (10B tokens)", "OLMo-1 (50B tokens)", "OLMo-1 (early)", "OLMo-1 (early-mid)", "OLMo-1 (mid)", "OLMo-1 (mid-late)",
                                     "OLMo-2", "OLMo-2 (10B tokens)", "OLMo-2 (50B tokens)", "OLMo-2 (early)", "OLMo-2 (early-mid)", "OLMo-2 (mid)", "OLMo-2 (mid-late)",
                                     "OLMo-3", "OLMo-3 (10B tokens)", "OLMo-3 (50B tokens)", "OLMo-3 (early)", "OLMo-3 (early-mid)", "OLMo-3 (mid)", "OLMo-3 (mid-late)",
                                     "BabyLM", "C4",
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
  "GPT-2"                = "#2166ac",
  "GPT-Neo"              = "#4dac26",
  "OPT"                  = "#b2182b",
  # OLMo-1 gradient: very light → dark red (#d6604d)
  "OLMo-1 (10B tokens)"  = "#feede9",
  "OLMo-1 (50B tokens)"  = "#fcd7cf",
  "OLMo-1 (early)"       = "#f5a99d",
  "OLMo-1 (early-mid)"   = "#e87e72",
  "OLMo-1 (mid)"         = "#de6a5e",
  "OLMo-1 (mid-late)"    = "#da6458",
  "OLMo-1"               = "#d6604d",
  # OLMo-2 gradient: very light → medium orange (#f4a582)
  "OLMo-2 (10B tokens)"  = "#fff6f1",
  "OLMo-2 (50B tokens)"  = "#fee8d8",
  "OLMo-2 (early)"       = "#fdd5bc",
  "OLMo-2 (early-mid)"   = "#fabf9c",
  "OLMo-2 (mid)"         = "#f7b28c",
  "OLMo-2 (mid-late)"    = "#f6ad88",
  "OLMo-2"               = "#f4a582",
  # OLMo-3 gradient: very light → steel blue (#92c5de)
  "OLMo-3 (10B tokens)"  = "#f0f9fd",
  "OLMo-3 (50B tokens)"  = "#d8eef7",
  "OLMo-3 (early)"       = "#bce0f0",
  "OLMo-3 (early-mid)"   = "#a9d5e9",
  "OLMo-3 (mid)"         = "#9dcde4",
  "OLMo-3 (mid-late)"    = "#9dcbe3",
  "OLMo-3"               = "#92c5de",
  "BabyLM"              = "#238443",   # dark green
  "BabyLM (early)"      = "#d9f0a3",   # light green
  "BabyLM (mid)"        = "#78c679",   # medium green
  "C4"                  = "#980043",   # dark purple-red
  "C4 (early)"          = "#d4b9da",   # light purple
  "C4 (mid)"            = "#df65b0"    # medium pink-purple
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
FACET_LEVELS <- c("GPT-2", "GPT-Neo", "OPT",
                  "OLMo-1", "OLMo-2", "OLMo-3",
                  "BabyLM", "C4")

# Collapse checkpoint sub-families to their parent for facet column assignment
to_facet_family <- function(fam) {
  dplyr::case_when(
    grepl("^OLMo-1 \\(", fam) ~ "OLMo-1",
    grepl("^OLMo-2 \\(", fam) ~ "OLMo-2",
    grepl("^OLMo-3 \\(", fam) ~ "OLMo-3",
    fam %in% c("BabyLM (early)", "BabyLM (mid)") ~ "BabyLM",
    fam %in% c("C4 (early)",     "C4 (mid)")     ~ "C4",
    TRUE ~ as.character(fam)
  )
}

results_long <- results |>
  pivot_longer(
    cols      = c(delta_ll, pref_coef, estimate, relfreq_coef),
    names_to  = "metric",
    values_to = "value"
  ) |>
  filter(!is.na(value)) |>
  mutate(
    metric = factor(metric,
                    levels = c("delta_ll", "pref_coef", "estimate", "relfreq_coef"),
                    labels = c("\u0394LL", "Model Pref \u03b2", "AbsPref \u03b2", "RelFreq \u03b2")),
    # Collapse early/mid into the parent family column for faceting
    facet_family = factor(to_facet_family(as.character(model_family)), levels = FACET_LEVELS)
  )

# ── Polynomial smooth lines (no CI) for all families × metrics ───────────────
poly_smooth_all <- function(data_long, n_points = 200) {
  data_long |>
    group_by(metric, model_family, facet_family) |>
    group_modify(function(d, k) {
      x_raw <- d$perplexity
      y_raw <- d$value
      ok    <- !is.na(x_raw) & !is.na(y_raw) & is.finite(x_raw) & x_raw > 0
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
    range = ymax - ymin,
    ymin  = ymin - range * 0.15,
    ymax  = ymax + range * 0.35    # extra top room for size-label text
  ) |>
  select(-range)

# ── Reference lines (y = 0) only for pref / estimate rows ────────────────────
hline_df <- expand.grid(
  metric       = factor(c("Model Pref \u03b2", "AbsPref \u03b2", "RelFreq \u03b2"),
                        levels = levels(results_long$metric)),
  facet_family = factor(FACET_LEVELS, levels = FACET_LEVELS),
  stringsAsFactors = FALSE
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
# facet_family already present via results_long

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
  facet_grid(metric ~ facet_family, scales = "free_y") +
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

ggsave(OUT_PDF, p_combined, width = 22, height = 12)
ggsave(OUT_PNG, p_combined, width = 22, height = 12, dpi = 150)
message(sprintf("\nSaved:\n  %s\n  %s", OUT_PDF, OUT_PNG))

# ══════════════════════════════════════════════════════════════════════════════
# Frequency-binned plots
# ══════════════════════════════════════════════════════════════════════════════
message("\n\nBuilding frequency-binned plots ...")

# Tertile bins based on log overall frequency (OverallFreq_google = log(OverallFreq))
freq_bins <- binom_items |>
  filter(!is.na(OverallFreq_google)) |>
  mutate(freq_bin = ntile(OverallFreq_google, 3),
         freq_bin_label = factor(freq_bin,
                                 labels = c("Low frequency", "Mid frequency", "High frequency"))) |>
  select(binom, freq_bin_label)

message(sprintf("  Binomial counts per bin: %s",
                paste(names(table(freq_bins$freq_bin_label)),
                      table(freq_bins$freq_bin_label), sep = " = ", collapse = ", ")))

# ── Helper: refit all regressions for a binom subset ─────────────────────────
compute_results_for_binoms <- function(binoms_sub) {
  d  <- dat         |> filter(binom %in% binoms_sub)
  bi <- binom_items |> filter(binom %in% binoms_sub)
  mp <- model_prefs |> filter(binom %in% binoms_sub)

  # delta_ll
  dll <- d |>
    group_by(model, model_family, model_params, model_label) |>
    group_modify(function(d, k) {
      d <- choose_freq(d)
      if (any(is.na(d$freq_use))) d <- filter(d, !is.na(freq_use))
      if (nrow(d) < 5) return(tibble(delta_ll = NA_real_, n_trials = nrow(d), n_items = NA_integer_))
      d <- d |> mutate(ovf_c = as.numeric(scale(ovf_raw)),
                       preference_s = as.numeric(scale(preference)))
      baseline_mod <- glmer(resp_alpha ~ freq_use + ovf_c + freq_use:ovf_c + (1 | binom) + (1 | participant),
                            family = binomial, data = d, control = glmerControl(optimizer = "bobyqa"))
      full_mod <- glmer(resp_alpha ~ freq_use + ovf_c + freq_use:ovf_c + preference_s + (1 | binom) + (1 | participant),
                        family = binomial, data = d, control = glmerControl(optimizer = "bobyqa"))
      tibble(delta_ll = as.numeric(logLik(full_mod)) - as.numeric(logLik(baseline_mod)),
             n_trials = nrow(d), n_items = n_distinct(d$binom))
    }) |>
    ungroup()

  # pref_coef
  pc <- d |>
    group_by(model, model_family, model_params, model_label) |>
    group_modify(function(d, k) {
      d <- filter(d, !is.na(preference))
      if (nrow(d) < 5) return(tibble(pref_coef = NA_real_))
      d <- d |> mutate(preference_s = as.numeric(scale(preference)))
      fit <- tryCatch(
        glmer(resp_alpha ~ preference_s + (1 | binom) + (1 | participant),
              family = binomial, data = d, control = glmerControl(optimizer = "bobyqa")),
        error = function(e) NULL
      )
      if (is.null(fit)) return(tibble(pref_coef = NA_real_))
      tibble(pref_coef = fixef(fit)[["preference_s"]])
    }) |>
    ungroup()

  # abspref
  ac <- mp |>
    group_by(model, model_family, model_params, model_label) |>
    group_modify(function(d, k) {
      this_model <- k$model
      na_row <- tibble(estimate = NA_real_, se = NA_real_, t = NA_real_,
                       p = NA_real_, relfreq_coef = NA_real_, n = NA_integer_)

      if (!has_by_prompt || !this_model %in% models_with_prompt)
        return(na_row)

      d_fit <- model_prefs_by_prompt |>
        filter(model == this_model, binom %in% binoms_sub) |>
        left_join(bi, by = "binom")
      if (!is.null(dyn_freq_final) && this_model %in% dyn_freq_final$model) {
        d_fit <- d_fit |>
          left_join(dyn_freq_final |>
                      filter(model == this_model) |>
                      select(binom, freq_prob_c_dyn, log_total_dyn),
                    by = "binom")
      }
      d_fit <- choose_rf(d_fit)

      d_fit <- d_fit |> filter(!is.na(AbsPref), !is.na(rf), !is.na(ovf), !is.na(preference))
      if (nrow(d_fit) < 5) return(na_row |> mutate(n = nrow(d_fit)))
      d_fit <- d_fit |> mutate(
        AbsPref_c = AbsPref - 0.5,
        rf_c      = rf,
        ovf_c     = as.numeric(scale(ovf))
      )
      fit <- tryCatch(
        lmer(preference ~ AbsPref_c + rf_c + ovf_c + AbsPref_c:ovf_c + rf_c:ovf_c +
               (1 | binom) + (1 | prompt), data = d_fit, REML = TRUE),
        error = function(e) NULL
      )
      if (is.null(fit)) return(na_row)
      cf <- summary(fit)$coefficients
      if (!"AbsPref_c" %in% rownames(cf)) return(na_row)
      tibble(estimate     = cf["AbsPref_c", "Estimate"],
             se           = cf["AbsPref_c", "Std. Error"],
             t            = cf["AbsPref_c", "t value"],
             p            = if ("Pr(>|t|)" %in% colnames(cf)) cf["AbsPref_c", "Pr(>|t|)"] else NA_real_,
             relfreq_coef = if ("rf_c" %in% rownames(cf)) cf["rf_c", "Estimate"] else NA_real_,
             n            = nrow(d_fit))
    }) |>
    ungroup()

  # Combine main model results
  r <- dll |>
    left_join(pc |> select(model, pref_coef), by = "model") |>
    left_join(ac |> select(model, model_family, model_params, model_label,
                            estimate, se, t, p, relfreq_coef),
              by = c("model", "model_family", "model_params", "model_label")) |>
    left_join(ppl |> select(model, perplexity), by = "model") |>
    filter(!is.na(perplexity)) |>
    mutate(params_M = as.numeric(sub("M$", "", model_params)))

  # Checkpoints
  if (exists("ck_dat") && !is.null(ck_dat)) {
    ck_dat_b <- ck_dat   |> filter(binom %in% binoms_sub)
    cf_b     <- if (!is.null(ck_freq)) ck_freq |> filter(binom %in% binoms_sub) else NULL

    ck_dll_b <- ck_dat_b |>
      group_by(model, model_family, model_params, model_label, ppl_key) |>
      group_modify(function(d, k) {
        d <- choose_freq(d)
        d <- filter(d, !is.na(freq_use), !is.na(ovf_raw), !is.na(preference))
        if (nrow(d) < 5) return(tibble(delta_ll = NA_real_, n_trials = nrow(d), n_items = 0L))
        d <- d |> mutate(ovf_c = as.numeric(scale(ovf_raw)),
                         preference_s = as.numeric(scale(preference)))
        baseline <- glmer(resp_alpha ~ freq_use + ovf_c + freq_use:ovf_c + (1 | binom) + (1 | participant),
                          family = binomial, data = d, control = glmerControl(optimizer = "bobyqa"))
        full     <- glmer(resp_alpha ~ freq_use + ovf_c + freq_use:ovf_c + preference_s + (1 | binom) + (1 | participant),
                          family = binomial, data = d, control = glmerControl(optimizer = "bobyqa"))
        tibble(delta_ll = as.numeric(logLik(full)) - as.numeric(logLik(baseline)),
               n_trials = nrow(d), n_items = n_distinct(d$binom))
      }) |>
      ungroup()

    ck_pc_b <- ck_dat_b |>
      group_by(model, model_family, model_params, model_label, ppl_key) |>
      group_modify(function(d, k) {
        d <- filter(d, !is.na(preference))
        if (nrow(d) < 5) return(tibble(pref_coef = NA_real_))
        d <- d |> mutate(preference_s = as.numeric(scale(preference)))
        fit <- tryCatch(
          glmer(resp_alpha ~ preference_s + (1 | binom) + (1 | participant),
                family = binomial, data = d, control = glmerControl(optimizer = "bobyqa")),
          error = function(e) NULL
        )
        if (is.null(fit)) return(tibble(pref_coef = NA_real_))
        tibble(pref_coef = fixef(fit)[["preference_s"]])
      }) |>
      ungroup()

    cab_b <- train_ck |>
      filter(binom %in% binoms_sub) |>
      inner_join(ck_map |> select(model, step, model_family, model_params, model_label, ppl_key),
                 by = c("model", "step")) |>
      left_join(bi, by = "binom")
    if (!is.null(cf_b)) cab_b <- cab_b |> left_join(cf_b, by = c("model", "step", "binom"))

    ck_ac_b <- cab_b |>
      group_by(model, model_family, model_params, model_label, ppl_key) |>
      group_modify(function(d, k) {
        d <- choose_rf(d)
        d <- filter(d, !is.na(AbsPref), !is.na(rf), !is.na(ovf), !is.na(preference))
        if (nrow(d) < 5) return(tibble(estimate = NA_real_, se = NA_real_,
                                       t = NA_real_, p = NA_real_,
                                       relfreq_coef = NA_real_, n = nrow(d)))
        d <- d |> mutate(AbsPref_c = AbsPref - 0.5,
                         rf_c      = rf,
                         ovf_c     = as.numeric(scale(ovf)))
        fit <- tryCatch(
          lmer(preference ~ AbsPref_c + rf_c + ovf_c + AbsPref_c:ovf_c + rf_c:ovf_c +
                 (1 | binom), data = d, REML = TRUE),
          error = function(e) NULL
        )
        if (is.null(fit)) return(tibble(estimate = NA_real_, se = NA_real_, t = NA_real_,
                                        p = NA_real_, relfreq_coef = NA_real_, n = nrow(d)))
        cf <- summary(fit)$coefficients
        if (!"AbsPref_c" %in% rownames(cf)) return(tibble(estimate = NA_real_, se = NA_real_,
                                                           t = NA_real_, p = NA_real_,
                                                           relfreq_coef = NA_real_, n = nrow(d)))
        tibble(estimate     = cf["AbsPref_c", "Estimate"],
               se           = cf["AbsPref_c", "Std. Error"],
               t            = cf["AbsPref_c", "t value"],
               p            = if ("Pr(>|t|)" %in% colnames(cf)) cf["AbsPref_c", "Pr(>|t|)"] else NA_real_,
               relfreq_coef = if ("rf_c" %in% rownames(cf)) cf["rf_c", "Estimate"] else NA_real_,
               n            = nrow(d))
      }) |>
      ungroup()

    ck_res_b <- ck_dll_b |>
      left_join(ck_pc_b |> select(model, model_family, ppl_key, pref_coef),
                by = c("model", "model_family", "ppl_key")) |>
      left_join(ck_ac_b |> select(model, model_family, ppl_key, estimate, se, t, p, relfreq_coef),
                by = c("model", "model_family", "ppl_key")) |>
      left_join(ppl |> select(model, perplexity), by = c("ppl_key" = "model")) |>
      filter(!is.na(perplexity)) |>
      mutate(params_M = as.numeric(sub("M$", "", model_params)))

    r <- bind_rows(r, ck_res_b)
  }

  r |>
    mutate(model_family = factor(model_family,
                                 levels = c("GPT-2", "GPT-Neo", "OPT",
                                            "OLMo-1", "OLMo-2", "OLMo-3",
                                            "BabyLM", "C4",
                                            "BabyLM (early)", "BabyLM (mid)",
                                            "C4 (early)", "C4 (mid)"))) |>
    arrange(model_family, params_M)
}

# ── Helper: build faceted plot from any results tibble ───────────────────────
make_faceted_plot <- function(res, subtitle) {
  res_long <- res |>
    pivot_longer(cols = c(delta_ll, pref_coef, estimate, relfreq_coef),
                 names_to = "metric", values_to = "value") |>
    filter(!is.na(value)) |>
    mutate(
      metric = factor(metric,
                      levels = c("delta_ll", "pref_coef", "estimate", "relfreq_coef"),
                      labels = c("\u0394LL", "Model Pref \u03b2", "AbsPref \u03b2", "RelFreq \u03b2")),
      facet_family = case_when(
        model_family %in% c("BabyLM (early)", "BabyLM (mid)") ~ "BabyLM",
        model_family %in% c("C4 (early)",     "C4 (mid)")     ~ "C4",
        TRUE ~ as.character(model_family)
      ),
      facet_family = factor(facet_family, levels = FACET_LEVELS)
    )

  smooth_b <- poly_smooth_all(res_long)

  y_lim_b <- res_long |>
    group_by(metric) |>
    summarise(ymin = min(value, na.rm = TRUE), ymax = max(value, na.rm = TRUE), .groups = "drop") |>
    mutate(range = ymax - ymin, ymin = ymin - range * 0.15, ymax = ymax + range * 0.35) |>
    select(-range)

  hline_b <- expand.grid(
    metric       = factor(c("Model Pref \u03b2", "AbsPref \u03b2", "RelFreq \u03b2"),
                          levels = levels(res_long$metric)),
    facet_family = factor(FACET_LEVELS, levels = FACET_LEVELS),
    stringsAsFactors = FALSE
  )

  labels_b <- res_long |>
    mutate(params_num = as.numeric(gsub("M$", "", model_params)),
           size_label = case_when(params_num >= 1000 ~ paste0(round(params_num / 1000, 1), "B"),
                                  TRUE               ~ paste0(params_num, "M")))

  ggplot(res_long, aes(x = perplexity, y = value, colour = model_family)) +
    geom_hline(data = hline_b, aes(yintercept = 0),
               linetype = "dotted", colour = "grey50", inherit.aes = FALSE) +
    geom_line(data = smooth_b,
              aes(x = perplexity, y = fit, colour = model_family, group = model_family),
              linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE) +
    geom_point(size = 2.5, alpha = 0.9) +
    geom_text(data = labels_b,
              aes(label = size_label, colour = model_family),
              vjust = -0.7, hjust = 0.5, size = 2.6, fontface = "bold", show.legend = FALSE) +
    ggh4x::facetted_pos_scales(
      y = map2(y_lim_b$ymin, y_lim_b$ymax, ~ scale_y_continuous(limits = c(.x, .y)))
    ) +
    scale_x_continuous("Validation perplexity (log\u2082 scale)", trans = "log2",
                       labels = scales::trans_format("log2", scales::math_format(2^.x))) +
    scale_y_continuous(NULL) +
    scale_colour_manual(values = family_colours, name = "Model family") +
    guides(colour = guide_legend(override.aes = list(size = 3))) +
    facet_grid(metric ~ facet_family, scales = "free_y") +
    ggtitle(subtitle) +
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
      panel.spacing.y   = unit(0.5, "lines"),
      plot.title        = element_text(face = "bold", hjust = 0.5)
    )
}

# ── Run binned analyses and save plots ────────────────────────────────────────
BIN_LABELS   <- c("Low frequency", "Mid frequency", "High frequency")
BIN_SUFFIXES <- c("low", "mid", "high")

for (i in seq_along(BIN_LABELS)) {
  bin_label  <- BIN_LABELS[[i]]
  bin_suffix <- BIN_SUFFIXES[[i]]
  message(sprintf("\n── %s binomials ──", bin_label))

  bin_binoms <- freq_bins |> filter(freq_bin_label == bin_label) |> pull(binom)
  message(sprintf("  %d binomials", length(bin_binoms)))

  res_b <- suppressWarnings(compute_results_for_binoms(bin_binoms))
  p_b   <- make_faceted_plot(res_b, subtitle = bin_label)

  out_pdf_b <- sub("\\.pdf$", paste0("_freq_", bin_suffix, ".pdf"), OUT_PDF)
  out_png_b <- sub("\\.png$", paste0("_freq_", bin_suffix, ".png"), OUT_PNG)
  ggsave(out_pdf_b, p_b, width = 22, height = 12)
  ggsave(out_png_b, p_b, width = 22, height = 12, dpi = 150)
  message(sprintf("  Saved: %s", out_pdf_b))
}

message("\nAll frequency-binned plots complete.")