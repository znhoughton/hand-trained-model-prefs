#!/usr/bin/env Rscript
# prepare_results_new.R ───────────────────────────────────────────────────────
# Generates only the new CSVs added in the latest analysis round:
#   humfit_traj_coefs.csv      (Analysis 2a: human pref ~ model pref trajectory)
#   constr_vif_traj_coefs.csv  (Analysis 2g: VIF-filtered constraint trajectory)
#   pref_btraj_coefs.csv       (Analysis 2d: pref ~ genpref + bigram + freq trajectory)
#   sg_btraj_coefs.csv         (Analysis 2e: signed_gap ~ genpref + bigram + freq trajectory)
#   constraint_cor_traj.csv    (descriptive: constraint × preference correlations)
#
# Run with:  Rscript analysis-scripts/prepare_results_new.R
# ─────────────────────────────────────────────────────────────────────────────

library(tidyverse)
library(brms)
library(duckdb)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR <- if (interactive() && requireNamespace("rstudioapi", quietly = TRUE) &&
                  rstudioapi::isAvailable()) {
  dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
} else {
  args     <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0)
    dirname(normalizePath(sub("--file=", "", file_arg)))
  else
    normalizePath(".")
}

MODELS_DIR <- file.path(SCRIPT_DIR, "models")
OUT_DIR    <- normalizePath(file.path(SCRIPT_DIR, "../Data/processed/results"),
                            mustWork = FALSE)
CSV_PATH   <- normalizePath(file.path(SCRIPT_DIR, "../Data/processed/checkpoint_results_with_exposures.csv"),
                            mustWork = FALSE)

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Constants ─────────────────────────────────────────────────────────────────
N_TRAJ_CHECKPOINTS  <- 20
SIZE_LEVELS         <- c("125M", "350M", "1.3B")
CONSTRAINT_COLS     <- c("Form", "Percept", "Culture", "Power", "Intense",
                         "Icon", "Freq", "Len", "Lapse", "BStress")
CONSTRAINT_COLS_REDUCED <- setdiff(CONSTRAINT_COLS, c("Lapse", "Percept"))

# ── Helpers ───────────────────────────────────────────────────────────────────
parse_corpus <- function(x) {
  case_when(
    grepl("babylm", x, ignore.case = TRUE) ~ "BabyLM",
    grepl("opt.c4", x, ignore.case = TRUE) ~ "C4",
    TRUE ~ NA_character_
  )
}
parse_size <- function(x) {
  case_when(
    grepl("125m",    x, ignore.case = TRUE) ~ "125M",
    grepl("350m",    x, ignore.case = TRUE) ~ "350M",
    grepl("1[._]3b", x, ignore.case = TRUE) ~ "1.3B",
    TRUE ~ NA_character_
  )
}
normalise_terms <- function(term) {
  dplyr::recode(term,
    "tokens_c:genpref_c"      = "genpref_c:tokens_c",
    "tokens_c:freq_prob_c"    = "freq_prob_c:tokens_c",
    "tokens_c:log_total_c"    = "log_total_c:tokens_c",
    "log_total_c:genpref_c"   = "genpref_c:log_total_c",
    "log_total_c:freq_prob_c" = "freq_prob_c:log_total_c"
  )
}
extract_coefs <- function(fit, model_name, term_levels, term_labels) {
  fe <- as.data.frame(fixef(fit))
  fe$term <- normalise_terms(rownames(fe))
  fe |>
    as_tibble() |>
    rename(estimate = Estimate, se = Est.Error, lo = Q2.5, hi = Q97.5) |>
    filter(term != "Intercept") |>
    mutate(
      term       = factor(term, levels = term_levels, labels = term_labels),
      p_gt0      = pnorm(0, mean = estimate, sd = se, lower.tail = FALSE),
      model_name = model_name,
      corpus     = parse_corpus(model_name),
      model_size = factor(parse_size(model_name), levels = SIZE_LEVELS)
    )
}
list_fits <- function(prefix) {
  list.files(MODELS_DIR, pattern = paste0("^", prefix, ".+\\.rds$"), full.names = TRUE)
}
save_csv <- function(df, fname) {
  write_csv(
    df |> mutate(across(where(is.factor), as.character)),
    file.path(OUT_DIR, fname)
  )
  message(sprintf("  Saved %-35s  (%d rows)", fname, nrow(df)))
}

# ── TRAJ_META (needed for token-count x-axis in trajectory CSVs) ──────────────
message("Building TRAJ_META...")
if (file.exists(CSV_PATH)) {
  CSV_PATH_FWD <- gsub("\\\\", "/", normalizePath(CSV_PATH))
  con <- dbConnect(duckdb::duckdb())
  dbExecute(con, sprintf(
    "CREATE VIEW ckpt AS SELECT * FROM read_csv_auto('%s')", CSV_PATH_FWD
  ))
  models_meta <- sort(dbGetQuery(con, "SELECT DISTINCT model FROM ckpt")$model)
  TRAJ_META <- bind_rows(lapply(models_meta, function(m) {
    slug      <- tolower(gsub("[^a-z0-9]", "_", m))
    all_ckpts <- dbGetQuery(con, sprintf(
      "SELECT DISTINCT tokens FROM ckpt WHERE model = '%s' ORDER BY tokens",
      gsub("'", "''", m)))$tokens
    n   <- min(N_TRAJ_CHECKPOINTS, length(all_ckpts))
    idx <- unique(round(seq(1, length(all_ckpts), length.out = n)))
    tibble(slug = slug, ck_idx = seq_along(idx), tokens = all_ckpts[idx])
  }))
  dbDisconnect(con, shutdown = TRUE)
  message(sprintf("  %d rows", nrow(TRAJ_META)))
} else {
  message("  checkpoint_results_with_exposures.csv not found — token x-axis will be NA")
  TRAJ_META <- tibble(slug = character(), ck_idx = integer(), tokens = double())
}

# ── Trajectory extractor ──────────────────────────────────────────────────────
extract_traj_and_save <- function(prefix, term_levels, term_labels, fname) {
  paths <- list_fits(prefix)
  if (length(paths) == 0) {
    message(sprintf("  No .rds files for '%s'; skipping", prefix))
    return(invisible(NULL))
  }
  message(sprintf("Processing %d file(s) for '%s'...", length(paths), prefix))
  coefs <- map_dfr(paths, function(path) {
    fname_base      <- sub("\\.rds$", "", basename(path))
    fname_no_prefix <- sub(paste0("^", prefix), "", fname_base)
    slug    <- sub("_ck[0-9]+$", "", fname_no_prefix)
    ck_idx  <- as.integer(sub("^.*_ck([0-9]+)$", "\\1", fname_no_prefix))
    row     <- filter(TRAJ_META, .data$slug == !!slug, .data$ck_idx == !!ck_idx)
    ck_tokens <- if (nrow(row) == 1) row$tokens else NA_real_
    fit    <- readRDS(path)
    result <- extract_coefs(fit, slug, term_levels, term_labels) |>
      mutate(tokens = ck_tokens)
    rm(fit); gc()
    result
  })
  save_csv(coefs, fname)
}

# ── Run new exports ───────────────────────────────────────────────────────────

# 2a: human pref ~ model preference (logistic trajectory)
extract_traj_and_save(
  "humfit_traj_",
  term_levels = "preference",
  term_labels = "Model preference",
  fname       = "humfit_traj_coefs.csv"
)

# 2g: VIF-filtered constraint trajectory
TERM_LEVELS_VIF <- c(
  CONSTRAINT_COLS_REDUCED, "log_total_c", "freq_prob_c",
  paste0(CONSTRAINT_COLS_REDUCED, ":log_total_c"), "freq_prob_c:log_total_c"
)
TERM_LABELS_VIF <- c(
  CONSTRAINT_COLS_REDUCED,
  "Overall freq\n(ln total)", "RelFreq\n(P(alpha)\u22120.5)",
  paste0(CONSTRAINT_COLS_REDUCED, " \u00d7\nOverall freq"),
  "RelFreq \u00d7\nOverall freq"
)
extract_traj_and_save("constr_vif_traj_", TERM_LEVELS_VIF, TERM_LABELS_VIF,
                      "constr_vif_traj_coefs.csv")

# 2d/2e: pref/signed_gap + bigram log-odds trajectory
TERM_LEVELS_BTRAJ <- c(
  "genpref_c", "freq_prob_c", "bigram_logodds_c", "log_total_c",
  "genpref_c:log_total_c", "freq_prob_c:log_total_c"
)
TERM_LABELS_BTRAJ <- c(
  "GenPref", "RelFreq", "BigramLogOdds",
  "Overall freq\n(ln, z)", "GenPref \u00d7\nOverall freq", "RelFreq \u00d7\nOverall freq"
)
extract_traj_and_save("pref_btraj_", TERM_LEVELS_BTRAJ, TERM_LABELS_BTRAJ, "pref_btraj_coefs.csv")
extract_traj_and_save("sg_btraj_",   TERM_LEVELS_BTRAJ, TERM_LABELS_BTRAJ, "sg_btraj_coefs.csv")

# Constraint–preference correlation trajectories
message("Computing constraint-preference correlation trajectories...")
train_csv <- normalizePath(file.path(SCRIPT_DIR, "../Data/processed/training_attested.csv"),
                           mustWork = FALSE)
if (file.exists(train_csv)) {
  train_dat <- read_csv(train_csv, show_col_types = FALSE) |>
    mutate(
      corpus     = parse_corpus(model),
      model_size = factor(parse_size(model), levels = SIZE_LEVELS),
      pref_dev   = preference - qlogis(pmax(pmin(RelFreq, 1 - 1e-9), 1e-9))
    ) |>
    filter(!is.na(corpus), !is.na(model_size))

  sample_steps_pr <- function(steps_vec, n = N_TRAJ_CHECKPOINTS) {
    steps_vec <- sort(unique(steps_vec))
    idx <- unique(round(seq(1, length(steps_vec), length.out = min(n, length(steps_vec)))))
    steps_vec[idx]
  }

  cor_rows <- train_dat |>
    group_by(model, corpus, model_size) |>
    group_modify(function(mdat, key) {
      sampled_steps <- sample_steps_pr(mdat$step)
      map_dfr(sampled_steps, function(s) {
        ck_dat <- filter(mdat, step == s)
        tok    <- ck_dat$tokens[1]
        map_dfr(CONSTRAINT_COLS_REDUCED, function(constr) {
          x <- ck_dat[[constr]]
          tibble(
            tokens       = tok,
            constraint   = constr,
            cor_pref     = if (sum(!is.na(x) & !is.na(ck_dat$preference)) > 3)
                             cor(ck_dat$preference, x, use = "complete.obs") else NA_real_,
            cor_pref_dev = if (sum(!is.na(x) & !is.na(ck_dat$pref_dev)) > 3)
                             cor(ck_dat$pref_dev, x, use = "complete.obs") else NA_real_
          )
        })
      })
    }) |>
    ungroup()

  save_csv(cor_rows |> select(model, corpus, model_size, tokens, constraint,
                               cor_pref, cor_pref_dev),
           "constraint_cor_traj.csv")
} else {
  message("  training_attested.csv not found; skipping constraint_cor_traj.csv")
}

message("\nDone.")
