#!/usr/bin/env Rscript
# prepare_results.R ──────────────────────────────────────────────────────────
#
# Pre-computes all coefficient tables and trajectory summaries from fitted
# brms models, writing them to Data/processed/results/ as CSVs so that
# writeup.qmd can render without loading any .rds files or opening a DuckDB
# connection.
#
# Can be run from any working directory:
#   Rscript path/to/prepare_results.R
# or sourced interactively in RStudio with the script open.
#
# Outputs (all in ../Data/processed/results/ relative to this script):
#   traj_meta.csv          token counts for each checkpoint slug × ck_idx
#   pref_final_coefs.csv   preference model, final checkpoint
#   pref_traj_coefs.csv    preference model, 20-checkpoint trajectory
#   pref_ckpt_coefs.csv    preference model, checkpoint-interacted
#   sg_final_coefs.csv     signed-gap model, final checkpoint
#   sg_traj_coefs.csv      signed-gap model, 20-checkpoint trajectory
#   sg_ckpt_coefs.csv      signed-gap model, checkpoint-interacted
#   sg_gap_summary.csv     per-checkpoint mean |gap| and freq-tracking r
#   absgap_coefs.csv       absolute signed-gap model, checkpoint-interacted
# ─────────────────────────────────────────────────────────────────────────────

library(tidyverse)
library(brms)
library(duckdb)

# ── Paths (resolved relative to this script, not the working directory) ───────
SCRIPT_DIR <- if (interactive() && requireNamespace("rstudioapi", quietly = TRUE) &&
                  rstudioapi::isAvailable()) {
  dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
} else {
  args      <- commandArgs(trailingOnly = FALSE)
  file_arg  <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0)
    dirname(normalizePath(sub("--file=", "", file_arg)))
  else
    normalizePath(".")   # fallback: current directory
}

MODELS_DIR <- file.path(SCRIPT_DIR, "models")
OUT_DIR    <- normalizePath(file.path(SCRIPT_DIR, "../Data/processed/results"),
                            mustWork = FALSE)
CSV_PATH   <- normalizePath(file.path(SCRIPT_DIR, "../Data/processed/checkpoint_results_with_exposures.csv"),
                            mustWork = FALSE)

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Constants ─────────────────────────────────────────────────────────────────
N_TRAJ_CHECKPOINTS <- 20
SIZE_LEVELS <- c("125M", "350M", "1.3B")

TERM_LEVELS <- c("freq_prob_c", "genpref_c", "log_total_c",
                 "freq_prob_c:log_total_c", "genpref_c:log_total_c")
TERM_LABELS <- c("RelFreq\n(P(alpha)\u22120.5)", "GenPref",
                 "Overall freq\n(ln total)",
                 "RelFreq \u00d7\nOverall freq", "GenPref \u00d7\nOverall freq")

TERM_LEVELS_CK <- c(
  "genpref_c", "freq_prob_c", "log_total_c", "tokens_c",
  "genpref_c:tokens_c", "freq_prob_c:tokens_c", "log_total_c:tokens_c"
)
TERM_LABELS_CK <- c(
  "GenPref", "RelFreq\n(P(alpha)\u22120.5)", "Overall freq\n(ln total)",
  "Checkpoint\n(tokens, z)",
  "GenPref \u00d7\nCheckpoint", "RelFreq \u00d7\nCheckpoint",
  "Overall freq \u00d7\nCheckpoint"
)

TERM_LEVELS_ABSGAP <- c("tokens_c", "log_total_c", "log_total_c:tokens_c")
TERM_LABELS_ABSGAP <- c(
  "Checkpoint\n(tokens, z)",
  "Overall freq\n(ln total, z within ckpt)",
  "Checkpoint \u00d7\nOverall freq"
)

# ── Helper functions ──────────────────────────────────────────────────────────
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
  pat <- paste0("^", prefix, ".+\\.rds$")
  list.files(MODELS_DIR, pattern = pat, full.names = TRUE)
}
save_csv <- function(df, fname) {
  write_csv(
    df |> mutate(across(where(is.factor), as.character)),
    file.path(OUT_DIR, fname)
  )
  message(sprintf("  Saved %-30s  (%d rows)", fname, nrow(df)))
}

# ── TRAJ_META ─────────────────────────────────────────────────────────────────
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
  message("  checkpoint_results_with_exposures.csv not found; saving empty table")
  TRAJ_META <- tibble(slug = character(), ck_idx = integer(), tokens = double())
}
save_csv(TRAJ_META, "traj_meta.csv")

# ── Final / checkpoint-interacted coefficient tables ──────────────────────────
extract_and_save <- function(prefix, term_levels, term_labels, fname) {
  paths <- list_fits(prefix)
  if (length(paths) == 0) {
    message(sprintf("  No .rds files for '%s'; skipping", prefix))
    return(invisible(NULL))
  }
  message(sprintf("Processing %d file(s) for '%s'...", length(paths), prefix))
  coefs <- map_dfr(paths, function(path) {
    slug <- sub("\\.rds$", "", sub(paste0("^", prefix), "", basename(path)))
    fit  <- readRDS(path)
    result <- extract_coefs(fit, slug, term_levels, term_labels)
    rm(fit); gc()
    result
  })
  save_csv(coefs, fname)
}

# ── Trajectory coefficient tables (include token counts) ─────────────────────
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
    fit  <- readRDS(path)
    result <- extract_coefs(fit, slug, term_levels, term_labels) |>
      mutate(tokens = ck_tokens)
    rm(fit); gc()
    result
  })
  save_csv(coefs, fname)
}

# ── Gap summary ───────────────────────────────────────────────────────────────
save_gap_summary <- function(traj_prefix, fname) {
  paths <- list_fits(traj_prefix)
  if (length(paths) == 0) {
    message(sprintf("  No .rds files for '%s'; skipping", traj_prefix))
    return(invisible(NULL))
  }
  message(sprintf("Computing gap summary for '%s'...", traj_prefix))
  rows <- map_dfr(paths, function(path) {
    fname_base      <- sub("\\.rds$", "", basename(path))
    fname_no_prefix <- sub(paste0("^", traj_prefix), "", fname_base)
    slug    <- sub("_ck[0-9]+$", "", fname_no_prefix)
    ck_idx  <- as.integer(sub("^.*_ck([0-9]+)$", "\\1", fname_no_prefix))
    row     <- filter(TRAJ_META, .data$slug == !!slug, .data$ck_idx == !!ck_idx)
    ck_tokens <- if (nrow(row) == 1) row$tokens else NA_real_
    fit <- readRDS(path); d <- fit$data; rm(fit); gc()
    n    <- nrow(d)
    m    <- mean(d$signed_gap, na.rm = TRUE)
    m_ab <- mean(abs(d$signed_gap), na.rm = TRUE)
    s    <- sd(d$signed_gap, na.rm = TRUE)
    s_ab <- sd(abs(d$signed_gap), na.rm = TRUE)
    log_freq_ratio <- qlogis(d$freq_prob_c + 0.5)
    preference_rec <- d$signed_gap + log_freq_ratio
    tibble(
      slug = slug, ck_idx = ck_idx, tokens = ck_tokens,
      corpus     = parse_corpus(slug),
      model_size = parse_size(slug),
      n = n, mean = m, se = s / sqrt(n),
      mean_abs = m_ab, se_abs = s_ab / sqrt(n),
      cor_pref_freq = cor(preference_rec, log_freq_ratio, use = "complete.obs")
    ) |> mutate(
      lo95     = mean     - 1.96 * se,
      hi95     = mean     + 1.96 * se,
      lo95_abs = mean_abs - 1.96 * se_abs,
      hi95_abs = mean_abs + 1.96 * se_abs
    )
  })
  save_csv(rows, fname)
}

# ── Run ───────────────────────────────────────────────────────────────────────
extract_and_save("pref_final_",  TERM_LEVELS,        TERM_LABELS,        "pref_final_coefs.csv")
extract_and_save("pref_ckpt_",   TERM_LEVELS_CK,     TERM_LABELS_CK,     "pref_ckpt_coefs.csv")
extract_and_save("sg_final_",    TERM_LEVELS,        TERM_LABELS,        "sg_final_coefs.csv")
extract_and_save("sg_ckpt_",     TERM_LEVELS_CK,     TERM_LABELS_CK,     "sg_ckpt_coefs.csv")
extract_and_save("absgap_ckpt_", TERM_LEVELS_ABSGAP, TERM_LABELS_ABSGAP, "absgap_coefs.csv")

extract_traj_and_save("pref_traj_", TERM_LEVELS, TERM_LABELS, "pref_traj_coefs.csv")
extract_traj_and_save("sg_traj_",   TERM_LEVELS, TERM_LABELS, "sg_traj_coefs.csv")

save_gap_summary("sg_traj_", "sg_gap_summary.csv")

message("\nDone. All results saved to: ", normalizePath(OUT_DIR))
