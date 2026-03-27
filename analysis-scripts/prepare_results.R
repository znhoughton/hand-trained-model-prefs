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
#   absgap_coefs.csv       absolute signed-gap model, checkpoint-interacted (Gaussian)
#   absgap_ln_coefs.csv    absolute signed-gap model, checkpoint-interacted (log-normal)
# ─────────────────────────────────────────────────────────────────────────────

library(tidyverse)
library(brms)
library(duckdb)
library(ggplot2)
library(bayesplot)
library(furrr)
library(future)
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
  }, .progress = sprintf("  %s", prefix))
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
  }, .progress = sprintf("  %s", prefix))
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
  rows <- map_dfr(paths, .progress = sprintf("  %s gap", traj_prefix), function(path) {
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
extract_and_save("absgap_ckpt_",    TERM_LEVELS_ABSGAP, TERM_LABELS_ABSGAP, "absgap_coefs.csv")
extract_and_save("absgap_ln_ckpt_", TERM_LEVELS_ABSGAP, TERM_LABELS_ABSGAP, "absgap_ln_coefs.csv")

extract_traj_and_save("pref_traj_", TERM_LEVELS, TERM_LABELS, "pref_traj_coefs.csv")
extract_traj_and_save("sg_traj_",   TERM_LEVELS, TERM_LABELS, "sg_traj_coefs.csv")

save_gap_summary("sg_traj_", "sg_gap_summary.csv")

# ── Analysis 2: human pref ~ model preference (logistic trajectory) ───────────
TERM_LEVELS_HUM <- "preference"
TERM_LABELS_HUM <- "Model preference"
extract_traj_and_save("humfit_traj_", TERM_LEVELS_HUM, TERM_LABELS_HUM, "humfit_traj_coefs.csv")

# ── Analysis 2: model pref ~ individual constraints (trajectory) ──────────────
CONSTRAINT_COLS <- c("Form", "Percept", "Culture", "Power", "Intense",
                     "Icon", "Freq", "Len", "Lapse", "BStress")
TERM_LEVELS_CONSTR <- c(
  CONSTRAINT_COLS,
  "log_total_c", "freq_prob_c",
  paste0(CONSTRAINT_COLS, ":log_total_c"),
  "freq_prob_c:log_total_c"
)
TERM_LABELS_CONSTR <- c(
  CONSTRAINT_COLS,
  "Overall freq\n(ln total)", "RelFreq\n(P(alpha)\u22120.5)",
  paste0(CONSTRAINT_COLS, " \u00d7\nOverall freq"),
  "RelFreq \u00d7\nOverall freq"
)
extract_traj_and_save("constr_traj_", TERM_LEVELS_CONSTR, TERM_LABELS_CONSTR, "constr_traj_coefs.csv")

# ── Posterior predictive checks ───────────────────────────────────────────────
# Save tidy data frames (observed y + yrep draws) rather than pre-rendered
# figures, so the writeup can build ggplot2 plots with full aesthetic control.
PPC_DIR <- file.path(OUT_DIR, "ppc")
dir.create(PPC_DIR, showWarnings = FALSE, recursive = TRUE)

save_ppc_data <- function(prefix, fname, ndraws = 100, overwrite = FALSE) {
  out_path <- file.path(PPC_DIR, fname)
  if (!overwrite && file.exists(out_path)) {
    message(sprintf("  %s already exists; skipping (pass overwrite=TRUE to regenerate)", fname))
    return(invisible(NULL))
  }
  paths <- list_fits(prefix)
  if (length(paths) == 0) {
    message(sprintf("  No .rds files for '%s'; skipping PPC", prefix))
    return(invisible(NULL))
  }
  message(sprintf("Extracting PPC data for '%s' (%d models)...", prefix, length(paths)))

  rows <- lapply(paths, function(path) {
    slug   <- sub("\\.rds$", "", sub(paste0("^", prefix), "", basename(path)))
    corpus <- parse_corpus(slug)
    size   <- parse_size(slug)
    fit    <- readRDS(path)

    # Response variable name from the brms formula
    resp_name <- as.character(fit$formula$formula[[2]])
    y    <- fit$data[[resp_name]]
    yrep <- posterior_predict(fit, ndraws = ndraws)
    rm(fit); gc()

    n_obs    <- length(y)
    n_draws  <- nrow(yrep)

    # Observed
    obs_df <- data.frame(
      corpus     = corpus,
      model_size = size,
      source     = "y",
      draw       = NA_integer_,
      value      = y,
      stringsAsFactors = FALSE
    )

    # Replicated: yrep is ndraws × n_obs; unroll row-major
    rep_df <- data.frame(
      corpus     = corpus,
      model_size = size,
      source     = "yrep",
      draw       = rep(seq_len(n_draws), each = n_obs),
      value      = as.vector(t(yrep)),
      stringsAsFactors = FALSE
    )

    rbind(obs_df, rep_df)
  })

  out <- do.call(rbind, rows)
  out$model_size <- factor(out$model_size, levels = SIZE_LEVELS)
  out_path <- file.path(PPC_DIR, fname)
  saveRDS(out, out_path)
  message(sprintf("  Saved %s  (%s rows)", out_path, format(nrow(out), big.mark = ",")))
}

save_ppc_data("pref_final_",  "ppc_pref_final.rds")
save_ppc_data("sg_final_",    "ppc_sg_final.rds")
# absgap is not saved as RDS — too many checkpoint models; densities written directly

# ── PPC density CSVs (lightweight alternative to raw RDS files) ───────────────
# Pre-compute density curves (512 x/y points per draw) so writeup.qmd can
# plot PPCs with geom_line() on a tiny CSV rather than geom_density() on a
# multi-million-row RDS.
save_ppc_densities <- function(rds_fname, csv_fname, n_grid = 512, overwrite = FALSE) {
  rds_path <- file.path(PPC_DIR, rds_fname)
  csv_path <- file.path(PPC_DIR, csv_fname)
  if (!file.exists(rds_path)) {
    message(sprintf("  %s not found; skipping density CSV", rds_fname))
    return(invisible(NULL))
  }
  if (!overwrite && file.exists(csv_path)) {
    message(sprintf("  %s already exists; skipping (pass overwrite=TRUE to regenerate)", csv_fname))
    return(invisible(NULL))
  }
  message(sprintf("  Computing densities from %s ...", rds_fname))
  dat <- readRDS(rds_path)

  groups <- unique(dat[, c("corpus", "model_size")])

  rows <- lapply(seq_len(nrow(groups)), function(i) {
    corp <- groups$corpus[i]
    size <- groups$model_size[i]
    sub  <- dat[dat$corpus == corp & dat$model_size == size, ]

    # Observed density
    y_vals <- sub$value[sub$source == "y"]
    x_range <- range(c(y_vals, sub$value[sub$source == "yrep"]), na.rm = TRUE)
    dens_y <- density(y_vals, n = n_grid, from = x_range[1], to = x_range[2])
    obs_df <- data.frame(
      corpus = corp, model_size = size,
      source = "y", draw = NA_integer_,
      x = dens_y$x, dens = dens_y$y,
      stringsAsFactors = FALSE
    )

    # One density per replicated draw; reshape wide (n_obs × ndraws) then apply
    draws    <- sort(unique(sub$draw[sub$source == "yrep"]))
    yrep_mat <- matrix(sub$value[sub$source == "yrep"], ncol = length(draws))
    dens_rep <- apply(yrep_mat, 2, function(col)
      density(col, n = n_grid, from = x_range[1], to = x_range[2])$y)

    rep_df <- data.frame(
      corpus = corp, model_size = size,
      source = "yrep",
      draw   = rep(draws, each = n_grid),
      x      = rep(dens_y$x, times = length(draws)),
      dens   = as.vector(dens_rep),
      stringsAsFactors = FALSE
    )
    rbind(obs_df, rep_df)
  })

  out <- do.call(rbind, rows)
  out$model_size <- factor(out$model_size, levels = SIZE_LEVELS)
  write_csv(out |> mutate(across(where(is.factor), as.character)), csv_path)
  message(sprintf("  Saved %-35s (%s rows)", csv_fname, format(nrow(out), big.mark = ",")))
}

# ── Direct density CSV for absgap (no intermediate RDS) ──────────────────────
# Loads each fitted model one at a time across 4 workers, draws
# posterior_predict, computes densities immediately, then discards raw draws.
save_ppc_densities_direct <- function(prefix, csv_fname,
                                      ndraws = 100, n_grid = 512,
                                      n_workers = 4, overwrite = FALSE,
                                      log_transform = FALSE) {
  csv_path <- file.path(PPC_DIR, csv_fname)

  # Load existing data and determine which models are already done
  existing <- NULL
  done_keys <- character(0)
  if (!overwrite && file.exists(csv_path)) {
    existing   <- read_csv(csv_path, show_col_types = FALSE)
    done_keys  <- unique(paste(existing$corpus, existing$model_size, sep = "|"))
    message(sprintf("  %s exists with %d model(s) already computed; checking for new ones...",
                    csv_fname, length(done_keys)))
  }

  paths <- list_fits(prefix)
  if (length(paths) == 0) {
    message(sprintf("  No .rds files for '%s'; skipping", prefix))
    return(invisible(NULL))
  }

  # Filter to paths not yet in the CSV
  paths_todo <- Filter(function(path) {
    slug   <- sub("\\.rds$", "", sub(paste0("^", prefix), "", basename(path)))
    key    <- paste(parse_corpus(slug), parse_size(slug), sep = "|")
    !key %in% done_keys
  }, paths)

  if (length(paths_todo) == 0) {
    message(sprintf("  All models already in %s; nothing to do.", csv_fname))
    return(invisible(NULL))
  }

  message(sprintf("  Computing PPC densities for '%s' (%d new model(s), %d workers)...",
                  prefix, length(paths_todo), n_workers))

  plan(multisession, workers = n_workers)
  on.exit(plan(sequential))

  new_rows <- future_map(
    paths_todo,
    .progress = TRUE,
    .options  = furrr_options(packages = c("brms"), seed = TRUE),
    function(path) {
      slug   <- sub("\\.rds$", "", sub(paste0("^", prefix), "", basename(path)))
      corpus <- parse_corpus(slug)
      size   <- parse_size(slug)
      fit    <- readRDS(path)

      resp_name <- as.character(fit$formula$formula[[2]])
      y    <- fit$data[[resp_name]]
      yrep <- posterior_predict(fit, ndraws = ndraws)
      rm(fit); gc()

      if (log_transform) {
        y    <- log(pmax(y,    1e-10))
        yrep <- log(pmax(yrep, 1e-10))
      }

      x_range  <- range(c(y, yrep), na.rm = TRUE)
      dens_y   <- density(y, n = n_grid, from = x_range[1], to = x_range[2])
      dens_rep <- apply(yrep, 1, function(row)
        density(row, n = n_grid, from = x_range[1], to = x_range[2])$y)

      obs_df <- data.frame(
        corpus = corpus, model_size = size,
        source = "y", draw = NA_integer_,
        x = dens_y$x, dens = dens_y$y,
        stringsAsFactors = FALSE
      )
      rep_df <- data.frame(
        corpus = corpus, model_size = size,
        source = "yrep",
        draw   = rep(seq_len(ncol(dens_rep)), each = n_grid),
        x      = rep(dens_y$x, times = ncol(dens_rep)),
        dens   = as.vector(dens_rep),
        stringsAsFactors = FALSE
      )
      rbind(obs_df, rep_df)
    }
  )

  new_data <- do.call(rbind, new_rows)
  out <- if (!is.null(existing)) rbind(existing, new_data) else new_data
  out$model_size <- factor(out$model_size, levels = SIZE_LEVELS)
  write_csv(out |> mutate(across(where(is.factor), as.character)), csv_path)
  message(sprintf("  Saved %-35s (%s rows)", csv_fname, format(nrow(out), big.mark = ",")))
}

save_ppc_densities("ppc_pref_final.rds", "ppc_pref_final_dens.csv")
save_ppc_densities("ppc_sg_final.rds",   "ppc_sg_final_dens.csv")
save_ppc_densities_direct("absgap_ckpt_",    "ppc_absgap_dens.csv")
save_ppc_densities_direct("absgap_ln_ckpt_", "ppc_absgap_ln_dens.csv")
save_ppc_densities_direct("absgap_ln_ckpt_", "ppc_absgap_ln_log_dens.csv", log_transform = TRUE)

message("\nDone. All results saved to: ", normalizePath(OUT_DIR))
