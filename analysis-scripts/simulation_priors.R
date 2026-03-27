## simulation_priors.R
##
## Simulation study: do RE / FE priors distort fixed-effect recovery?
##
## Two DGPs (both have random intercepts by group):
##   signal:  y = b1*x1 + b2*x2 + b12*x1:x2 + u_binom + epsilon   (all non-zero)
##   null:    y = 0  + 0*x1 + 0*x2 + 0*x1:x2 + u_binom + epsilon  (all zero)
##
## Manipulations (fully crossed):
##   data amount  — n_binoms × n_obs
##   FE prior SD  — student_t(3, 0, fixef_sd)
##   RE prior SD  — student_t(3, 0, ranef_sd)
##
## 100 replicate datasets per (DGP × data level) combination.
## Same datasets reused across all prior combinations within that (DGP × data level).
##
## Outputs:
##   simulation_results.csv        — full posterior summaries
##   simulation_plot_recovery.png  — coefficient recovery (signal DGP)
##   simulation_plot_errors.png    — Type 1 / Type 2 error rates

suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(furrr)
  library(scales)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR <- {
  knitr_input <- tryCatch(knitr::current_input(), error = function(e) NULL)
  if (!is.null(knitr_input) && nchar(knitr_input) > 0) {
    dirname(normalizePath(knitr_input, mustWork = FALSE))
  } else if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (nchar(p) > 0) dirname(normalizePath(p, mustWork = FALSE))
    else normalizePath(".", mustWork = FALSE)
  } else {
    normalizePath(".", mustWork = FALSE)
  }
}
OUT_DIR <- SCRIPT_DIR

# ── True parameters ────────────────────────────────────────────────────────────
SIGMA_RE  <- 0.5   # true random-intercept SD (same in both DGPs)
SIGMA_EPS <- 0.5   # true residual SD

SIGNAL_BETAS <- c(b0 = 0.0, b1 = 0.4, b2 = 0.6, b12 = 0.3)
NULL_BETAS   <- c(b0 = 0.0, b1 = 0.0, b2 = 0.0, b12 = 0.0)

# Map brms term names → true values for each DGP
TRUE_VALS <- list(
  signal = c(Intercept = 0.0, x1 = 0.4, x2 = 0.6, `x1:x2` = 0.3),
  null   = c(Intercept = 0.0, x1 = 0.0, x2 = 0.0, `x1:x2` = 0.0)
)

# Which terms to evaluate for Type 1 / Type 2 (not intercept)
EVAL_TERMS <- c("x1", "x2", "x1:x2")

# ── Simulation grid ────────────────────────────────────────────────────────────
DATA_LEVELS <- list(
  small  = list(n_binoms = 40,  n_obs = 3),    # 120 obs
  medium = list(n_binoms = 150, n_obs = 5),    # 750 obs
  large  = list(n_binoms = 400, n_obs = 10)    # 4000 obs
)

FIXEF_SD_LEVELS <- c(0.5, 2.5, 10)
RANEF_SD_LEVELS <- c(0.5, 2.5, 10)

N_REPS <- 100   # independent datasets per (DGP × data level)

# brms settings
CHAINS <- 1
WARMUP <- 1000
ITER   <- 2000   # total iterations (1000 post-warmup)

# ── Data generation ────────────────────────────────────────────────────────────
simulate_data <- function(n_binoms, n_obs, seed, betas) {
  set.seed(seed)
  binom_ids <- paste0("g", seq_len(n_binoms))
  re        <- rnorm(n_binoms, 0, SIGMA_RE)
  names(re) <- binom_ids
  expand.grid(binom = binom_ids, obs = seq_len(n_obs), stringsAsFactors = FALSE) |>
    mutate(
      x1 = runif(n(), -1, 1),
      x2 = runif(n(), -1, 1),
      y  = betas["b0"] +
           betas["b1"]  * x1 +
           betas["b2"]  * x2 +
           betas["b12"] * x1 * x2 +
           re[binom] +
           rnorm(n(), 0, SIGMA_EPS)
    ) |>
    select(binom, x1, x2, y)
}

# ── Build datasets ─────────────────────────────────────────────────────────────
# Seeds depend only on (dgp, data_level, rep) — same dataset reused across priors.
cat("Simulating datasets...\n")
datasets <- crossing(
  dgp        = c("signal", "null"),
  data_level = names(DATA_LEVELS),
  rep        = seq_len(N_REPS)
) |>
  mutate(
    n_binoms   = map_int(data_level, ~ DATA_LEVELS[[.x]]$n_binoms),
    n_obs      = map_int(data_level, ~ DATA_LEVELS[[.x]]$n_obs),
    n_total    = n_binoms * n_obs,
    # unique seed per (dgp, data_level, rep)
    seed       = (as.integer(factor(dgp)) * 10000 +
                  as.integer(factor(data_level, levels = names(DATA_LEVELS))) * 100 +
                  rep),
    betas      = map(dgp, ~ if (.x == "signal") SIGNAL_BETAS else NULL_BETAS),
    dat        = pmap(list(n_binoms, n_obs, seed, betas), simulate_data)
  )

# ── Full grid ─────────────────────────────────────────────────────────────────
prior_combos <- crossing(
  fixef_sd = FIXEF_SD_LEVELS,
  ranef_sd = RANEF_SD_LEVELS
) |>
  mutate(combo_id = row_number())

grid <- crossing(
  datasets |> select(dgp, data_level, rep, n_binoms, n_obs, n_total, dat),
  fixef_sd = FIXEF_SD_LEVELS,
  ranef_sd = RANEF_SD_LEVELS
) |>
  left_join(prior_combos, by = c("fixef_sd", "ranef_sd")) |>
  mutate(row_id = row_number())

cat(sprintf("Total fits: %d  (%d signal + %d null)\n",
            nrow(grid), sum(grid$dgp == "signal"), sum(grid$dgp == "null")))

# ── Pre-compile one template per prior combination ────────────────────────────
# brms bakes prior values into the Stan code, so each unique (fixef_sd, ranef_sd)
# pair requires its own compiled binary. We compile 9 templates on a tiny dummy
# dataset, then reuse them via update(..., recompile = FALSE) for all actual fits.
cat(sprintf("Pre-compiling %d template models...\n", nrow(prior_combos)))
dummy_dat <- simulate_data(10, 3, seed = 0, betas = SIGNAL_BETAS)

template_fits <- map(seq_len(nrow(prior_combos)), function(i) {
  cat(sprintf("  Template %d/%d  (fixef_sd=%s, ranef_sd=%s)\n",
              i, nrow(prior_combos),
              prior_combos$fixef_sd[i], prior_combos$ranef_sd[i]))
  brm(
    y ~ x1 * x2 + (1 | binom),
    data    = dummy_dat,
    family  = gaussian(),
    prior   = c(
      set_prior(sprintf("student_t(3, 0, %s)", prior_combos$fixef_sd[i]), class = "b"),
      set_prior(sprintf("student_t(3, 0, %s)", prior_combos$ranef_sd[i]), class = "sd")
    ),
    chains  = 1,
    iter    = 200,
    warmup  = 100,
    refresh = 0,
    silent  = 2
  )
})
cat("  Templates ready.\n")

# ── Run in parallel ────────────────────────────────────────────────────────────
n_workers <- 6
cat(sprintf("Running %d fits with %d parallel workers...\n", nrow(grid), n_workers))
plan(multisession, workers = n_workers)

sim_results <- future_map(
  seq_len(nrow(grid)),
  function(i) {
    row      <- grid[i, ]
    template <- template_fits[[row$combo_id]]
    
    message(sprintf("[%d/%d] dgp=%s data=%s rep=%d fixef_sd=%s ranef_sd=%s",
                    i, nrow(grid),
                    row$dgp, row$data_level, row$rep,
                    row$fixef_sd, row$ranef_sd))
    cat(sprintf("[%d/%d] dgp=%s data=%s rep=%d fixef_sd=%s ranef_sd=%s\n",
                i, nrow(grid),
                row$dgp, row$data_level, row$rep,
                row$fixef_sd, row$ranef_sd),
        file = "sim_progress.txt", append = TRUE)
    
    fit <- tryCatch(
      update(template,
             newdata   = row$dat[[1]],
             recompile = FALSE,
             chains    = CHAINS,
             iter      = ITER,
             warmup    = WARMUP,
             refresh   = 0,
             silent    = 2),
      error = function(e) {
        message(sprintf("Row %d failed: %s", i, conditionMessage(e)))
        NULL
      }
    )
    if (is.null(fit)) return(NULL)
    fe <- fixef(fit)
    bind_cols(
      row |> select(dgp, data_level, rep, n_binoms, n_obs, n_total, fixef_sd, ranef_sd),
      tibble(
        term     = rownames(fe),
        estimate = fe[, "Estimate"],
        lo       = fe[, "Q2.5"],
        hi       = fe[, "Q97.5"]
      )
    )
  },
  .options = furrr_options(seed = TRUE)
)

plan(sequential)

results <- bind_rows(sim_results) |>
  mutate(
    true_val = map2_dbl(dgp, term, ~ TRUE_VALS[[.x]][.y]),
    covered  = lo <= true_val & true_val <= hi,
    # CI excludes zero: used for error-rate decisions
    sig      = lo > 0 | hi < 0
  )

write.csv(results, file.path(OUT_DIR, "simulation_results.csv"), row.names = FALSE)
cat(sprintf("Saved simulation_results.csv (%d rows)\n", nrow(results)))

# ── Helper labels ─────────────────────────────────────────────────────────────
add_labels <- function(df) {
  df |>
    mutate(
      fixef_label = factor(
        sprintf("FE prior SD = %s", fixef_sd),
        levels = sprintf("FE prior SD = %s", sort(unique(fixef_sd)))
      ),
      ranef_label = factor(
        sprintf("RE prior SD = %s", ranef_sd),
        levels = sprintf("RE prior SD = %s", sort(unique(ranef_sd)))
      )
    )
}

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Coefficient recovery (signal DGP, all terms)
# ═══════════════════════════════════════════════════════════════════════════════
cat("Plotting coefficient recovery...\n")

recovery <- results |>
  filter(dgp == "signal") |>
  add_labels() |>
  group_by(term, data_level, n_total, fixef_sd, ranef_sd,
           fixef_label, ranef_label) |>
  summarise(
    mean_est  = mean(estimate),
    mean_lo   = mean(lo),
    mean_hi   = mean(hi),
    true_val  = first(true_val),
    coverage  = mean(covered),
    .groups   = "drop"
  ) |>
  filter(term %in% EVAL_TERMS) |>
  mutate(term = factor(term, levels = EVAL_TERMS))

p_recovery <- ggplot(recovery,
                     aes(x = factor(n_total), colour = term, group = term)) +
  geom_hline(aes(yintercept = true_val, colour = term),
             linetype = "dashed", linewidth = 0.6, alpha = 0.5) +
  geom_linerange(aes(ymin = mean_lo, ymax = mean_hi),
                 position = position_dodge(width = 0.5),
                 linewidth = 0.8, alpha = 0.7) +
  geom_point(aes(y = mean_est),
             position = position_dodge(width = 0.5),
             size = 2.5) +
  scale_x_discrete("Total observations") +
  scale_y_continuous("Posterior mean \u00b1 avg 95% CI") +
  scale_colour_brewer("Term", palette = "Dark2") +
  facet_grid(ranef_label ~ fixef_label) +
  labs(
    title    = "Fixed-effect recovery (signal DGP)",
    subtitle = sprintf(
      "Averaged over %d replicates | \u03c3_RE = %.1f, \u03c3_\u03b5 = %.1f | %d chain, %d warmup + %d samples",
      N_REPS, SIGMA_RE, SIGMA_EPS, CHAINS, WARMUP, ITER - WARMUP
    ),
    caption  = "Dashed lines = true parameter values"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position  = "right",
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = 13),
    strip.text       = element_text(face = "bold", size = 11)
  )

ggsave(file.path(OUT_DIR, "simulation_plot_recovery.png"),
       p_recovery, width = 11, height = 8, dpi = 150)
cat("  Saved simulation_plot_recovery.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Type 1 and Type 2 error rates
#
#   Type 1 (from null DGP):    rate at which CI excludes 0 when true effect = 0
#   Type 2 (from signal DGP):  rate at which CI includes 0 when true effect ≠ 0
# ═══════════════════════════════════════════════════════════════════════════════
cat("Plotting error rates...\n")

error_rates <- bind_rows(
  # Type 1: null DGP, CI excludes 0 = false positive
  results |>
    filter(dgp == "null", term %in% EVAL_TERMS) |>
    group_by(term, data_level, n_total, fixef_sd, ranef_sd) |>
    summarise(rate = mean(sig), .groups = "drop") |>
    mutate(error_type = "Type 1 (false positive)"),

  # Type 2: signal DGP, CI includes 0 = missed effect
  results |>
    filter(dgp == "signal", term %in% EVAL_TERMS) |>
    group_by(term, data_level, n_total, fixef_sd, ranef_sd) |>
    summarise(rate = mean(!sig), .groups = "drop") |>
    mutate(error_type = "Type 2 (false negative)")
) |>
  mutate(term = factor(term, levels = EVAL_TERMS)) |>
  add_labels()

p_errors <- ggplot(error_rates,
                   aes(x = factor(n_total), y = rate,
                       colour = term, linetype = error_type,
                       group = interaction(term, error_type))) +
  geom_hline(yintercept = 0.05, colour = "grey40",
             linetype = "dotted", linewidth = 0.6) +
  geom_line(linewidth = 0.8, alpha = 0.9) +
  geom_point(size = 2.5) +
  scale_x_discrete("Total observations") +
  scale_y_continuous("Error rate", labels = percent_format(accuracy = 1),
                     limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
  scale_colour_brewer("Term", palette = "Dark2") +
  scale_linetype_manual("Error type",
                        values = c("Type 1 (false positive)" = "solid",
                                   "Type 2 (false negative)" = "dashed")) +
  facet_grid(ranef_label ~ fixef_label) +
  labs(
    title    = "Type 1 and Type 2 error rates by prior width and data amount",
    subtitle = sprintf(
      "Decision rule: 95%% CI excludes 0 | %d replicates per condition | dotted line = \u03b1 = .05",
      N_REPS
    ),
    caption  = paste0(
      "Type 1: null DGP (true \u03b2 = 0), CI excludes 0\n",
      "Type 2: signal DGP (true \u03b2 \u2260 0), CI includes 0"
    )
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position  = "right",
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = 13),
    strip.text       = element_text(face = "bold", size = 11)
  )

ggsave(file.path(OUT_DIR, "simulation_plot_errors.png"),
       p_errors, width = 12, height = 9, dpi = 150)
cat("  Saved simulation_plot_errors.png\n")

print(p_recovery)
print(p_errors)

# ── Summary table ─────────────────────────────────────────────────────────────
cat("\nError rate summary:\n")
error_rates |>
  select(error_type, term, data_level, fixef_sd, ranef_sd, rate) |>
  arrange(error_type, term, data_level, fixef_sd, ranef_sd) |>
  print(n = Inf)
