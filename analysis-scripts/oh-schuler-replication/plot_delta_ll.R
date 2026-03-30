# plot_delta_ll.R
#
# Oh & Schuler (2023) replication with binomial ordering preferences.
#
# Uses GPT-2 family model preferences from get_model_prefs.py on our human
# experiment binomials, then replicates the ΔLL × Perplexity plot.
#
# Method (identical to abspref_casestudy.Rmd):
#   outcome  = resp_alpha (binary: 1 = human chose α ordering)
#   baseline = glm(resp_alpha ~ RelFreq,              family = binomial)
#   full     = glm(resp_alpha ~ RelFreq + preference, family = binomial)
#   ΔLL      = logLik(full) − logLik(baseline)
#
# Perplexity = validation perplexity on WikiText-2 test set (from get_model_prefs.py)
#
# Output: delta_ll_plot.pdf / .png

library(tidyverse)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR <- normalizePath(dirname(sys.frame(1)$ofile), mustWork = FALSE)
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, "..", ".."), mustWork = FALSE)
DATA_DIR   <- file.path(BASE_DIR, "Data")

PREFS_CSV  <- file.path(SCRIPT_DIR, "oh_schuler_prefs.csv")
PPL_CSV    <- file.path(SCRIPT_DIR, "oh_schuler_perplexity.csv")
HUMAN_CSV  <- file.path(DATA_DIR, "all_human_data.csv")
BINOMS_CSV <- file.path(DATA_DIR, "nonce_and_attested_binoms.csv")
OUT_PDF    <- file.path(SCRIPT_DIR, "delta_ll_plot.pdf")
OUT_PNG    <- file.path(SCRIPT_DIR, "delta_ll_plot.png")

# ── Load data ─────────────────────────────────────────────────────────────────
message("Loading data ...")

LOG_PATH <- file.path(SCRIPT_DIR, "plot_delta_ll.log")
log_line <- function(...) {
  msg <- paste0(...)
  message(msg)
  cat(msg, "\n", file = LOG_PATH, append = TRUE)
}
cat(sprintf("\n%s\nRun: %s\n", strrep("=", 60), format(Sys.time())),
    file = LOG_PATH, append = TRUE)

# Attested binomials only
all_binoms_df   <- read_csv(BINOMS_CSV, show_col_types = FALSE)
attested_binoms <- all_binoms_df |> filter(Attested == 1) |> pull(Alpha)
nonce_binoms    <- all_binoms_df |> filter(Attested == 0) |> pull(Alpha)

human_trials_all <- read_csv(HUMAN_CSV, show_col_types = FALSE) |>
  mutate(binom = Alpha, resp_alpha = as.integer(resp == "alpha")) |>
  select(binom, resp_alpha, RelFreq)

human_trials <- human_trials_all |> filter(binom %in% attested_binoms)

model_prefs_all <- read_csv(PREFS_CSV, show_col_types = FALSE)
model_prefs     <- model_prefs_all |> filter(binom %in% attested_binoms)

ppl <- read_csv(PPL_CSV, show_col_types = FALSE)

# ── Log drop counts ───────────────────────────────────────────────────────────
n_human_before  <- n_distinct(human_trials_all$binom)
n_human_after   <- n_distinct(human_trials$binom)
n_human_dropped <- n_human_before - n_human_after

n_prefs_before  <- n_distinct(model_prefs_all$binom)
n_prefs_after   <- n_distinct(model_prefs$binom)
n_prefs_dropped <- n_prefs_before - n_prefs_after

dropped_human <- setdiff(unique(human_trials_all$binom), attested_binoms)
dropped_prefs <- setdiff(unique(model_prefs_all$binom),  attested_binoms)

log_line(sprintf(
  "Human trials:  %d → %d unique binomials retained  (%d dropped as non-attested)",
  n_human_before, n_human_after, n_human_dropped
))
if (length(dropped_human) > 0) {
  log_line("  Dropped from human_trials: ", paste(dropped_human, collapse = ", "))
}

log_line(sprintf(
  "Model prefs:   %d → %d unique binomials retained  (%d dropped as non-attested)",
  n_prefs_before, n_prefs_after, n_prefs_dropped
))
if (length(dropped_prefs) > 0) {
  log_line("  Dropped from model_prefs: ", paste(dropped_prefs, collapse = ", "))
}

log_line(sprintf(
  "Final: %d human trials, %d unique attested binomials, %d models",
  nrow(human_trials), n_distinct(human_trials$binom), n_distinct(model_prefs$model)
))
message(sprintf("  (full log → %s)", LOG_PATH))

# ── Join & compute ΔLL per model ─────────────────────────────────────────────
dat <- human_trials |>
  inner_join(model_prefs, by = "binom")

message("Fitting logistic regressions ...")
delta_ll <- dat |>
  group_by(model, model_family, model_params, model_label) |>
  group_modify(function(d, k) {
    baseline_mod <- glm(resp_alpha ~ RelFreq,              family = binomial, data = d)
    full_mod     <- glm(resp_alpha ~ RelFreq + preference, family = binomial, data = d)
    tibble(
      delta_ll = as.numeric(logLik(full_mod)) - as.numeric(logLik(baseline_mod)),
      n_trials = nrow(d),
      n_items  = n_distinct(d$binom)
    )
  }) |>
  ungroup()

results <- delta_ll |>
  left_join(ppl |> select(model, perplexity), by = "model")

message("\nResults:")
print(results |> select(model_family, model_label, model_params, perplexity, delta_ll),
      n = Inf)

# ── Plot aesthetics ───────────────────────────────────────────────────────────
# Numeric param count for ordering within each family
results <- results |>
  mutate(
    params_M = as.numeric(sub("M$", "", model_params)),
    model_family = factor(model_family, levels = c("GPT-2", "GPT-Neo", "OPT"))
  ) |>
  arrange(model_family, params_M)

family_colours <- c("GPT-2" = "#2166ac", "GPT-Neo" = "#4dac26", "OPT" = "#b2182b")

# ── Significance test per family ──────────────────────────────────────────────
slope_tests <- results |>
  group_by(model_family) |>
  group_modify(function(d, k) {
    if (nrow(d) < 3) return(tibble(r = NA_real_, p = NA_real_))
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

# ── Plot ──────────────────────────────────────────────────────────────────────
p <- ggplot(results,
            aes(x = perplexity, y = delta_ll,
                colour = model_family, shape = model_family)) +
  facet_wrap(~ model_family, scales = "free_x") +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              linetype = "dashed", linewidth = 0.9) +
  geom_point(aes(size = params_M), alpha = 0.9) +
  geom_text(aes(label = model_label),
            hjust = -0.1, vjust = 0.4,
            size = 3, fontface = "bold", show.legend = FALSE) +
  geom_text(data = slope_tests, aes(label = lbl),
            x = -Inf, y = Inf, hjust = -0.05, vjust = 1.4,
            size = 3.2, fontface = "italic", colour = "grey30",
            inherit.aes = FALSE) +
  scale_x_continuous(
    "Validation perplexity on WikiText-2 (lower \u2192 better)",
    trans  = "log2",
    labels = scales::trans_format("log2", scales::math_format(2^.x))
  ) +
  scale_y_continuous("\u0394LL") +
  scale_colour_manual(values = family_colours, name = "Model family") +
  scale_shape_manual(values = c("GPT-2" = 16L, "GPT-Neo" = 17L, "OPT" = 15L),
                     name = "Model family") +
  scale_size_continuous(name = "Parameters (M)", range = c(2, 6)) +
  guides(colour = guide_legend(override.aes = list(size = 3)),
         shape  = guide_legend(override.aes = list(size = 3))) +
  labs(
    title    = "\u0394LL vs. validation perplexity — Oh & Schuler (2023) models",
    subtitle = paste0(
      "\u0394LL = logLik(resp_alpha \u223c RelFreq + model pref) \u2212 logLik(resp_alpha \u223c RelFreq). ",
      "One point per model. Dashed = OLS fit per family."
    )
  ) +
  theme_classic(base_size = 13) +
  theme(
    strip.background = element_blank(),
    strip.text       = element_text(face = "bold", size = 13),
    panel.grid.major = element_line(colour = "grey92", linewidth = 0.4),
    axis.line        = element_line(colour = "grey40"),
    axis.ticks       = element_line(colour = "grey40"),
    legend.position  = "right"
  )

print(p)

ggsave(OUT_PDF, p, width = 8, height = 5)
ggsave(OUT_PNG, p, width = 8, height = 5, dpi = 150)
message(sprintf("\nSaved:\n  %s\n  %s", OUT_PDF, OUT_PNG))
