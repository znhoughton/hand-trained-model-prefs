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
OUT_PDF    <- file.path(SCRIPT_DIR, "delta_ll_plot.pdf")
OUT_PNG    <- file.path(SCRIPT_DIR, "delta_ll_plot.png")

# ── Load data ─────────────────────────────────────────────────────────────────
message("Loading data ...")

# Individual human trial data
human_trials <- read_csv(HUMAN_CSV, show_col_types = FALSE) |>
  mutate(
    binom      = Alpha,
    resp_alpha = as.integer(resp == "alpha")
  ) |>
  select(binom, resp_alpha, RelFreq)

# Model preferences (already averaged over prompts)
model_prefs <- read_csv(PREFS_CSV, show_col_types = FALSE)

# Validation perplexities
ppl <- read_csv(PPL_CSV, show_col_types = FALSE)

message(sprintf(
  "  %d human trials, %d unique binomials, %d models",
  nrow(human_trials),
  n_distinct(human_trials$binom),
  n_distinct(model_prefs$model)
))

# ── Join trials with model preferences ───────────────────────────────────────
dat <- human_trials |>
  inner_join(model_prefs, by = "binom")   # RelFreq from human data (static corpus freq)

# ── ΔLL per model ─────────────────────────────────────────────────────────────
message("Fitting logistic regressions ...")
delta_ll <- dat |>
  group_by(model, model_params, model_label) |>
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

# ── Join perplexity ───────────────────────────────────────────────────────────
results <- delta_ll |>
  left_join(ppl |> select(model, perplexity), by = "model")

message("\nResults:")
print(results |> select(model_label, model_params, perplexity, delta_ll, n_items))

# ── Plot aesthetics ───────────────────────────────────────────────────────────
# Order models by parameter count for colour scale
param_order  <- c("117M", "345M", "762M", "1542M")
model_colours <- c(
  "117M"  = "#2166ac",
  "345M"  = "#74add1",
  "762M"  = "#f4a582",
  "1542M" = "#b2182b"
)
model_shapes <- c(
  "117M"  = 16L,
  "345M"  = 17L,
  "762M"  = 15L,
  "1542M" = 18L
)

results <- results |>
  mutate(model_params = factor(model_params, levels = param_order))

# ── Significance test (slope of log2 perplexity → ΔLL) ───────────────────────
# With only 4 points, treat this as descriptive; report r and p from cor.test.
ct <- cor.test(log2(results$perplexity), results$delta_ll)
slope_lbl <- sprintf(
  "r = %.2f, t(%d) = %.2f, p %s %.3f",
  ct$estimate, ct$parameter, ct$statistic,
  ifelse(ct$p.value < .001, "<", "="),
  max(ct$p.value, 0.001)
)
message("\nCorrelation (log2 perplexity × ΔLL): ", slope_lbl)

# ── Plot ──────────────────────────────────────────────────────────────────────
p <- ggplot(results,
            aes(x = perplexity, y = delta_ll,
                colour = model_params, shape = model_params)) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              linetype = "dashed", linewidth = 0.9,
              colour = "grey50", aes(group = 1)) +  # single line across all models
  geom_point(size = 4) +
  geom_text(aes(label = model_label),
            hjust = -0.12, vjust = 0.4,
            size = 3.5, fontface = "bold",
            show.legend = FALSE) +
  annotate("text", x = Inf, y = -Inf,
           label = slope_lbl,
           hjust = 1.05, vjust = -0.5,
           size = 3.5, fontface = "italic", colour = "grey30") +
  scale_x_continuous(
    "Validation perplexity on WikiText-2 (lower \u2192 better)",
    trans  = "log2",
    labels = scales::trans_format("log2", scales::math_format(2^.x))
  ) +
  scale_y_continuous("\u0394LL") +
  scale_colour_manual(values = model_colours, name = "Parameters") +
  scale_shape_manual(values  = model_shapes,  name = "Parameters") +
  guides(colour = guide_legend(override.aes = list(size = 3)),
         shape  = guide_legend(override.aes = list(size = 3))) +
  labs(
    title    = "\u0394LL vs. validation perplexity — GPT-2 family (Oh & Schuler 2023 models)",
    subtitle = paste0(
      "\u0394LL = logLik(resp_alpha \u223c RelFreq + model pref) \u2212 logLik(resp_alpha \u223c RelFreq). ",
      "One point per model. Dashed = OLS fit."
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
