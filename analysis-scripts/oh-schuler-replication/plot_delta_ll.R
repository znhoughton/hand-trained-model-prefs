# plot_delta_ll.R
#
# Oh & Schuler (2023) replication with binomial ordering preferences.
#
# Left panel  — ΔLL × Perplexity:
#   outcome  = resp_alpha (binary: 1 = human chose α ordering)
#   baseline = glm(resp_alpha ~ RelFreq,              family = binomial)
#   full     = glm(resp_alpha ~ RelFreq + preference, family = binomial)
#   ΔLL      = logLik(full) − logLik(baseline)
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

PREFS_CSV          <- file.path(SCRIPT_DIR, "oh_schuler_prefs.csv")
PREFS_BY_PROMPT_CSV <- file.path(SCRIPT_DIR, "oh_schuler_prefs_by_prompt.csv")
PPL_CSV       <- file.path(SCRIPT_DIR, "oh_schuler_perplexity.csv")
HUMAN_CSV     <- file.path(DATA_DIR, "all_human_data.csv")
BINOMS_CSV    <- file.path(DATA_DIR, "nonce_and_attested_binoms.csv")
PILE_FREQ_CSV <- file.path(SCRIPT_DIR, "pile_corpus_freq.csv")
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
model_prefs     <- model_prefs_all |> filter(binom %in% attested_binoms)

# Per-prompt preferences for mixed model (if available)
has_by_prompt <- file.exists(PREFS_BY_PROMPT_CSV)
if (has_by_prompt) {
  model_prefs_by_prompt <- read_csv(PREFS_BY_PROMPT_CSV, show_col_types = FALSE) |>
    filter(binom %in% attested_binoms)
} else {
  message("⚠  Per-prompt preferences not found — right panel will use averaged data with lm().")
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
    "⚠  Pile corpus freq not found (%s) — falling back to Google Books RelFreq.\n",
    "   Run compute_pile_freq.py to generate it.", PILE_FREQ_CSV
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
PILE_FREQ_FAMILIES <- c("OPT", "OLMo")

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
    if (any(is.na(d$freq_use))) {
      d <- filter(d, !is.na(freq_use))
    }
    if (nrow(d) < 5) return(tibble(delta_ll = NA_real_, n_trials = nrow(d), n_items = NA_integer_))
    baseline_mod <- glm(resp_alpha ~ freq_use,              family = binomial, data = d)
    full_mod     <- glm(resp_alpha ~ freq_use + preference, family = binomial, data = d)
    tibble(
      delta_ll = as.numeric(logLik(full_mod)) - as.numeric(logLik(baseline_mod)),
      n_trials = nrow(d),
      n_items  = n_distinct(d$binom)
    )
  }) |>
  ungroup()

# ── Fit item-level mixed model for AbsPref coefficient ────────────────────────
# Uses per-prompt data (one row per binom×prompt) with random intercepts for
# binom and prompt. Falls back to lm() on averaged data if by-prompt CSV absent.
message("Fitting AbsPref mixed models ...")

# Source data for the right-panel regression
rp_source <- if (has_by_prompt) model_prefs_by_prompt else model_prefs

abspref_coef <- rp_source |>
  group_by(model, model_family, model_params, model_label) |>
  group_modify(function(d, k) {
    d <- d |> left_join(binom_items, by = "binom")

    # Choose RelFreq and OverallFreq based on model family & availability
    if (use_pile_freq && k$model_family %in% PILE_FREQ_FAMILIES &&
        "RelFreq_pile" %in% names(d) && "OverallFreq_pile" %in% names(d)) {
      d$rf  <- d$RelFreq_pile
      d$ovf <- d$OverallFreq_pile
    } else {
      d$rf  <- d$RelFreq_google
      d$ovf <- d$OverallFreq_google
    }

    d <- d |> filter(!is.na(AbsPref), !is.na(rf), !is.na(ovf), !is.na(preference))
    if (nrow(d) < 5) return(tibble(estimate = NA_real_, se = NA_real_, t = NA_real_, p = NA_real_, n = nrow(d)))

    # Center all predictors
    d <- d |> mutate(
      AbsPref_c = as.numeric(scale(AbsPref, scale = FALSE)),
      rf_c      = as.numeric(scale(rf,      scale = FALSE)),
      ovf_c     = as.numeric(scale(ovf,     scale = FALSE))
    )

    if (has_by_prompt && "prompt" %in% names(d)) {
      fit <- tryCatch(
        lmer(preference ~ AbsPref_c + rf_c + ovf_c +
               AbsPref_c:ovf_c + rf_c:ovf_c +
               (1 | binom) + (1 | prompt),
             data = d, REML = TRUE),
        error = function(e) NULL
      )
      if (is.null(fit)) {
        # Fall back to lm if lmer fails (e.g., singular fit)
        fit <- lm(preference ~ AbsPref_c + rf_c + ovf_c +
                    AbsPref_c:ovf_c + rf_c:ovf_c, data = d)
        cf  <- summary(fit)$coefficients
      } else {
        cf <- summary(fit)$coefficients
      }
    } else {
      fit <- lm(preference ~ AbsPref_c + rf_c + ovf_c +
                  AbsPref_c:ovf_c + rf_c:ovf_c, data = d)
      cf  <- summary(fit)$coefficients
    }

    if (!"AbsPref_c" %in% rownames(cf)) return(tibble(estimate = NA_real_, se = NA_real_, t = NA_real_, p = NA_real_, n = nrow(d)))
    tibble(
      estimate = cf["AbsPref_c", "Estimate"],
      se       = cf["AbsPref_c", "Std. Error"],
      t        = cf["AbsPref_c", "t value"],
      p        = if ("Pr(>|t|)" %in% colnames(cf)) cf["AbsPref_c", "Pr(>|t|)"] else NA_real_,
      n        = nrow(d)
    )
  }) |>
  ungroup()

# ── Combine and join perplexity ───────────────────────────────────────────────
results <- delta_ll |>
  left_join(abspref_coef, by = c("model", "model_family", "model_params", "model_label")) |>
  left_join(ppl |> select(model, perplexity), by = "model") |>
  filter(!is.na(perplexity)) |>
  mutate(
    params_M     = as.numeric(sub("M$", "", model_params)),
    model_family = factor(model_family, levels = c("GPT-2", "GPT-Neo", "OPT", "OLMo"))
  ) |>
  arrange(model_family, params_M)

message("\nResults:")
print(results |> select(model_family, model_label, model_params, perplexity, delta_ll, estimate),
      n = Inf)

# ── Plot aesthetics ───────────────────────────────────────────────────────────
family_colours <- c("GPT-2" = "#2166ac", "GPT-Neo" = "#4dac26", "OPT" = "#b2182b", "OLMo" = "#d6604d")

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

# ── Left panel: ΔLL ──────────────────────────────────────────────────────────
p_left <- ggplot(filter(results, !is.na(delta_ll)),
                 aes(x = perplexity, y = delta_ll, colour = model_family)) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = FALSE,
              linetype = "dashed", linewidth = 0.9,
              aes(group = model_family)) +
  geom_point(size = 3, alpha = 0.9) +
  geom_text(aes(label = model_params),
            hjust = -0.12, vjust = 0.4,
            size = 3, fontface = "bold", show.legend = FALSE) +
  scale_y_continuous("\u0394LL") +
  labs(
    title    = "\u0394LL vs. validation perplexity",
    subtitle = paste0(
      "\u0394LL = logLik(resp_alpha \u223c ",
      ifelse(use_pile_freq, "Pile freq [OPT] / Google freq [GPT]", "Google Books RelFreq"),
      " + model pref) \u2212 logLik(baseline). Dashed = quadratic fit per family."
    )
  ) +
  shared_layers

# ── Right panel: AbsPref coefficient ─────────────────────────────────────────
p_right <- ggplot(filter(results, !is.na(estimate)),
                  aes(x = perplexity, y = estimate, colour = model_family,
                      fill = model_family)) +
  geom_hline(yintercept = 0, linetype = "dotted", colour = "grey50") +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = TRUE,
              linetype = "dashed", linewidth = 0.9, alpha = 0.15,
              aes(group = model_family)) +
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

# ── Combine and save ──────────────────────────────────────────────────────────
p_combined <- p_left + p_right +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

print(p_combined)

ggsave(OUT_PDF, p_combined, width = 14, height = 5.5)
ggsave(OUT_PNG, p_combined, width = 14, height = 5.5, dpi = 150)
message(sprintf("\nSaved:\n  %s\n  %s", OUT_PDF, OUT_PNG))
