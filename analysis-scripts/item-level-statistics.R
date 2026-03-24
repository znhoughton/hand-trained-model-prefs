# item-level-statistics.R
#
# Computes summary statistics for the stimulus set:
#   - Total number of attested binomials
#   - Number with > 0 count in BOTH orderings (alpha and beta)
#   - Mean and SD of log overall frequency after that exclusion
#
# Uses the step-exposure files (final step) for corpus counts, since the
# alpha_c4_raw / beta_c4_raw columns in binomial_corpus_counts.csv are empty.
#
# Outputs a LaTeX table (printed to console).

library(tidyverse)

# ── Load stimulus list (attested flag) ───────────────────────────────────────

stim <- read_csv("../Data/nonce_and_attested_binoms.csv", show_col_types = FALSE)
attested_binoms <- stim |> filter(Attested == 1) |> pull(Alpha)

# ── Helper: extract final-step counts from a step-exposures file ─────────────

final_counts <- function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  df |>
    filter(tokens == max(tokens)) |>
    filter(binom %in% attested_binoms) |>
    select(binom, alpha_seen, beta_seen)
}

# ── BabyLM ───────────────────────────────────────────────────────────────────

babylm <- final_counts("../Data/babylm_step_exposures.csv")

babylm_total  <- length(attested_binoms)
babylm_both   <- babylm |> filter(alpha_seen > 0, beta_seen > 0)
babylm_n_both <- nrow(babylm_both)
babylm_logfreq <- log(babylm_both$alpha_seen + babylm_both$beta_seen)
babylm_mean   <- mean(babylm_logfreq)
babylm_sd     <- sd(babylm_logfreq)

# ── C4 ───────────────────────────────────────────────────────────────────────

c4 <- final_counts("../Data/c4_step_exposures.csv")

c4_total  <- length(attested_binoms)
c4_both   <- c4 |> filter(alpha_seen > 0, beta_seen > 0)
c4_n_both <- nrow(c4_both)
c4_logfreq <- log(c4_both$alpha_seen + c4_both$beta_seen)
c4_mean   <- mean(c4_logfreq)
c4_sd     <- sd(c4_logfreq)

# ── Print summary to console ─────────────────────────────────────────────────

cat("\n=== Summary Statistics ===\n\n")
cat(sprintf("BabyLM: %d total attested | %d with both orderings > 0 | log freq: M = %.2f, SD = %.2f\n",
            babylm_total, babylm_n_both, babylm_mean, babylm_sd))
cat(sprintf("C4:     %d total attested | %d with both orderings > 0 | log freq: M = %.2f, SD = %.2f\n",
            c4_total, c4_n_both, c4_mean, c4_sd))

# ── LaTeX table ──────────────────────────────────────────────────────────────

cat('\n=== LaTeX Table ===\n\n')

latex <- sprintf(
'\\begin{table}[h]
\\centering
\\caption{Summary statistics for attested binomials in each training corpus.
  \\textit{Both orderings attested} counts binomials for which both the
  alphabetical and non-alphabetical orderings appear at least once in the corpus.
  Log overall frequency is computed as $\\ln(n_{\\alpha} + n_{\\beta})$ after
  applying this exclusion, where $n_{\\alpha}$ and $n_{\\beta}$ are raw corpus
  counts for each ordering.}
\\label{tab:corpus-stats}
\\begin{tabular}{lcc}
\\toprule
 & BabyLM & C4 \\\\
\\midrule
Total attested binomials & %d & %d \\\\
Both orderings attested ($n > 0$) & %d & %d \\\\
\\addlinespace
\\multicolumn{3}{l}{\\textit{After exclusion:}} \\\\
\\quad Log overall frequency (mean) & %.2f & %.2f \\\\
\\quad Log overall frequency (SD)   & %.2f & %.2f \\\\
\\bottomrule
\\end{tabular}
\\end{table}',
  babylm_total, c4_total,
  babylm_n_both, c4_n_both,
  babylm_mean, c4_mean,
  babylm_sd, c4_sd
)

cat(latex, "\n")
