#!/usr/bin/env Rscript
# fig1_options.R ──────────────────────────────────────────────────────────────
# Option C extended to three panels:
#   Left   – scoring computation table (real per-prompt log-probs)
#   Middle – preference scores for example binomials
#   Right  – preference deviation for the same binomials
#
# Reads real data from:
#   ../Data/checkpoint_results/opt-c4-125m-seed964_step-12192.csv
#   ../Data/processed/final_attested.csv
#
# In RStudio: source this file.  Non-interactive: saves fig1_option_c.pdf.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(readr)
  library(tibble)
})

G <- "#154733"   # Oregon green — alpha ordering / positive values
R <- "#F0B323"   # UO Gold      — non-alpha / negative values / result highlight

# ── Resolve paths relative to this script ────────────────────────────────────
SCRIPT_DIR <- if (interactive() && requireNamespace("rstudioapi", quietly = TRUE) &&
                  rstudioapi::isAvailable()) {
  dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path))
} else {
  args     <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg)))
  else normalizePath(".")
}

CKPT_CSV   <- file.path(SCRIPT_DIR,
  "../Data/checkpoint_results/opt-c4-125m-seed964_step-12192.csv")
FINAL_CSV  <- file.path(SCRIPT_DIR,
  "../Data/processed/final_attested.csv")

# ── Panel 1 data: per-prompt log-probs for "bread and butter" ─────────────────
message("Reading checkpoint results...")
ckpt <- read_csv(CKPT_CSV, show_col_types = FALSE)

bb_all <- ckpt |> filter(binom == "bread and butter")
mean_pref <- mean(bb_all$preference)

# Pick four short, recognisable prompts to display.
# Use trimws() on both sides so trailing spaces in the CSV don't break matching.
show_prompts <- c(" ", "Well, ", "Or even ", "Maybe a ")
bb_table <- bb_all |>
  filter(trimws(prompt) %in% trimws(show_prompts)) |>
  arrange(match(trimws(prompt), trimws(show_prompts)))

# If any target prompts were missing, fall back to the first four rows available
if (nrow(bb_table) == 0) {
  warning("None of the target prompts found; using first 4 rows instead.")
  bb_table <- head(bb_all, 4)
} else if (nrow(bb_table) < 4) {
  warning(sprintf("Only %d of 4 target prompts found.", nrow(bb_table)))
}

# ── Panel 2 & 3 data: preference + preference deviation ───────────────────────
message("Reading final_attested.csv...")
final <- read_csv(FINAL_CSV, show_col_types = FALSE)

c4_125m <- final |>
  filter(grepl("c4-125m", model, ignore.case = TRUE)) |>
  filter(step == max(step)) |>
  mutate(
    freq_log_odds = qlogis(RelFreq),          # ln(c_alpha / c_nonalpha)
    pref_dev      = preference - freq_log_odds
  )

# Automatically pick 5 binomials with a nice spread across the preference axis.
# Require Attested == TRUE and abs(RelFreq - 0.5) > 0.05 (some frequency signal).
candidates <- c4_125m |>
  filter(Attested == "True" | Attested == TRUE) |>
  filter(abs(RelFreq - 0.5) > 0.05) |>
  arrange(preference)

n   <- nrow(candidates)
idx <- unique(round(seq(1, n, length.out = 5)))
examples <- candidates[idx, ]

message(sprintf("Example binomials selected: %s",
                paste(examples$binom, collapse = ", ")))

# ── Panel 1: Scoring computation table ───────────────────────────────────────
p_left <- local({
  # Build display rows via transmute — avoids size-mismatch if bb_table is small
  body <- bb_table |>
    transmute(
      y      = rev(seq_len(n())),   # top row = highest y
      prompt = case_when(
        trimws(prompt) == "" ~ "(empty)",
        TRUE                 ~ sprintf('"%s"', trimws(prompt))
      ),
      lp_a   = sprintf("%.2f", alpha_logprob),
      lp_n   = sprintf("%.2f", nonalpha_logprob),
      diff   = sprintf("%+.2f", preference)
    )

  # Compact vertical spacing: rows at 0.65 unit intervals
  ROW_H  <- 0.50
  body   <- body |> mutate(y = rev(seq_len(n())) * ROW_H)

  ellipsis_y <- 0
  more_label  <- sprintf("... (%d more prompts)", nrow(bb_all) - nrow(bb_table))

  X_PFX <- 0; X_A <- 3.9; X_N <- 5.7; X_D <- 7.2
  Y_HDR  <- max(body$y) + 0.50
  Y_SEP1 <- max(body$y) + 0.27
  Y_SEP2 <- ellipsis_y - 0.27
  Y_AVG  <- ellipsis_y - 0.60

  ggplot() +
    # Column headers
    annotate("text", x = X_PFX, y = Y_HDR, label = "Prompt prefix",
             hjust = 0,   fontface = "bold", size = 3.6) +
    annotate("text", x = X_A,   y = Y_HDR, label = "bread and butter",
             hjust = 0.5, fontface = "bold", size = 3.4, colour = G) +
    annotate("text", x = X_N,   y = Y_HDR, label = "butter and bread",
             hjust = 0.5, fontface = "bold", size = 3.4, colour = R) +
    annotate("text", x = X_D,   y = Y_HDR, label = "\u0394 log P",
             hjust = 0.5, fontface = "bold", size = 3.6) +
    annotate("segment", x = -0.2, xend = 8.2, y = Y_SEP1, yend = Y_SEP1,
             colour = "grey50", linewidth = 0.4) +
    # Body rows
    geom_text(data = body, aes(x = X_PFX, y = y, label = prompt),
              hjust = 0,   size = 3.4) +
    geom_text(data = body, aes(x = X_A,   y = y, label = lp_a),
              hjust = 0.5, size = 3.4, colour = G) +
    geom_text(data = body, aes(x = X_N,   y = y, label = lp_n),
              hjust = 0.5, size = 3.4, colour = R) +
    geom_text(data = body, aes(x = X_D,   y = y, label = diff),
              hjust = 0.5, size = 3.4) +
    # Ellipsis / "more prompts" note
    annotate("text", x = X_PFX, y = ellipsis_y, label = more_label,
             hjust = 0, size = 3.0, colour = "grey50", fontface = "italic") +
    # Footer separator + average row
    annotate("segment", x = -0.2, xend = 8.2, y = Y_SEP2, yend = Y_SEP2,
             colour = "grey50", linewidth = 0.4) +
    annotate("text", x = X_PFX, y = Y_AVG,
             label = sprintf("preference  =  mean(\u0394 log P)  =  %.2f", mean_pref),
             hjust = 0, fontface = "bold", size = 3.5) +
    annotate("text", x = X_D, y = Y_AVG,
             label = sprintf("%+.2f", mean_pref),
             hjust = 0.5, fontface = "bold", size = 4.0, colour = G) +
    xlim(-0.3, 8.7) +
    ylim(Y_AVG - 0.4, Y_HDR + 0.35) +
    theme_void(base_size = 11) +
    theme(plot.background = element_rect(fill = "white", colour = NA),
          plot.margin     = margin(6, 8, 6, 8))
})

# ── Shared helper: horizontal lollipop chart ─────────────────────────────────
# Binomials on the y-axis (ordered by score), score on the x-axis.
# A stem from 0 to the value and a coloured dot eliminates all text overlap.

lollipop <- function(ex, x_var, x_lab, neg_label, pos_label) {
  ex <- ex |> arrange(.data[[x_var]]) |>
    mutate(binom = factor(binom, levels = binom),
           score = .data[[x_var]])

  x_range  <- range(ex$score)
  x_pad    <- max(diff(x_range) * 0.15, 0.5)
  x_limits <- c(x_range[1] - x_pad, x_range[2] + x_pad)

  ggplot(ex, aes(x = score, y = binom)) +
    geom_vline(xintercept = 0, linetype = "dashed",
               colour = "grey65", linewidth = 0.4) +
    geom_segment(aes(x = 0, xend = score, yend = binom),
                 colour = "grey78", linewidth = 0.7) +
    geom_point(aes(colour = score > 0), size = 3.2) +
    geom_text(aes(label = sprintf("%.2f", score),
                  colour = score > 0,
                  hjust  = ifelse(score >= 0, -0.35, 1.35)),
              size = 3.2, fontface = "bold") +
    scale_colour_manual(values = c(R, G), guide = "none") +
    scale_x_continuous(name = x_lab, limits = x_limits) +
    scale_y_discrete(name = NULL) +
    annotate("text", x = x_range[1] - x_pad * 0.55, y = Inf,
             label = neg_label, hjust = 0, vjust = 1.8,
             size = 2.9, colour = R, fontface = "italic") +
    annotate("text", x = x_pad * 0.1, y = Inf,
             label = pos_label, hjust = 0, vjust = 1.8,
             size = 2.9, colour = G, fontface = "italic") +
    theme_classic(base_size = 11) +
    theme(
      axis.line.x     = element_line(colour = "grey60", linewidth = 0.4),
      axis.line.y     = element_blank(),
      axis.ticks.y    = element_blank(),
      axis.text.y     = element_text(colour = "grey20", size = 10,
                                     margin = margin(r = 2)),
      axis.text.x     = element_text(colour = "grey30", size = 9),
      axis.title.x    = element_text(colour = "grey20", size = 10,
                                     margin = margin(t = 4)),
      panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.3),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin     = margin(4, 10, 6, 4)
    )
}

p_mid <- lollipop(
  examples, "preference",
  x_lab     = "Preference  (log P(\u03b1) \u2212 log P(non-\u03b1), avg. over 51 prefixes)",
  neg_label = "\u2190 prefers non-\u03b1 order",
  pos_label = "prefers \u03b1 order \u2192"
)

p_right <- lollipop(
  examples, "pref_dev",
  x_lab     = "Preference deviation  (preference \u2212 log relative frequency ratio)",
  neg_label = "\u2190 below freq. prediction",
  pos_label = "above freq. prediction \u2192"
)

# ── Assemble ──────────────────────────────────────────────────────────────────
# Stack the two lollipop panels vertically on the right; a top spacer pushes
# them toward the bottom of the figure to align with the table footer.

right_col <- (p_mid / p_right / plot_spacer()) +
  plot_layout(heights = c(1, 1, 0.15))

fig1 <- (p_left | right_col) +
  plot_layout(widths = c(1.7, 1)) +
  plot_annotation(
    tag_levels = "a",
    theme = theme(
      plot.tag        = element_text(face = "bold", size = 10),
      plot.background = element_rect(fill = "white", colour = "grey82"),
      plot.margin     = margin(10, 12, 10, 12)
    )
  )

# ── Display ───────────────────────────────────────────────────────────────────
if (exists("QUARTO_BUILD") && isTRUE(QUARTO_BUILD)) {
  # called from writeup.qmd — caller handles printing
} else if (interactive()) {
  print(fig1)
  message("Done.")
} else {
  outfile <- file.path(SCRIPT_DIR, "fig1_option_c.pdf")
  pdf(outfile, width = 15, height = 5.5)
  print(fig1)
  dev.off()
  message("Saved: ", outfile)
}
