#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: Rscript analysis/plot_layers.R INPUT_CSV_DIR OUTPUT_DIR")
figure_dir <- args[[2]]
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
data <- read.csv(file.path(args[[1]], "unique_layer_selection.csv"), stringsAsFactors = FALSE)

ink <- "#202020"
muted <- "#666666"
grid <- "#dedede"
blue <- "#285caa"
orange <- "#cc611b"

draw <- function() {
  par(mar = c(3.7, 4.2, 1.5, 1.0), mgp = c(2.25, 0.65, 0), tcl = -0.25)
  plot(
    data$layer,
    data$validation_r2,
    type = "n",
    xlim = c(0, 31),
    ylim = c(0.54, 0.79),
    axes = FALSE,
    xlab = "Decoder block (zero-indexed)",
    ylab = expression(R^2 ~ "against four-rollout mean"),
    cex.lab = 1.02
  )
  abline(h = seq(0.55, 0.75, 0.05), col = grid, lwd = 0.8)
  abline(v = 18, col = adjustcolor(ink, alpha.f = 0.28), lty = 3, lwd = 1.3)
  axis(1, at = c(seq(0, 28, 4), 31), col = NA, col.axis = muted, cex.axis = 0.90)
  axis(2, at = seq(0.55, 0.75, 0.05), las = 1, col = NA, col.axis = muted, cex.axis = 0.90)
  box(col = grid)
  title("Qwen3.5 layer selection", line = 0.28, cex.main = 1.02)

  lines(data$layer, data$validation_r2, col = blue, lwd = 2.5)
  points(data$layer, data$validation_r2, col = blue, bg = "white", pch = 21, cex = 0.82, lwd = 1.25)
  lines(data$layer, data$test_r2, col = orange, lwd = 2.0, lty = 2)
  points(data$layer, data$test_r2, col = orange, pch = 16, cex = 0.52)

  selected <- data[data$layer == 18, ]
  points(18, selected$validation_r2, pch = 21, bg = blue, col = ink, cex = 1.45, lwd = 1.4)
  text(18.6, selected$validation_r2 + 0.01, "selected: layer 18", adj = 0,
       cex = 0.90, font = 2, col = ink)

  legend(
    "bottomright",
    legend = c("Validation (selection criterion)", "Held-out test (reported after selection)"),
    col = c(blue, orange),
    lty = c(1, 2),
    lwd = c(2.5, 2.0),
    pch = c(21, 16),
    pt.bg = c("white", orange),
    pt.cex = c(0.82, 0.52),
    bty = "n",
    cex = 0.88,
    text.col = ink
  )
}

svg(
  file.path(figure_dir, "layer_selection_10k.svg"),
  width = 7.2,
  height = 3.35,
  pointsize = 10,
  bg = "white",
  family = "sans"
)
draw()
dev.off()

pdf(
  file.path(figure_dir, "layer_selection_10k.pdf"),
  width = 7.2,
  height = 3.35,
  pointsize = 10,
  bg = "white",
  family = "Helvetica",
  useDingbats = FALSE
)
draw()
dev.off()
