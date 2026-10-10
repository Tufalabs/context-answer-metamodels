#!/usr/bin/env Rscript

# Plot scaling, representation, and behavior results from exported metrics.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: Rscript analysis/plot_results.R INPUT_CSV_DIR OUTPUT_DIR")
input_dir <- normalizePath(args[[1]], mustWork = TRUE)
fig_dir <- args[[2]]
file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

blue <- "#277DA1"
orange <- "#E76F51"
green <- "#2A9D8F"
purple <- "#7B6DAD"
gray <- "#7A7A7A"
light_gray <- "#D9D9D9"
ink <- "#222222"

# Semantic model-family colors used wherever family, rather than domain or
# conditioning model, is the color encoding.
linear_col <- gray
mlp_col <- blue
gaussian_col <- green
flow_col <- orange

full_prompt_ticks <- c(100, 1000, 10000, 100000, 500000)
full_prompt_labels <- c("100", "1k", "10k", "100k", "500k")
full_prompt_xlim <- range(log10(full_prompt_ticks))
late_prompt_ticks <- c(10000, 50000, 100000, 500000)
late_prompt_labels <- c("10k", "50k", "100k", "500k")
late_prompt_xlim <- range(log10(late_prompt_ticks))

save_plot <- function(stem, width, height, draw) {
  pdf(file.path(fig_dir, paste0(stem, ".pdf")), width = width,
      height = height, family = "Helvetica", useDingbats = FALSE,
      paper = "special")
  draw()
  dev.off()
  svg(file.path(fig_dir, paste0(stem, ".svg")), width = width,
      height = height, pointsize = 10, family = "sans")
  draw()
  dev.off()
}

axis_plain <- function(side, at = NULL, labels = TRUE, ...) {
  axis(side, at = at, labels = labels, col = ink, col.axis = ink,
       lwd = 0.7, tck = -0.025, ...)
}

# ---------------------------------------------------------------------------
# Main Figure 2: matched method and domain scaling.
# ---------------------------------------------------------------------------
domain_methods_path <- file.path(input_dir, "unique_main_metrics.csv")
domain_methods <- read.csv(domain_methods_path)

# Auxiliary distribution-fit diagnostics are integrated into the corresponding
# primary Gaussian/flow scaling figures below.
domain_aux_path <- file.path(input_dir, "unique_main_auxiliary.csv")
cross_aux_path <- file.path(input_dir, "unique_cross_qwen35_auxiliary.csv")
if (!file.exists(domain_aux_path) || !file.exists(cross_aux_path)) {
  stop("Integrated distribution figures require both auxiliary-metrics files")
}
domain_aux <- read.csv(domain_aux_path)
cross_aux <- read.csv(cross_aux_path)
aux_panels <- list(
  list(field = "nll_raw", title = "NLL per dimension",
       ylab = "NLL / dimension"),
  list(field = "predictive_mmd2",
       title = expression("Predictive " * plain(MMD)^2),
       ylab = expression(plain(MMD)^2)),
  list(field = "predictive_sliced_w2", title = "PCA sliced-Wasserstein",
       ylab = expression("Sliced " * W[2]))
)
# Preserve the original shared auxiliary-axis limits and ticks exactly.
aux_metric_axes <- dget(file.path(script_dir, "axes.R"))

# Keep auxiliary-panel titles in one regular-weight sans-serif font; the text
# superscript avoids plotmath switching fonts within the title.
aux_panel_title <- function(prefix, panel) {
  if (panel$field == "predictive_mmd2") {
    return(paste0(prefix, ": Predictive MMD\u00b2"))
  }
  paste0(prefix, ": ", panel$title)
}

draw_scaling <- function() {
  old <- par(no.readonly = TRUE); on.exit(par(old))
  layout(matrix(c(1:3, 4, 4, 4), nrow = 2, byrow = TRUE),
         heights = c(1, 0.19), widths = c(1.25, 1, 1))
  par(oma = c(0, 0, 0.50, 0), mgp = c(1.62, 0.38, 0), las = 1,
      family = "sans", cex = 0.78)
  domains <- c("LMSYS", "WeirdChat", "IFEval")
  scaling_label_ticks <- c(100, 1000, 10000, 500000)
  scaling_label_text <- c("100", "1k", "10k", "500k")
  family_specs <- list(
    linear = list(col = linear_col, pch = 15),
    mlp = list(col = mlp_col, pch = 16),
    flow = list(col = flow_col, pch = 17),
    gaussian = list(col = gaussian_col, pch = 1)
  )
  metric_rows <- list(
    list(field = "mean_r2", families = c("linear", "mlp", "flow"),
         ylim = c(-0.28, 0.96), yticks = c(-0.2, 0.2, 0.6, 0.9),
         ylab = expression("Mean " * R^2))
  )
  for (row in seq_along(metric_rows)) {
    metric <- metric_rows[[row]]
    for (j in seq_along(domains)) {
      par(mar = c(2.30, if (j == 1) 3.85 else 0.55, 1.25, 0.40))
      plot(NA, xlim = full_prompt_xlim + c(-0.035, 0.16),
           ylim = metric$ylim, axes = FALSE, xlab = "", ylab = "")
      abline(h = metric$yticks, col = "#EEEEEE", lwd = 0.7)
      axis_plain(1, at = log10(full_prompt_ticks), labels = FALSE)
      mtext(scaling_label_text, side = 1, at = log10(scaling_label_ticks),
            line = 0.30, cex = 0.82)
      if (j == 1) axis_plain(2, at = metric$yticks)
      box(col = ink, lwd = 0.7)
      if (j == 1) {
        mtext(metric$ylab, side = 2, line = 2.20, cex = 0.88, las = 0)
      }
      if (j == 2) {
        mtext("LMSYS training prompts", side = 1, line = 1.15, cex = 0.84)
      }
      for (family in metric$families) {
        d <- domain_methods[domain_methods$domain == domains[j] &
                              domain_methods$family == family, ]
        d <- d[order(d$n_train), ]
        spec <- family_specs[[family]]
        lines(log10(d$n_train), d[[metric$field]], col = spec$col, lwd = 1.65)
        points(log10(d$n_train), d[[metric$field]], col = spec$col,
               pch = spec$pch, cex = 0.66)
      }
      title(domains[j], line = 0.12, cex.main = 0.88)
    }
  }
  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend("center",
         c("Linear", "MLP", "Flow mean"),
         col = c(linear_col, mlp_col, flow_col),
         lty = rep(1, 3), pch = c(15, 16, 17),
         lwd = 1.6, bty = "n", cex = 0.82, ncol = 3, x.intersp = 0.65)
}
save_plot("main_scaling_domain", 7.05, 2.55, draw_scaling)

# ---------------------------------------------------------------------------
# Supplement: prompt retrieval across domains.
# ---------------------------------------------------------------------------
draw_domain_retrieval_scaling <- function() {
  bank_sizes <- c(LMSYS = 3000, WeirdChat = 2660, IFEval = 541)
  stopifnot("n_retrieval_candidates" %in% names(domain_methods),
            all(domain_methods$n_retrieval_candidates ==
                bank_sizes[domain_methods$domain]))
  old <- par(no.readonly = TRUE); on.exit(par(old))
  layout(matrix(c(1:3, 4, 4, 4), nrow = 2, byrow = TRUE),
         heights = c(1, 0.21), widths = rep(1, 3))
  par(oma = c(0, 0.55, 0.35, 0), mgp = c(1.62, 0.38, 0), las = 1,
      family = "sans", cex = 0.84)
  domains <- c("LMSYS", "WeirdChat", "IFEval")
  scaling_label_ticks <- c(100, 1000, 10000, 500000)
  scaling_label_text <- c("100", "1k", "10k", "500k")
  family_specs <- list(
    linear = list(col = linear_col, pch = 15),
    mlp = list(col = mlp_col, pch = 16),
    flow = list(col = flow_col, pch = 17)
  )
  for (j in seq_along(domains)) {
    par(mar = c(2.35, 2.75, 1.40, 0.40))
    plot(NA, xlim = full_prompt_xlim + c(-0.035, 0.16),
         ylim = c(0, 1), axes = FALSE, xlab = "", ylab = "")
    abline(h = seq(0, 1, 0.25), col = "#EEEEEE", lwd = 0.7)
    axis_plain(1, at = log10(full_prompt_ticks), labels = FALSE)
    mtext(scaling_label_text, side = 1, at = log10(scaling_label_ticks),
          line = 0.30, cex = 0.82)
    if (j == 1) axis_plain(2, at = seq(0, 1, 0.25))
    box(col = ink, lwd = 0.7)
    if (j == 1) {
      mtext("Top-1 prompt retrieval", side = 2, line = 2.15,
            cex = 0.92, las = 0)
    }
    if (j == 2) {
      mtext("LMSYS training prompts", side = 1, line = 1.20, cex = 0.90)
    }
    for (family in c("linear", "mlp", "flow")) {
      d <- domain_methods[domain_methods$domain == domains[j] &
                            domain_methods$family == family, ]
      d <- d[order(d$n_train), ]
      spec <- family_specs[[family]]
      lines(log10(d$n_train), d$retrieval, col = spec$col, lwd = 1.65)
      points(log10(d$n_train), d$retrieval, col = spec$col,
             pch = spec$pch, cex = 0.66)
    }
    title(paste0(domains[j], " (n = ", bank_sizes[domains[j]], ")"),
          line = 0.25, cex.main = 0.92, font.main = 1)
  }
  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend("center", c("Linear", "MLP", "Flow mean"),
         col = c(linear_col, mlp_col, flow_col),
         lty = rep(1, 3), pch = c(15, 16, 17),
         lwd = 1.6, bty = "n", cex = 0.86, ncol = 3, x.intersp = 0.65)
}
save_plot("supp_domain_retrieval_scaling", 7.05, 2.65,
          draw_domain_retrieval_scaling)

# ---------------------------------------------------------------------------
# Supplement: Gaussian and flow distribution scaling across domains.
# ---------------------------------------------------------------------------
draw_domain_distribution_scaling <- function() {
  old <- par(no.readonly = TRUE); on.exit(par(old))
  layout(matrix(c(1:12, 13, 13, 13), nrow = 5, byrow = TRUE),
         heights = c(1, 1, 1, 1.08, 0.20), widths = rep(1, 3))
  par(oma = c(0, 0.55, 0.35, 0), mgp = c(1.62, 0.38, 0), las = 1,
      family = "sans", cex = 0.78)
  domains <- c("LMSYS", "WeirdChat", "IFEval")
  scaling_label_ticks <- c(100, 1000, 10000, 500000)
  scaling_label_text <- c("100", "1k", "10k", "500k")
  family_specs <- list(
    gaussian = list(col = gaussian_col, pch = 1),
    flow = list(col = flow_col, pch = 17)
  )
  metric_rows <- list(
    list(field = "energy", ylim = c(0.045, 0.23),
         yticks = c(0.05, 0.10, 0.15, 0.20),
         ylab = expression("Energy score" / sqrt(d))),
    list(field = "variance_spearman", ylim = c(-0.25, 0.95),
         yticks = c(-0.2, 0.2, 0.6, 0.9),
         ylab = "Variance Spearman"),
    list(field = "variance_ratio", ylim = c(0, 8.2),
         yticks = c(1, 3, 5, 7), ylab = "Variance ratio")
  )
  for (row in seq_along(metric_rows)) {
    metric <- metric_rows[[row]]
    for (j in seq_along(domains)) {
      par(mar = c(if (row == length(metric_rows)) 0.55 else 0.25,
                  2.75, if (row == 1) 1.45 else 0.25, 0.40))
      plot(NA, xlim = full_prompt_xlim + c(-0.035, 0.16),
           ylim = metric$ylim, axes = FALSE, xlab = "", ylab = "")
      abline(h = metric$yticks, col = "#EEEEEE", lwd = 0.7)
      if (metric$field == "variance_ratio") {
        abline(h = 1, col = gray, lty = 3, lwd = 0.8)
      }
      if (row == length(metric_rows)) {
        axis_plain(1, at = log10(full_prompt_ticks), labels = FALSE)
      }
      if (j == 1) axis_plain(2, at = metric$yticks)
      box(col = ink, lwd = 0.7)
      if (j == 1) {
        mtext(metric$ylab, side = 2, line = 1.85, cex = 0.92, las = 0)
      }
      for (family in c("gaussian", "flow")) {
        d <- domain_methods[domain_methods$domain == domains[j] &
                              domain_methods$family == family, ]
        d <- d[order(d$n_train), ]
        spec <- family_specs[[family]]
        lines(log10(d$n_train), d[[metric$field]], col = spec$col, lwd = 1.65)
        points(log10(d$n_train), d[[metric$field]], col = spec$col,
               pch = spec$pch, cex = 0.66)
      }
      if (row == 1) {
        title(domains[j], line = 0.25, cex.main = 0.92, font.main = 1)
      }
    }
  }
  for (i in seq_along(aux_panels)) {
    p <- aux_panels[[i]]
    metric_axis <- aux_metric_axes[[i]]
    par(mar = c(2.50, 2.75, 1.75, 0.40))
    plot(NA, xlim = full_prompt_xlim + c(-0.035, 0.16),
         ylim = metric_axis$ylim, axes = FALSE, xlab = "", ylab = "")
    abline(h = metric_axis$yticks, col = "#EEEEEE", lwd = 0.7)
    axis_plain(1, at = log10(full_prompt_ticks), labels = FALSE)
    mtext(scaling_label_text, side = 1, at = log10(scaling_label_ticks),
          line = 0.30, cex = 0.82)
    axis_plain(2, at = metric_axis$yticks)
    box(col = ink, lwd = 0.7)
    mtext(p$ylab, side = 2, line = 1.85, cex = 0.90, las = 0)
    if (i == 2) {
      mtext("LMSYS training prompts", side = 1, line = 1.25, cex = 0.90)
    }
    for (family in c("gaussian", "flow")) {
      d <- domain_aux[domain_aux$family == family, ]
      d <- d[order(d$n_train), ]
      spec <- family_specs[[family]]
      lines(log10(d$n_train), d[[p$field]], col = spec$col, lwd = 1.65)
      points(log10(d$n_train), d[[p$field]], col = spec$col,
             pch = spec$pch, cex = 0.66)
    }
    title(aux_panel_title("LMSYS", p), line = 0.40, cex.main = 0.88,
          font.main = 1)
  }
  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend_labels <- c("Gaussian", "Flow")
  legend("center", legend_labels,
         col = c(gaussian_col, flow_col), lty = c(1, 1),
         pch = c(1, 17), lwd = 1.6, bty = "n", cex = 0.86,
         horiz = TRUE, x.intersp = 0.65,
         text.width = strwidth(paste0(legend_labels, "    "), cex = 0.86))
}
save_plot("supp_domain_distribution_scaling", 7.05, 8.80,
          draw_domain_distribution_scaling)

# ---------------------------------------------------------------------------
# Main Figure 3: absolute cross-model and matched-model forecast metrics.
# ---------------------------------------------------------------------------
cross_path <- file.path(input_dir, "unique_cross_qwen35.csv")
cross_model <- read.csv(cross_path)
cross_sizes <- sort(unique(cross_model$n_train_contexts))
cross_xticks <- c(100, 1000, 10000, 100000)
cross_xlim <- log10(c(100, 100000)) + c(-0.04, 0.08)

draw_combined_cross_model <- function() {
  old <- par(no.readonly = TRUE); on.exit(par(old))
  layout(matrix(c(1:6, 7, 7, 7), nrow = 3, byrow = TRUE),
         heights = c(1, 1, 0.18), widths = c(1.25, 1, 1))
  par(mgp = c(1.85, 0.45, 0), las = 1, family = "sans", cex = 0.78)
  targets <- list(
    list(name = "Qwen3.5", metrics = cross_model, conditions = c(3584, 4096)),
    list(name = "Gemma 4",
         metrics = read.csv(file.path(input_dir, "unique_cross_gemma.csv")),
         conditions = c(3584, 3840))
  )
  families <- c("linear", "mlp", "flow")
  titles <- c("Linear mean", "MLP mean", "Flow mean")
  condition_colors <- c(purple, blue)
  condition_symbols <- c(16, 17)
  for (row in seq_along(targets)) {
    target <- targets[[row]]
    sizes <- sort(unique(target$metrics$n_train_contexts))
    for (column in seq_along(families)) {
      par(mar = c(if (row == 2) 2.55 else 0.65,
                  if (column == 1) 4.75 else 0.55,
                  if (row == 1) 1.55 else 1.0, 0.40))
      plot(NA, xlim = cross_xlim, ylim = c(0, 0.93), axes = FALSE,
           xlab = "", ylab = "")
      abline(h = seq(0, 0.8, 0.2), col = "#EEEEEE", lwd = 0.7)
      axis_plain(1, at = log10(cross_xticks),
                 labels = if (row == 2) c("100", "1k", "10k", "100k") else FALSE)
      if (column == 1) {
        axis_plain(2, at = seq(0, 0.8, 0.2))
        mtext(expression("Mean " * R^2), side = 2, line = 2.05, cex = 0.88, las = 0)
        mtext(paste(target$name, "target"), side = 2, line = 3.45,
              cex = 0.88, las = 0, font = 2)
      }
      box(col = ink, lwd = 0.7)
      if (row == 1) title(titles[column], line = 0.25, cex.main = 0.88)
      if (row == 2 && column == 2) {
        mtext("LMSYS training prompts", side = 1, line = 1.45, cex = 0.84)
      }
      for (condition in seq_along(target$conditions)) {
        means <- deviations <- numeric(length(sizes))
        for (scale in seq_along(sizes)) {
          values <- target$metrics$sample_mean_r2[
            target$metrics$family == families[column] &
            target$metrics$condition_hidden == target$conditions[condition] &
            target$metrics$n_train_contexts == sizes[scale]]
          stopifnot(length(values) > 0, all(is.finite(values)))
          means[scale] <- mean(values)
          deviations[scale] <- if (length(values) > 1) sd(values) else 0
        }
        stopifnot(all(means - deviations >= 0), all(means + deviations <= 0.93))
        lines(log10(sizes), means, col = condition_colors[condition], lwd = 1.7)
        points(log10(sizes), means, col = condition_colors[condition],
               pch = condition_symbols[condition], cex = 0.66)
        segments(log10(sizes), means - deviations, log10(sizes), means + deviations,
                 col = condition_colors[condition], lwd = 0.7)
      }
    }
  }
  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend("center", c("Qwen2.5 conditioner", "Target-model conditioner"),
         col = condition_colors, lty = 1, lwd = 1.7, pch = condition_symbols,
         bty = "n", cex = 0.82, horiz = TRUE, x.intersp = 0.55)
}
save_plot("main_cross_model_scaling", 7.05, 3.80, draw_combined_cross_model)

# ---------------------------------------------------------------------------
# Supplement: cross-model prompt retrieval.
# ---------------------------------------------------------------------------
draw_cross_model_retrieval <- function(metrics = cross_model,
    conditions = c(3584, 4096),
    condition_labels = c("Qwen2.5 -> Qwen3.5", "Qwen3.5 -> Qwen3.5")) {
  stopifnot("n_retrieval_candidates" %in% names(metrics),
            all(metrics$n_retrieval_candidates == 3000))
  cross_sizes <- sort(unique(metrics$n_train_contexts))
  old <- par(no.readonly = TRUE); on.exit(par(old))
  layout(matrix(c(1:3, 4, 4, 4), nrow = 2, byrow = TRUE),
         heights = c(1, 0.20), widths = rep(1, 3))
  par(oma = c(0, 0.55, 0.30, 0), mgp = c(1.70, 0.42, 0), las = 1,
      family = "sans", cex = 0.82)
  condition_cols <- c(purple, blue)
  condition_pch <- c(16, 17)
  panels <- list(
    list(title = "Linear retrieval", family = "linear"),
    list(title = "MLP retrieval", family = "mlp"),
    list(title = "Flow-mean retrieval", family = "flow")
  )
  for (i in seq_along(panels)) {
    p <- panels[[i]]
    xx <- log10(cross_sizes)
    par(mar = c(2.55, 2.80, 1.60, 0.40))
    plot(NA, xlim = cross_xlim, ylim = c(0, 1), axes = FALSE,
         xlab = "", ylab = "")
    abline(h = seq(0, 1, 0.2), col = "#EEEEEE", lwd = 0.7)
    axis_plain(1, at = log10(cross_xticks),
               labels = c("100", "1k", "10k", "100k"))
    if (i == 1) axis_plain(2, at = seq(0, 1, 0.2))
    box(col = ink, lwd = 0.7)
    if (i == 1) {
      mtext("Top-1 prompt retrieval", side = 2, line = 1.90,
            cex = 0.90, las = 0)
    }
    if (i == 2) {
      mtext("LMSYS training prompts", side = 1, line = 1.45, cex = 0.88)
    }
    for (j in seq_along(conditions)) {
      means <- sds <- numeric(length(cross_sizes))
      for (k in seq_along(cross_sizes)) {
        rows <- metrics[metrics$family == p$family &
                          metrics$condition_hidden == conditions[j] &
                          metrics$n_train_contexts == cross_sizes[k], ]
        values <- rows$retrieval_top1_accuracy
        stopifnot(length(values) > 0, all(is.finite(values)))
        means[k] <- mean(values)
        sds[k] <- if (length(values) > 1) sd(values) else 0
      }
      lines(xx, means, col = condition_cols[j], lty = 1, lwd = 1.7)
      points(xx, means, col = condition_cols[j], bg = condition_cols[j],
             pch = condition_pch[j], cex = 0.66, lwd = 1.0)
      segments(xx, means - sds, xx, means + sds,
               col = condition_cols[j], lwd = 0.7)
    }
    title(p$title, line = 0.35, cex.main = 0.90, font.main = 1)
  }
  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend("center", condition_labels,
         col = c(purple, blue), lty = c(1, 1),
         lwd = c(1.7, 1.7), pch = c(16, 17),
         bty = "n", cex = 0.82, horiz = TRUE, x.intersp = 0.55,
         text.width = strwidth(paste0(condition_labels, "    "), cex = 0.82))
}
save_plot("supp_cross_model_retrieval_scaling", 7.05, 2.65,
          draw_cross_model_retrieval)

# ---------------------------------------------------------------------------
# Supplement: cross-model Gaussian and flow distribution scaling.
# ---------------------------------------------------------------------------
draw_cross_model_distribution_scaling <- function(metrics = cross_model,
    auxiliary = cross_aux, conditions = c(3584, 4096),
    condition_labels = c("Qwen2.5 -> Qwen3.5", "Qwen3.5 -> Qwen3.5"),
    primary_axes = NULL, auxiliary_axes = NULL,
    axis_margin = 3.15, axis_label_line = 2.00) {
  cross_sizes <- sort(unique(metrics$n_train_contexts))
  old <- par(no.readonly = TRUE); on.exit(par(old))
  layout(matrix(c(1:12, 13, 13, 13), nrow = 5, byrow = TRUE),
         heights = c(1, 1, 1, 1, 0.17), widths = rep(1, 3))
  par(oma = c(0, 0.35, 0.30, 0), mgp = c(1.62, 0.40, 0), las = 1,
      family = "sans", cex = 0.76)
  condition_cols <- c(purple, blue)
  condition_pch <- c(16, 17)
  panels <- list(
    list(title = "Gaussian: energy score", family = "gaussian",
         field = "distribution_energy_per_sqrt_dimension", ylim = c(0.06, 0.15),
         yticks = c(0.06, 0.09, 0.12, 0.15),
         ylab = expression("Energy score" / sqrt(d))),
    list(title = "Gaussian: variance ranking", family = "gaussian",
         field = "uncertainty_spearman", ylim = c(-0.1, 0.9),
         yticks = c(-0.1, 0.2, 0.5, 0.8), ylab = "Variance Spearman"),
    list(title = "Gaussian: variance ratio", family = "gaussian",
         field = "generated_real_variance_trace_ratio", ylim = c(0, 8.2),
         yticks = c(0, 2, 4, 6, 8), ylab = "Generated / observed"),
    list(title = "Flow: energy score", family = "flow",
         field = "distribution_energy_per_sqrt_dimension", ylim = c(0.06, 0.15),
         yticks = c(0.06, 0.09, 0.12, 0.15),
         ylab = expression("Energy score" / sqrt(d))),
    list(title = "Flow: variance ranking", family = "flow",
         field = "uncertainty_spearman", ylim = c(-0.1, 0.9),
         yticks = c(-0.1, 0.2, 0.5, 0.8), ylab = "Variance Spearman"),
    list(title = "Flow: variance ratio", family = "flow",
         field = "generated_real_variance_trace_ratio", ylim = c(0, 8.2),
         yticks = c(0, 2, 4, 6, 8), ylab = "Generated / observed")
  )
  for (i in seq_along(panels)) {
    p <- panels[[i]]
    if (!is.null(primary_axes[[p$field]])) {
      p$ylim <- primary_axes[[p$field]]$ylim
      p$yticks <- primary_axes[[p$field]]$yticks
    }
    xx <- log10(cross_sizes)
    par(mar = c(0.35, axis_margin, 1.65, 0.40))
    plot(NA, xlim = cross_xlim, ylim = p$ylim, axes = FALSE,
         xlab = "", ylab = "")
    abline(h = p$yticks, col = "#EEEEEE", lwd = 0.7)
    if (p$field == "generated_real_variance_trace_ratio") {
      abline(h = 1, col = gray, lty = 3, lwd = 0.8)
    }
    axis_plain(2, at = p$yticks)
    box(col = ink, lwd = 0.7)
    mtext(p$ylab, side = 2, line = axis_label_line, cex = 0.84, las = 0)
    for (j in seq_along(conditions)) {
      means <- sds <- numeric(length(cross_sizes))
      for (k in seq_along(cross_sizes)) {
        rows <- metrics[metrics$family == p$family &
                          metrics$condition_hidden == conditions[j] &
                          metrics$n_train_contexts == cross_sizes[k], ]
        values <- rows[[p$field]]
        stopifnot(length(values) > 1, all(is.finite(values)))
        means[k] <- mean(values)
        sds[k] <- sd(values)
      }
      stopifnot(all(means - sds >= p$ylim[1]), all(means + sds <= p$ylim[2]))
      lines(xx, means, col = condition_cols[j], lty = 1, lwd = 1.7)
      points(xx, means, col = condition_cols[j], bg = condition_cols[j],
             pch = condition_pch[j], cex = 0.66, lwd = 1.0)
      segments(xx, means - sds, xx, means + sds,
               col = condition_cols[j], lwd = 0.7)
    }
    title(p$title, line = 0.40, cex.main = 0.84, font.main = 1)
  }
  for (family in c("gaussian", "flow")) {
    family_label <- if (family == "gaussian") "Gaussian" else "Flow"
    for (i in seq_along(aux_panels)) {
      p <- aux_panels[[i]]
      metric_axis <- aux_metric_axes[[i]]
      # Include the low-data Gaussian NLL and its full one-SD error bar.
      if (family == "gaussian" && p$field == "nll_raw") {
        metric_axis <- list(ylim = c(-4, 17), yticks = seq(-4, 16, 4))
      }
      if (!is.null(auxiliary_axes[[family]][[p$field]])) {
        metric_axis <- auxiliary_axes[[family]][[p$field]]
      }
      bottom_margin <- if (family == "flow") 2.65 else 0.35
      par(mar = c(bottom_margin, axis_margin, 1.65, 0.40))
      plot(NA, xlim = cross_xlim, ylim = metric_axis$ylim,
           axes = FALSE, xlab = "", ylab = "")
      abline(h = metric_axis$yticks, col = "#EEEEEE", lwd = 0.7)
      if (family == "flow") {
        axis_plain(1, at = log10(cross_xticks),
                   labels = c("100", "1k", "10k", "100k"))
      }
      axis_plain(2, at = metric_axis$yticks)
      box(col = ink, lwd = 0.7)
      if (family == "flow" && i == 2) {
        mtext("LMSYS training prompts", side = 1, line = 1.55, cex = 0.88)
      }
      mtext(p$ylab, side = 2, line = axis_label_line, cex = 0.84, las = 0)
      conditioners <- condition_labels
      for (j in seq_along(conditioners)) {
        d <- auxiliary[auxiliary$family == family &
                         auxiliary$conditioner == conditioners[j], ]
        sizes <- sort(unique(d$n_train))
        stopifnot(identical(sizes, cross_sizes))
        means <- sds <- numeric(length(sizes))
        for (k in seq_along(sizes)) {
          values <- d[d$n_train == sizes[k], p$field]
          stopifnot(length(values) > 1, all(is.finite(values)))
          means[k] <- mean(values)
          sds[k] <- sd(values)
        }
        stopifnot(all(means - sds >= metric_axis$ylim[1]),
                  all(means + sds <= metric_axis$ylim[2]))
        xx <- log10(sizes)
        lines(xx, means, col = condition_cols[j], lwd = 1.7)
        points(xx, means, col = condition_cols[j], bg = condition_cols[j],
               pch = condition_pch[j], cex = 0.66, lwd = 1.0)
        segments(xx, means - sds, xx, means + sds,
                 col = condition_cols[j], lwd = 0.7)
      }
      panel_title <- aux_panel_title(family_label, p)
      title(panel_title, line = 0.40, cex.main = 0.84, font.main = 1)
    }
  }
  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend("center", condition_labels,
         col = c(purple, blue), lty = c(1, 1), lwd = c(1.7, 1.7),
         pch = c(16, 17), bty = "n", cex = 0.82, horiz = TRUE,
         x.intersp = 0.60,
         text.width = strwidth(paste0(condition_labels, "    "), cex = 0.82))
}
save_plot("supp_cross_model_distribution_scaling", 7.05, 8.25,
          draw_cross_model_distribution_scaling)

# Matching Gemma-target diagnostics use the same panel layout and styles.
# Target-specific energy/NLL scales and full error bars determine the limits.
gemma_metrics <- read.csv(file.path(input_dir, "unique_cross_gemma.csv"))
gemma_auxiliary <- read.csv(file.path(input_dir, "unique_cross_gemma_auxiliary.csv"))
gemma_conditions <- c(3584, 3840)
gemma_condition_labels <- c("Qwen2.5 -> Gemma 4", "Gemma 4 -> Gemma 4")
save_plot("supp_gemma_cross_model_retrieval_scaling", 7.05, 2.65,
  function() draw_cross_model_retrieval(gemma_metrics, gemma_conditions,
                                      gemma_condition_labels))
gemma_primary_axes <- list(
  distribution_energy_per_sqrt_dimension = list(
    ylim = c(0.009, 0.026), yticks = seq(0.01, 0.025, 0.005)),
  generated_real_variance_trace_ratio = list(
    ylim = c(0, 23), yticks = seq(0, 20, 5))
)
gemma_auxiliary_axes <- list(
  gaussian = list(nll_raw = list(ylim = c(-120, 410), yticks = seq(-100, 400, 100))),
  flow = list(nll_raw = list(ylim = c(-4, -2.5), yticks = seq(-4, -2.5, 0.5)))
)
save_plot("supp_gemma_cross_model_distribution_scaling", 7.05, 8.25,
  function() draw_cross_model_distribution_scaling(gemma_metrics, gemma_auxiliary,
    gemma_conditions, gemma_condition_labels, gemma_primary_axes, gemma_auxiliary_axes,
    axis_margin = 3.80, axis_label_line = 2.65))


# ---------------------------------------------------------------------------
# Supplement: context representations, with three domains side by side.
# All three families share axes and legend terminology, including the full
# observed range from 100 prompts and negative low-data R-squared values.
# ---------------------------------------------------------------------------
draw_context_domains <- function(family, specs) {
  ylim <- c(-0.5, 1.0)
  yticks <- seq(-0.4, 1.0, 0.2)
  old <- par(no.readonly = TRUE); on.exit(par(old))
  layout(matrix(c(1, 2, 3, 4, 4, 4), nrow = 2, byrow = TRUE),
         heights = c(1, 0.19), widths = c(1.25, 1, 1))
  par(mgp = c(1.95, 0.50, 0), las = 1, family = "sans", cex = 0.80)
  domains <- c("lmsys", "weirdchat", "ifeval")
  titles <- c("Held-out LMSYS", "Zero-shot WeirdChat", "Zero-shot IFEval")
  for (i in seq_along(domains)) {
    if (family == "linear") {
      d <- data.frame(n_train = sort(unique(linear_representations$n_train)))
      for (spec in specs) {
        rows <- linear_representations[
          linear_representations$representation == spec$field, ]
        stopifnot(nrow(rows) == nrow(d), !anyDuplicated(rows$n_train))
        d[[spec$field]] <- rows[match(d$n_train, rows$n_train), paste0(domains[i], "_r2")]
      }
    } else {
      d <- read.csv(file.path(input_dir,
        paste0("unique_", family, "_representations_", domains[i], ".csv")))
    }
    d <- d[order(d$n_train), ]
    stopifnot(nrow(d) == 15, min(d$n_train) == 100, max(d$n_train) == 500000)
    par(mar = c(3.05, if (i == 1) 3.85 else 0.55, 1.50, 0.35))
    plot(NA, xlim = full_prompt_xlim, ylim = ylim, axes = FALSE, xlab = "", ylab = "")
    abline(h = yticks, col = "#EEEEEE", lwd = 0.7)
    axis_plain(1, at = log10(full_prompt_ticks), labels = full_prompt_labels,
               cex.axis = 0.78, gap.axis = 0.25)
    if (i == 1) axis_plain(2, at = yticks)
    box(col = ink, lwd = 0.7)
    if (i == 1) mtext(expression("Mean " * R^2), side = 2, line = 2.15, cex = 0.90, las = 0)
    if (i == 2) mtext("LMSYS training prompts", side = 1, line = 1.65, cex = 0.86)
    for (spec in specs) {
      stopifnot(all(is.finite(d[[spec$field]])),
                all(d[[spec$field]] >= ylim[1]), all(d[[spec$field]] <= ylim[2]))
      lines(log10(d$n_train), d[[spec$field]], col = spec$col,
            lty = spec$lty, lwd = spec$lwd)
      points(log10(d$n_train), d[[spec$field]], col = spec$col,
             pch = spec$pch, cex = spec$cex, lwd = spec$point_lwd)
    }
    title(titles[i], line = 0.28, cex.main = 0.90)
  }
  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend_labels <- vapply(specs, `[[`, character(1), "label")
  legend("center", legend_labels,
         col = vapply(specs, `[[`, character(1), "col"),
         lty = vapply(specs, `[[`, numeric(1), "lty"),
         pch = vapply(specs, `[[`, numeric(1), "pch"),
         lwd = vapply(specs, `[[`, numeric(1), "lwd"),
         bty = "n", cex = 0.88, horiz = TRUE, x.intersp = 0.65,
         text.width = strwidth(paste0(legend_labels, "    "), cex = 0.88))
}

draw_representations <- function() {
  specs <- list(
    list(field = "binned_32_mlp", label = "32 bins", col = blue, lty = 1, pch = 16),
    list(field = "exact_token_mlp", label = "Every token", col = orange, lty = 1, pch = 17),
    list(field = "last_token_mlp", label = "Last token", col = gray, lty = 2, pch = 1),
    list(field = "mean_token_mlp", label = "Mean token", col = purple, lty = 2, pch = 2)
  )
  specs <- lapply(specs, function(s) { s$lwd <- 2; s$cex <- 0.82; s$point_lwd <- 1.1; s })
  draw_context_domains("mlp", specs)
}
save_plot("supp_context_representation_scaling", 7.05, 3.45, draw_representations)

# ---------------------------------------------------------------------------
# Supplement: linear point-map representation scaling, including zero-shot OOD.
# ---------------------------------------------------------------------------
linear_representations <- read.csv(file.path(
  input_dir, "unique_linear_representations.csv"))

draw_linear_representations <- function() {
  specs <- list(
    list(field = "binned", label = "32 bins",
         col = blue, lty = 1, pch = 16),
    list(field = "last", label = "Last token",
         col = gray, lty = 2, pch = 1),
    list(field = "mean", label = "Mean token",
         col = purple, lty = 2, pch = 2)
  )
  specs <- lapply(specs, function(s) { s$lwd <- 1.9; s$cex <- 0.72; s$point_lwd <- 1; s })
  draw_context_domains("linear", specs)
}
save_plot("supp_linear_point_representation_scaling", 7.05, 3.45,
          draw_linear_representations)

# ---------------------------------------------------------------------------
# Supplement 1: number of training rollouts at fixed 100,000 prompts.
# ---------------------------------------------------------------------------
rollout_path <- file.path(input_dir, "unique_rollout_count.csv")
rollout <- read.csv(rollout_path)
rollout <- rollout[rollout$n_train_contexts == 100000 & rollout$split == "test" &
                     rollout$family %in% c("mlp", "flow"), ]

draw_rollouts <- function() {
  old <- par(no.readonly = TRUE); on.exit(par(old))
  par(mar = c(3.6, 3.8, 1.6, 0.7), mgp = c(2.2, 0.6, 0), las = 1,
      family = "sans", cex = 0.90)
  plot(NA, xlim = c(1.5, 16.5), ylim = c(0.825, 0.875), axes = FALSE,
       xlab = "Independent training rollouts per prompt",
       ylab = expression("Held-out mean " * R^2))
  abline(h = seq(0.83, 0.87, 0.01), col = "#EEEEEE", lwd = 0.7)
  abline(v = 8, col = "#AAAAAA", lty = 3)
  axis_plain(1, at = c(2, 4, 8, 12, 16))
  axis_plain(2, at = seq(0.83, 0.87, 0.01))
  box(col = ink, lwd = 0.7)
  title("Training-rollout scaling at 100,000 prompts",
        line = 0.30, cex.main = 0.94)
  for (fam in c("mlp", "flow")) {
    d <- rollout[rollout$family == fam, ]
    d <- d[order(d$n_train_rollout_seeds), ]
    cc <- if (fam == "mlp") blue else orange
    pp <- if (fam == "mlp") 16 else 17
    segments(d$n_train_rollout_seeds,
             d$sample_mean_r2_mean - d$sample_mean_r2_std,
             d$n_train_rollout_seeds,
             d$sample_mean_r2_mean + d$sample_mean_r2_std,
             col = cc, lwd = 1.1)
    lines(d$n_train_rollout_seeds, d$sample_mean_r2_mean, col = cc, lwd = 2)
    points(d$n_train_rollout_seeds, d$sample_mean_r2_mean,
           col = cc, pch = pp, cex = 0.9)
  }
  legend("bottomright", c("MLP mean", "Flow sample mean"),
         col = c(blue, orange), lwd = 2, pch = c(16, 17), bty = "n", cex = 0.84)
}
save_plot("supp_rollout_count", 5.7, 3.25, draw_rollouts)

# ---------------------------------------------------------------------------
# Supplement 2: flow conditioner architecture scaling.
# ---------------------------------------------------------------------------
draw_flow_arch <- function() {
  specs <- list(
    list(field = "binned_32_sample_mean_r2", label = "32 bins",
         col = blue, lty = 1, pch = 16, lwd = 2.4),
    list(field = "token_query_sample_mean_r2", label = "Every token",
         col = orange, lty = 1, pch = 17, lwd = 1.6),
    list(field = "last_token_sample_mean_r2", label = "Last token",
         col = gray, lty = 2, pch = 1, lwd = 1.6),
    list(field = "mean_token_sample_mean_r2", label = "Mean token",
         col = purple, lty = 2, pch = 2, lwd = 1.6)
  )
  specs <- lapply(specs, function(s) { s$cex <- 0.68; s$point_lwd <- 0.9; s })
  draw_context_domains("flow", specs)
}
save_plot("supp_flow_context_architectures", 7.05, 3.45, draw_flow_arch)

# ---------------------------------------------------------------------------
# Main figure: safety-related WeirdChat behavior forecasts.
# ---------------------------------------------------------------------------
safety_path <- file.path(input_dir, "unique_probes.csv")
safety <- read.csv(safety_path)

draw_safety <- function(metric = "auprc") {
  old <- par(no.readonly = TRUE); on.exit(par(old))
  par(mar = c(4.8, 3.8, 3.8, 0.6), mgp = c(2.2, 0.6, 0), las = 1,
      family = "sans", cex = 0.80)
  x <- 1:4
  labels <- c("Any target\nbehavior", "Any safety\nfailure",
              "Self-harm /\neating disorder", "Deception /\nfalse capability")
  methods <- c("Direct context" = "primary_direct",
    "Linear mean" = "primary_linear", "MLP mean" = "primary_mean",
    "Gaussian integration" = "primary_gaussian",
    "Flow integration" = "primary_flow", "Actual answer" = "primary_realized")
  vals <- lapply(methods, function(method) safety[[paste0(method, "_", metric)]])
  cols <- c("Direct context" = purple, "Linear mean" = linear_col,
            "MLP mean" = mlp_col, "Gaussian integration" = gaussian_col,
            "Flow integration" = flow_col, "Actual answer" = ink)
  values <- do.call(rbind, vals)
  y_max <- if (metric == "auprc") 0.65 else 1.0
  ticks <- if (metric == "auprc") seq(0, 0.6, 0.1) else seq(0, 1, 0.2)
  baseline <- if (metric == "auprc") safety$test_prevalence else NULL
  stopifnot(all(is.finite(values)), all(values >= 0), all(values <= y_max))
  positions <- barplot(values, beside = TRUE, plot = FALSE,
                       space = c(0.10, 0.55))
  plot(NA, xlim = c(min(positions) - 0.6, max(positions) + 0.6),
       ylim = c(0, y_max), axes = FALSE,
       xlab = "Luna-judged WeirdChat event",
       ylab = if (metric == "auprc") "Area under precision-recall curve" else
         "Area under ROC curve")
  abline(h = ticks, col = "#EEEEEE")
  axis_plain(2, at = ticks)
  barplot(values, beside = TRUE, add = TRUE, axes = FALSE,
          names.arg = rep("", length(labels)), col = unname(cols),
          border = NA, space = c(0.10, 0.55))
  axis_plain(1, at = colMeans(positions), labels = labels,
             tick = FALSE, line = 0.4)
  for (j in seq_along(baseline)) {
    segments(min(positions[, j]) - 0.3, baseline[j],
             max(positions[, j]) + 0.3, baseline[j],
             col = gray, lty = 2, lwd = 1.2)
  }
  box(col = ink, lwd = 0.7)
  title("Held-out WeirdChat behavior prediction", line = 2.55,
        cex.main = 0.84)
  legend_labels <- names(vals)
  legend_fill <- unname(cols)
  legend_lty <- rep(0, length(vals))
  legend_col <- rep(NA, length(vals))
  if (!is.null(baseline)) {
    legend_labels <- c(legend_labels, "Base rate")
    legend_fill <- c(legend_fill, NA)
    legend_lty <- c(legend_lty, 2)
    legend_col <- c(legend_col, gray)
  }
  legend(mean(par("usr")[1:2]), par("usr")[4] + 0.19 * diff(par("usr")[3:4]),
         legend_labels, fill = legend_fill, border = NA, lty = legend_lty,
         col = legend_col, bty = "n", cex = 0.70,
         ncol = if (metric == "auprc") 4 else 3,
         y.intersp = 0.9, xjust = 0.5, yjust = 1, xpd = NA)
}
save_plot("main_probe_results", 7.05, 3.2, function() draw_safety("auprc"))
save_plot("supp_probe_auroc", 7.05, 3.2, function() draw_safety("auroc"))

message("Wrote paper figures to: ", fig_dir)
