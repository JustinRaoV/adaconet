#!/usr/bin/env Rscript
# Generate independent benchmark data using SparseDOSSA2
# Different data-generating process from our v4 simulator:
#   v4: Multivariate Normal → exp → multinomial
#   SparseDOSSA2: Zero-inflated truncated log-normal + Gaussian copula
#
# Ground truth: empirical Spearman correlation of absolute abundances
# (Omega is a model parameter, not a standard correlation matrix)

.libPaths(c("/Users/justin/project/school/graduation/renv/library/macos/R-4.6/aarch64-apple-darwin23", .libPaths()))
library(SparseDOSSA2)

outdir <- "data/simulated/sparsedossa2"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# Use new_features=FALSE to keep template's real correlation structure
# Stool template has p=332 features from real HMP stool samples
configs <- list(
  list(n = 200, label = "Stool332_N200"),
  list(n = 500, label = "Stool332_N500")
)

for (cfg in configs) {
  cat(sprintf("\n=== Config: %s (n=%d) ===\n", cfg$label, cfg$n))
  
  set.seed(42)
  res <- SparseDOSSA2(
    template = "Stool",
    n_sample = cfg$n,
    new_features = FALSE,  # keep real stool taxa with correlations
    median_read_depth = 10000,
    verbose = FALSE
  )
  
  p <- nrow(res$simulated_data)
  n <- ncol(res$simulated_data)
  cat(sprintf("Data: %d samples x %d features\n", n, p))
  
  # Absolute abundances (ground truth data)
  a_spiked <- res$simulated_matrices$a_spiked  # p x n
  
  # Relative abundances (what methods see)
  rel <- res$simulated_matrices$rel  # p x n
  
  # Scale relative abundances to pseudo-counts
  counts <- round(rel * 100000)
  counts_t <- t(counts)  # n x p (samples x taxa)
  
  cat(sprintf("Zero fraction: %.1f%%\n", mean(counts_t == 0) * 100))
  
  # Pre-filter: remove taxa with zero total reads OR zero variance in log space
  col_sums <- colSums(counts_t)
  log_abs_all <- log2(a_spiked + 1)
  col_sd <- apply(log_abs_all, 1, sd)  # per taxon (row of a_spiked)
  keep <- col_sums > 0 & !is.na(col_sd) & col_sd > 0
  counts_t <- counts_t[, keep]
  a_spiked_filt <- a_spiked[keep, ]
  
  # Ground truth: Spearman correlation of log absolute abundances (AFTER filtering)
  log_abs_filt <- log2(a_spiked_filt + 1)
  truth_corr <- cor(t(log_abs_filt), method = "spearman")
  
  # Count edges
  iu <- upper.tri(truth_corr)
  n_edges_01 <- sum(abs(truth_corr[iu]) > 0.1)
  n_edges_03 <- sum(abs(truth_corr[iu]) > 0.3)
  n_edges_05 <- sum(abs(truth_corr[iu]) > 0.5)
  cat(sprintf("Edges: |r|>0.1: %d (%.1f%%), |r|>0.3: %d (%.1f%%), |r|>0.5: %d (%.1f%%)\n",
              n_edges_01, n_edges_01/sum(iu)*100,
              n_edges_03, n_edges_03/sum(iu)*100,
              n_edges_05, n_edges_05/sum(iu)*100))
  p_kept <- sum(keep)
  cat(sprintf("After filtering: %d x %d\n", nrow(counts_t), p_kept))
  
  # Save
  taxa_names <- paste0("T", 1:p_kept)
  colnames(counts_t) <- taxa_names
  colnames(truth_corr) <- taxa_names
  rownames(truth_corr) <- taxa_names
  
  write.csv(counts_t, file.path(outdir, paste0(cfg$label, "_counts.csv")), row.names = FALSE)
  write.csv(truth_corr, file.path(outdir, paste0(cfg$label, "_truth.csv")))
  
  cat(sprintf("Saved: %s (p=%d, n=%d)\n", cfg$label, p_kept, n))
}

cat("\n=== All configs generated ===\n")
