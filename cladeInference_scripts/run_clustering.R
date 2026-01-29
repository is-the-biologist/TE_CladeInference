#!/usr/bin/env Rscript
# run_clustering.R
# CLI wrapper for SubFamily_inference.R hierarchical clustering step.
# Called by run_pipeline.py or can be invoked standalone.
#
# Usage:
#   Rscript run_clustering.R <te_list> <sample_sheet> <cn_tables_dir> <outlier_dir> <cn_dir> <output_dir> <linkage> <color_pal_csv> <cutoffs_csv> <outlier_bool_csv>

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 10) {
  stop("Usage: Rscript run_clustering.R <te_list> <sample_sheet> <cn_tables_dir> <outlier_dir> <cn_dir> <output_dir> <linkage> <color_pal_csv> <cutoffs_csv> <outlier_bool_csv>")
}

te_list_file    <- args[1]
sample_sheet    <- args[2]
cn_tables_dir   <- args[3]
outlier_dir     <- args[4]
cn_dir          <- args[5]
output_dir      <- args[6]
linkage_method  <- args[7]
color_pal_csv   <- args[8]
cutoffs_csv     <- args[9]
outlier_bool_csv <- args[10]

# Source the core functions — determine script directory robustly
get_script_dir <- function() {
  # Works when called via Rscript
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  # Fallback: working directory
  return(getwd())
}
script_dir <- get_script_dir()
source(file.path(script_dir, "SubFamily_inference.R"))

# Parse inputs
color_pal <- strsplit(color_pal_csv, ",")[[1]]
cutoffs <- as.numeric(strsplit(cutoffs_csv, ",")[[1]])
outlier_bools <- as.logical(strsplit(outlier_bool_csv, ",")[[1]])

# Read TE list
te_df <- read.csv(te_list_file, header = FALSE)
te_names <- te_df$V1

if (length(cutoffs) == 1) {
  cutoffs <- rep(cutoffs, length(te_names))
}

if (length(outlier_bools) == 1) {
  outlier_bools <- rep(outlier_bools, length(te_names))
}

stopifnot(length(cutoffs) == length(te_names))
stopifnot(length(outlier_bools) == length(te_names))

# Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Run clustering for each TE
for (t in seq_along(te_names)) {
  te <- te_names[t]
  input_path <- file.path(cn_tables_dir, te)
  out_mask <- outlier_bools[t]

  cat(sprintf("[R Clustering] Processing %s (cutoff=%.4f, outlier=%s)\n", te, cutoffs[t], out_mask))

  extractHaplotypes(
    TE_name       = input_path,
    outlier       = out_mask,
    outlier_dir   = outlier_dir,
    output_dir    = output_dir,
    linkage       = linkage_method,
    dist_cutoff   = cutoffs[t],
    color_pal     = color_pal,
    alleleCN_dir  = cn_dir,
    sample        = sample_sheet
  )
}

cat("[R Clustering] Done.\n")
