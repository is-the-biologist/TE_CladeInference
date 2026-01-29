#!/usr/bin/env python3
"""
run_pipeline.py - Unified CLI for the TE Clade Inference pipeline.

Consolidates the two-step workflow (Python preprocessing + R clustering)
into a single command-line invocation.

Example usage with test data:
    python run_pipeline.py \
        --te-list TEST_DATA/ACTIVE_TES_full.tsv \
        --sample-sheet TEST_DATA/GDL_sample_sheet.csv \
        --cn-dir TEST_DATA/GDL_RUN-11-15-19 \
        --output-dir TEST_DATA/CLI_OUTPUT \
        --dist-cutoff 0.4
"""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="TE Clade Inference Pipeline - end-to-end CLI"
    )

    # Required paths
    parser.add_argument("--te-list", required=True,
                        help="Path to TE list TSV file (one TE name per line, no header)")
    parser.add_argument("--sample-sheet", required=True,
                        help="Path to sample sheet CSV (columns: population, strain_ID)")
    parser.add_argument("--cn-dir", required=True,
                        help="Path to directory containing allele CN numpy files (*_CN.npy)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for clustering results")

    # QC / filtering parameters
    parser.add_argument("--pi-filter", type=float, default=0.1,
                        help="Minimum positional sequence diversity (default: 0.1)")
    parser.add_argument("--cn-filter", type=float, default=0.5,
                        help="Minimum allele copy number threshold (default: 0.5)")
    parser.add_argument("--min-freq", type=float, default=0.1,
                        help="Minimum allele population frequency (default: 0.1)")
    parser.add_argument("--min-strains", type=int, default=10,
                        help="Minimum number of strains with allele (default: 10)")

    # Clustering parameters
    parser.add_argument("--dist-cutoff", type=float, default=0.5,
                        help="Global correlation distance cutoff (1-r) for dendrogram cutting (default: 0.5)")
    parser.add_argument("--dist-cutoffs-file", default=None,
                        help="Path to TSV file with per-TE cutoffs (columns: TE_name, cutoff). "
                             "Overrides --dist-cutoff for TEs listed in the file.")
    parser.add_argument("--linkage", default="average",
                        help="Hierarchical clustering linkage method (default: average)")
    parser.add_argument("--color-pal", default="#6F0000,#009191,#6DB6FF,orange,#490092",
                        help="Comma-separated color palette for population annotations")

    # Outlier removal
    parser.add_argument("--outliers-json", default=None,
                        help='Path to JSON file mapping TE names to outlier strain IDs, '
                             'e.g. {"ZAM_FULL": ["I23"], "BEL_FULL": ["I24","N13"]}')

    # Optional paths (defaults derived from cn-dir)
    parser.add_argument("--pi-dir", default=None,
                        help="Directory for Pi numpy files (default: <cn-dir>/../Pi)")
    parser.add_argument("--cn-tables-dir", default=None,
                        help="Directory for CN table CSVs (default: <cn-dir>/../CN_tables/FULL)")
    parser.add_argument("--ap-tables-dir", default=None,
                        help="Directory for AP table CSVs (default: <cn-dir>/../AP_tables)")
    parser.add_argument("--outlier-tables-dir", default=None,
                        help="Directory for outlier-removed CN tables (default: <cn-dir>/../CN_tables/NO_OUTLIERS)")

    # Control flags
    parser.add_argument("--skip-python", action="store_true",
                        help="Skip the Python preprocessing step (use existing CN/AP tables)")
    parser.add_argument("--skip-r", action="store_true",
                        help="Skip the R clustering step")

    return parser.parse_args()


def resolve_paths(args):
    """Derive default paths from cn-dir if not explicitly provided."""
    parent = os.path.dirname(os.path.abspath(args.cn_dir))

    if args.pi_dir is None:
        args.pi_dir = os.path.join(parent, "Pi")
    if args.cn_tables_dir is None:
        args.cn_tables_dir = os.path.join(parent, "CN_tables", "FULL")
    if args.ap_tables_dir is None:
        args.ap_tables_dir = os.path.join(parent, "AP_tables")
    if args.outlier_tables_dir is None:
        args.outlier_tables_dir = os.path.join(parent, "CN_tables", "NO_OUTLIERS")

    # Ensure output directories exist
    for d in [args.cn_tables_dir, args.ap_tables_dir, args.outlier_tables_dir, args.pi_dir]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def run_python_step(args, te_names):
    """Step 1: Calculate Pi and extract allele copy numbers."""
    from HaploTE import summary_stats, subfamilyInference

    print("=" * 60)
    print("STEP 1: Python preprocessing (Pi calculation + allele CN extraction)")
    print("=" * 60)

    stats = summary_stats(
        sample_sheet=args.sample_sheet,
        cn_dir=args.cn_dir,
        pi_dir=args.pi_dir
    )

    clade_inf = subfamilyInference(
        TE_list=args.te_list,
        sample_sheet=args.sample_sheet,
        cn_dir=args.cn_dir,
        pi_dir=args.pi_dir,
        cn_df_path=args.cn_tables_dir,
        ap_df_path=args.ap_tables_dir
    )

    # Calculate Pi for each TE
    for te in te_names:
        print(f"  [Pi] {te}")
        stats.calcPi(TE=te)

    # Extract allele copy numbers
    for te in te_names:
        print(f"  [CNAP] {te} (pi_filter={args.pi_filter}, CN_filter={args.cn_filter}, "
              f"minFreq={args.min_freq}, min_strains={args.min_strains})")
        clade_inf.CNAP_extraction(
            TE=te,
            pi_filter=args.pi_filter,
            CN_filter=args.cn_filter,
            minFreq=args.min_freq,
            min_strains=args.min_strains
        )

    # Handle outlier removal if specified
    if args.outliers_json:
        print("\n  [Outliers] Removing outlier strains...")
        with open(args.outliers_json) as f:
            outliers = json.load(f)

        for te, strains in outliers.items():
            print(f"    {te}: removing {strains}")
            clade_inf.drop_outliers(TE=te, strain_masks=strains)

    print("\nPython preprocessing complete.\n")
    return clade_inf


def build_cutoffs(args, te_names):
    """Build per-TE cutoff list, merging global default with per-TE file."""
    cutoffs = [args.dist_cutoff] * len(te_names)

    if args.dist_cutoffs_file:
        cutoff_df = pd.read_csv(args.dist_cutoffs_file, sep="\t", header=None, names=["TE_name", "cutoff"])
        cutoff_map = dict(zip(cutoff_df["TE_name"], cutoff_df["cutoff"]))
        for i, te in enumerate(te_names):
            if te in cutoff_map:
                cutoffs[i] = cutoff_map[te]

    return cutoffs


def build_outlier_bools(args, te_names):
    """Build per-TE boolean list indicating whether to use outlier-removed tables."""
    bools = [False] * len(te_names)

    if args.outliers_json:
        with open(args.outliers_json) as f:
            outliers = json.load(f)
        for i, te in enumerate(te_names):
            if te in outliers:
                bools[i] = True

    return bools


def run_r_step(args, te_names, cutoffs, outlier_bools):
    """Step 2: Run R hierarchical clustering via subprocess."""
    print("=" * 60)
    print("STEP 2: R hierarchical clustering")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    r_script = os.path.join(script_dir, "run_clustering.R")

    cutoffs_csv = ",".join(str(c) for c in cutoffs)
    outlier_bool_csv = ",".join(str(b).upper() for b in outlier_bools)

    # Locate Rscript from the same prefix as the running Python (handles conda envs)
    python_bin_dir = os.path.dirname(sys.executable)
    rscript_path = os.path.join(python_bin_dir, "Rscript")
    if not os.path.isfile(rscript_path):
        rscript_path = "Rscript"

    cmd = [
        rscript_path, r_script,
        args.te_list,
        args.sample_sheet,
        args.cn_tables_dir,
        args.outlier_tables_dir,
        args.cn_dir,
        args.output_dir,
        args.linkage,
        args.color_pal,
        cutoffs_csv,
        outlier_bool_csv
    ]

    # Ensure R's reticulate uses the same Python as the pipeline
    env = os.environ.copy()
    env["RETICULATE_PYTHON"] = sys.executable

    print(f"  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False, env=env)

    if result.returncode != 0:
        print(f"\nR clustering exited with return code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print("\nR clustering complete.\n")


def main():
    args = parse_args()
    args = resolve_paths(args)

    # Read TE list
    te_df = pd.read_csv(args.te_list, header=None)
    te_names = list(te_df.iloc[:, 0])

    print(f"Pipeline configuration:")
    print(f"  TEs: {len(te_names)}")
    print(f"  Sample sheet: {args.sample_sheet}")
    print(f"  CN directory: {args.cn_dir}")
    print(f"  Output: {args.output_dir}")
    print()

    if not args.skip_python:
        run_python_step(args, te_names)
    else:
        print("Skipping Python preprocessing step (--skip-python).\n")

    if not args.skip_r:
        cutoffs = build_cutoffs(args, te_names)
        outlier_bools = build_outlier_bools(args, te_names)
        run_r_step(args, te_names, cutoffs, outlier_bools)
    else:
        print("Skipping R clustering step (--skip-r).\n")

    print("=" * 60)
    print("Pipeline complete.")
    print(f"Results written to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
