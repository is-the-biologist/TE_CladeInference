#!/usr/bin/env python3
"""
Tests for the TE Clade Inference CLI pipeline.

Unit tests exercise argument parsing, path resolution, cutoff building,
and outlier bool logic without heavy bioinformatics dependencies.

Integration tests run the actual pipeline on the bundled TEST_DATA and
require all Python + R dependencies to be installed.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA = os.path.join(SCRIPT_DIR, "TEST_DATA")
TE_LIST = os.path.join(TEST_DATA, "ACTIVE_TES_full.tsv")
SAMPLE_SHEET = os.path.join(TEST_DATA, "GDL_sample_sheet.csv")
CN_DIR = os.path.join(TEST_DATA, "GDL_RUN-11-15-19")
CN_TABLES_DIR = os.path.join(TEST_DATA, "CN_tables", "FULL")
OUTLIER_TABLES_DIR = os.path.join(TEST_DATA, "CN_tables", "NO_OUTLIERS")
PI_DIR = os.path.join(TEST_DATA, "Pi")
AP_TABLES_DIR = os.path.join(TEST_DATA, "AP_tables")

# Import pipeline helpers (no heavy deps needed at import time)
import sys
sys.path.insert(0, SCRIPT_DIR)
from run_pipeline import parse_args, resolve_paths, build_cutoffs, build_outlier_bools

# Locate Rscript from the same prefix as the running Python interpreter
# (handles conda environments where Rscript lives alongside python)
_PYTHON_PREFIX = os.path.dirname(sys.executable)
_RSCRIPT = os.path.join(_PYTHON_PREFIX, "Rscript")
if not os.path.isfile(_RSCRIPT):
    import shutil
    _RSCRIPT = shutil.which("Rscript") or "Rscript"


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_output(tmp_path):
    """Provide a temporary output directory."""
    return str(tmp_path / "output")


@pytest.fixture
def minimal_args(tmp_output):
    """Return a minimal Namespace that mirrors required CLI args."""
    return argparse.Namespace(
        te_list=TE_LIST,
        sample_sheet=SAMPLE_SHEET,
        cn_dir=CN_DIR,
        output_dir=tmp_output,
        pi_filter=0.1,
        cn_filter=0.5,
        min_freq=0.1,
        min_strains=10,
        dist_cutoff=0.5,
        dist_cutoffs_file=None,
        linkage="average",
        color_pal="#6F0000,#009191,#6DB6FF,orange,#490092",
        outliers_json=None,
        pi_dir=None,
        cn_tables_dir=None,
        ap_tables_dir=None,
        outlier_tables_dir=None,
        skip_python=False,
        skip_r=False,
    )


@pytest.fixture
def te_names():
    """Load the TE names from the test TE list."""
    df = pd.read_csv(TE_LIST, header=None)
    return list(df.iloc[:, 0])


# ===========================================================================
# Unit Tests – argument parsing
# ===========================================================================

class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_required_args_only(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["run_pipeline.py",
             "--te-list", TE_LIST,
             "--sample-sheet", SAMPLE_SHEET,
             "--cn-dir", CN_DIR,
             "--output-dir", "/tmp/test_out"],
        )
        args = parse_args()
        assert args.te_list == TE_LIST
        assert args.sample_sheet == SAMPLE_SHEET
        assert args.cn_dir == CN_DIR
        assert args.output_dir == "/tmp/test_out"

    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["run_pipeline.py",
             "--te-list", TE_LIST,
             "--sample-sheet", SAMPLE_SHEET,
             "--cn-dir", CN_DIR,
             "--output-dir", "/tmp/test_out"],
        )
        args = parse_args()
        assert args.pi_filter == 0.1
        assert args.cn_filter == 0.5
        assert args.min_freq == 0.1
        assert args.min_strains == 10
        assert args.dist_cutoff == 0.5
        assert args.linkage == "average"
        assert args.dist_cutoffs_file is None
        assert args.outliers_json is None
        assert args.skip_python is False
        assert args.skip_r is False

    def test_custom_filter_args(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["run_pipeline.py",
             "--te-list", TE_LIST,
             "--sample-sheet", SAMPLE_SHEET,
             "--cn-dir", CN_DIR,
             "--output-dir", "/tmp/test_out",
             "--pi-filter", "0.2",
             "--cn-filter", "1.0",
             "--min-freq", "0.05",
             "--min-strains", "5"],
        )
        args = parse_args()
        assert args.pi_filter == 0.2
        assert args.cn_filter == 1.0
        assert args.min_freq == 0.05
        assert args.min_strains == 5

    def test_skip_flags(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["run_pipeline.py",
             "--te-list", TE_LIST,
             "--sample-sheet", SAMPLE_SHEET,
             "--cn-dir", CN_DIR,
             "--output-dir", "/tmp/test_out",
             "--skip-python", "--skip-r"],
        )
        args = parse_args()
        assert args.skip_python is True
        assert args.skip_r is True

    def test_missing_required_arg_exits(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["run_pipeline.py", "--te-list", TE_LIST],
        )
        with pytest.raises(SystemExit):
            parse_args()


# ===========================================================================
# Unit Tests – path resolution
# ===========================================================================

class TestResolvePaths:
    """Tests for resolve_paths deriving defaults from cn-dir."""

    def test_default_path_derivation(self, minimal_args):
        args = resolve_paths(minimal_args)
        parent = os.path.dirname(os.path.abspath(CN_DIR))
        assert args.pi_dir == os.path.join(parent, "Pi")
        assert args.cn_tables_dir == os.path.join(parent, "CN_tables", "FULL")
        assert args.ap_tables_dir == os.path.join(parent, "AP_tables")
        assert args.outlier_tables_dir == os.path.join(parent, "CN_tables", "NO_OUTLIERS")

    def test_explicit_paths_not_overridden(self, minimal_args, tmp_path):
        minimal_args.pi_dir = str(tmp_path / "custom_pi")
        minimal_args.cn_tables_dir = str(tmp_path / "custom_cn")
        minimal_args.ap_tables_dir = str(tmp_path / "custom_ap")
        minimal_args.outlier_tables_dir = str(tmp_path / "custom_outlier")
        args = resolve_paths(minimal_args)
        assert args.pi_dir == str(tmp_path / "custom_pi")
        assert args.cn_tables_dir == str(tmp_path / "custom_cn")
        assert args.ap_tables_dir == str(tmp_path / "custom_ap")
        assert args.outlier_tables_dir == str(tmp_path / "custom_outlier")

    def test_output_dir_created(self, minimal_args):
        args = resolve_paths(minimal_args)
        assert os.path.isdir(args.output_dir)


# ===========================================================================
# Unit Tests – cutoff building
# ===========================================================================

class TestBuildCutoffs:
    """Tests for build_cutoffs with global and per-TE file cutoffs."""

    def test_global_cutoff_only(self, minimal_args, te_names):
        minimal_args.dist_cutoff = 0.42
        cutoffs = build_cutoffs(minimal_args, te_names)
        assert len(cutoffs) == len(te_names)
        assert all(c == 0.42 for c in cutoffs)

    def test_per_te_file_overrides(self, minimal_args, te_names, tmp_path):
        cutoffs_file = tmp_path / "cutoffs.tsv"
        cutoffs_file.write_text("BARI_DM\t0.33\nDOC\t0.77\n")
        minimal_args.dist_cutoffs_file = str(cutoffs_file)
        minimal_args.dist_cutoff = 0.5

        cutoffs = build_cutoffs(minimal_args, te_names)

        bari_idx = te_names.index("BARI_DM")
        doc_idx = te_names.index("DOC")
        assert cutoffs[bari_idx] == 0.33
        assert cutoffs[doc_idx] == 0.77
        # Others still use global default
        for i, c in enumerate(cutoffs):
            if i not in (bari_idx, doc_idx):
                assert c == 0.5

    def test_file_with_unknown_te_ignored(self, minimal_args, te_names, tmp_path):
        cutoffs_file = tmp_path / "cutoffs.tsv"
        cutoffs_file.write_text("FAKE_TE\t0.99\n")
        minimal_args.dist_cutoffs_file = str(cutoffs_file)
        cutoffs = build_cutoffs(minimal_args, te_names)
        assert all(c == 0.5 for c in cutoffs)


# ===========================================================================
# Unit Tests – outlier bool building
# ===========================================================================

class TestBuildOutlierBools:
    """Tests for build_outlier_bools from JSON file."""

    def test_no_outliers_json(self, minimal_args, te_names):
        bools = build_outlier_bools(minimal_args, te_names)
        assert len(bools) == len(te_names)
        assert not any(bools)

    def test_outliers_json_sets_bools(self, minimal_args, te_names, tmp_path):
        outlier_file = tmp_path / "outliers.json"
        outlier_file.write_text(json.dumps({
            "ZAM_FULL": ["I23"],
            "BEL_FULL": ["I24", "N13"],
        }))
        minimal_args.outliers_json = str(outlier_file)

        bools = build_outlier_bools(minimal_args, te_names)
        zam_idx = te_names.index("ZAM_FULL")
        bel_idx = te_names.index("BEL_FULL")
        assert bools[zam_idx] is True
        assert bools[bel_idx] is True
        # Others are False
        for i, b in enumerate(bools):
            if i not in (zam_idx, bel_idx):
                assert b is False


# ===========================================================================
# Unit Tests – test data integrity
# ===========================================================================

class TestTestData:
    """Verify the bundled TEST_DATA is intact and consistent."""

    def test_te_list_exists(self):
        assert os.path.isfile(TE_LIST)

    def test_sample_sheet_exists(self):
        assert os.path.isfile(SAMPLE_SHEET)

    def test_te_list_has_41_entries(self, te_names):
        assert len(te_names) == 41

    def test_sample_sheet_has_85_strains(self):
        df = pd.read_csv(SAMPLE_SHEET, header=None)
        assert df.shape[0] == 85

    def test_sample_sheet_has_5_populations(self):
        df = pd.read_csv(SAMPLE_SHEET, header=None)
        pops = df[0].unique()
        assert len(pops) == 5

    def test_cn_numpy_files_exist(self, te_names):
        for te in te_names:
            path = os.path.join(CN_DIR, f"{te}_CN.npy")
            assert os.path.isfile(path), f"Missing CN numpy: {path}"

    def test_cn_numpy_shape(self):
        """Each CN numpy should have shape (85, 4, N) — strains x nucleotides x positions."""
        arr = np.load(os.path.join(CN_DIR, "BARI_DM_CN.npy"))
        assert arr.shape[0] == 85, f"Expected 85 strains, got {arr.shape[0]}"
        assert arr.shape[1] == 4, f"Expected 4 nucleotides, got {arr.shape[1]}"

    def test_cn_tables_match_te_list(self, te_names):
        existing = set(
            f.replace(".CN.GDL.minor.csv", "")
            for f in os.listdir(CN_TABLES_DIR)
            if f.endswith(".csv")
        )
        te_set = set(te_names)
        assert te_set == existing, f"Mismatch: in TE list but not tables: {te_set - existing}"

    def test_pi_files_exist_for_tes(self, te_names):
        for te in te_names:
            path = os.path.join(PI_DIR, f"{te}_pi.npy")
            assert os.path.isfile(path), f"Missing Pi file: {path}"

    def test_cn_table_has_85_rows(self):
        """CN tables should have 85 rows (one per strain)."""
        df = pd.read_csv(os.path.join(CN_TABLES_DIR, "BARI_DM.CN.GDL.minor.csv"))
        assert df.shape[0] == 85

    def test_cn_table_columns_are_allele_labels(self):
        """Column names should follow NT_position pattern (e.g. T_529)."""
        df = pd.read_csv(os.path.join(CN_TABLES_DIR, "BARI_DM.CN.GDL.minor.csv"))
        for col in df.columns:
            parts = col.split("_")
            assert len(parts) == 2, f"Unexpected column format: {col}"
            assert parts[0] in ("A", "T", "C", "G"), f"Unexpected NT: {parts[0]}"
            assert parts[1].isdigit(), f"Position not numeric: {parts[1]}"


# ===========================================================================
# Integration Tests – require full dependencies
# ===========================================================================

def _has_haploTE_deps():
    """Check whether HaploTE.py can be imported (needs scipy, seaborn, etc.)."""
    try:
        sys.path.insert(0, SCRIPT_DIR)
        import HaploTE  # noqa: F401
        return True
    except ImportError:
        return False


def _has_r_deps():
    """Check whether R clustering dependencies are available."""
    try:
        result = subprocess.run(
            [_RSCRIPT, "-e",
             "library(pheatmap); library(ggplot2); library(ggdendro); "
             "library(matrixStats); library(reticulate); library(viridis); "
             "library(igraph); library(Polychrome); library(assertthat); "
             "library(plyr); library(RColorBrewer); cat('OK')"],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0 and "OK" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


needs_python_deps = pytest.mark.skipif(
    not _has_haploTE_deps(), reason="HaploTE dependencies not installed"
)

needs_r_deps = pytest.mark.skipif(
    not _has_r_deps(), reason="R clustering dependencies not installed"
)

needs_all_deps = pytest.mark.skipif(
    not (_has_haploTE_deps() and _has_r_deps()),
    reason="Full pipeline dependencies not installed",
)


@needs_python_deps
class TestPythonStep:
    """Integration tests for the Python preprocessing step."""

    @pytest.fixture(autouse=True)
    def setup_output(self, tmp_path):
        self.output = str(tmp_path / "py_output")
        self.cn_out = str(tmp_path / "cn_tables")
        self.ap_out = str(tmp_path / "ap_tables")
        self.pi_out = str(tmp_path / "pi_out")
        self.outlier_out = str(tmp_path / "no_outliers")
        for d in [self.output, self.cn_out, self.ap_out, self.pi_out, self.outlier_out]:
            os.makedirs(d, exist_ok=True)

    def _make_args(self, te_subset=None):
        args = argparse.Namespace(
            te_list=TE_LIST,
            sample_sheet=SAMPLE_SHEET,
            cn_dir=CN_DIR,
            output_dir=self.output,
            pi_filter=0.1,
            cn_filter=0.5,
            min_freq=0.1,
            min_strains=10,
            pi_dir=PI_DIR,
            cn_tables_dir=self.cn_out,
            ap_tables_dir=self.ap_out,
            outlier_tables_dir=self.outlier_out,
            outliers_json=None,
            dist_cutoff=0.5,
            dist_cutoffs_file=None,
            linkage="average",
            color_pal="#6F0000,#009191,#6DB6FF,orange,#490092",
            skip_python=False,
            skip_r=True,
        )
        return args

    def test_calcPi_produces_output(self):
        from HaploTE import summary_stats
        stats = summary_stats(sample_sheet=SAMPLE_SHEET, cn_dir=CN_DIR, pi_dir=self.pi_out)
        stats.calcPi(TE="BARI_DM")
        out_file = os.path.join(self.pi_out, "BARI_DM_pi.npy")
        assert os.path.isfile(out_file)
        arr = np.load(out_file)
        assert arr.ndim >= 1

    def test_CNAP_extraction_produces_tables(self):
        from HaploTE import subfamilyInference
        ci = subfamilyInference(
            TE_list=TE_LIST, sample_sheet=SAMPLE_SHEET,
            cn_dir=CN_DIR, pi_dir=PI_DIR,
            cn_df_path=self.cn_out, ap_df_path=self.ap_out,
        )
        ci.CNAP_extraction(TE="BARI_DM", pi_filter=0.1, CN_filter=0.5, minFreq=0.1, min_strains=10)

        cn_file = os.path.join(self.cn_out, "BARI_DM.CN.GDL.minor.csv")
        ap_file = os.path.join(self.ap_out, "BARI_DM.AP.GDL.minor.csv")
        assert os.path.isfile(cn_file), "CN table not produced"
        assert os.path.isfile(ap_file), "AP table not produced"

        cn_df = pd.read_csv(cn_file)
        assert cn_df.shape[0] == 85, f"Expected 85 rows, got {cn_df.shape[0]}"
        assert cn_df.shape[1] > 0, "CN table has no allele columns"

    def test_CNAP_output_values_nonnegative(self):
        from HaploTE import subfamilyInference
        ci = subfamilyInference(
            TE_list=TE_LIST, sample_sheet=SAMPLE_SHEET,
            cn_dir=CN_DIR, pi_dir=PI_DIR,
            cn_df_path=self.cn_out, ap_df_path=self.ap_out,
        )
        ci.CNAP_extraction(TE="BARI_DM", pi_filter=0.1, CN_filter=0.5, minFreq=0.1, min_strains=10)
        cn_df = pd.read_csv(os.path.join(self.cn_out, "BARI_DM.CN.GDL.minor.csv"))
        assert (cn_df.values >= 0).all(), "CN table contains negative values"

    def test_drop_outliers_creates_reduced_table(self, tmp_path):
        from HaploTE import subfamilyInference
        # First produce a CN table
        ci = subfamilyInference(
            TE_list=TE_LIST, sample_sheet=SAMPLE_SHEET,
            cn_dir=CN_DIR, pi_dir=PI_DIR,
            cn_df_path=self.cn_out, ap_df_path=self.ap_out,
        )
        ci.CNAP_extraction(TE="BLOOD_FULL", pi_filter=0.1, CN_filter=0.5, minFreq=0.1, min_strains=10)

        # Now drop outliers
        ci.drop_outliers(TE="BLOOD_FULL", strain_masks=["I01"])

        outlier_file = os.path.join(self.outlier_out, "BLOOD_FULL.CN.GDL.minor.csv")
        assert os.path.isfile(outlier_file), "Outlier-removed table not produced"

        outlier_df = pd.read_csv(outlier_file)
        orig_df = pd.read_csv(os.path.join(self.cn_out, "BLOOD_FULL.CN.GDL.minor.csv"))
        assert outlier_df.shape[0] < orig_df.shape[0], "Outlier removal didn't reduce row count"

    def test_stricter_filters_produce_fewer_alleles(self):
        from HaploTE import subfamilyInference
        ci = subfamilyInference(
            TE_list=TE_LIST, sample_sheet=SAMPLE_SHEET,
            cn_dir=CN_DIR, pi_dir=PI_DIR,
            cn_df_path=self.cn_out, ap_df_path=self.ap_out,
        )
        # Loose filters
        ci.CNAP_extraction(TE="BARI_DM", pi_filter=0.05, CN_filter=0.1, minFreq=0.05, min_strains=5)
        loose_df = pd.read_csv(os.path.join(self.cn_out, "BARI_DM.CN.GDL.minor.csv"))
        loose_cols = loose_df.shape[1]

        # Moderately strict filters
        ci.CNAP_extraction(TE="BARI_DM", pi_filter=0.15, CN_filter=0.5, minFreq=0.15, min_strains=15)
        strict_path = os.path.join(self.cn_out, "BARI_DM.CN.GDL.minor.csv")
        try:
            strict_df = pd.read_csv(strict_path)
            strict_cols = strict_df.shape[1]
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # Filters were strict enough that no alleles passed — still valid
            strict_cols = 0

        assert strict_cols <= loose_cols, (
            f"Stricter filters should produce <= alleles: strict={strict_cols}, loose={loose_cols}"
        )


@needs_r_deps
class TestRStep:
    """Integration tests for the R clustering step."""

    @pytest.fixture(autouse=True)
    def setup_output(self, tmp_path):
        self.output = str(tmp_path / "r_output")
        os.makedirs(self.output, exist_ok=True)

    def test_single_te_clustering(self):
        """Run R clustering on a single TE with existing CN tables."""
        r_script = os.path.join(SCRIPT_DIR, "run_clustering.R")

        # Create a single-TE list
        te_file = os.path.join(self.output, "single_te.tsv")
        with open(te_file, "w") as f:
            f.write("BARI_DM,TIR\n")

        result = subprocess.run(
            [_RSCRIPT, r_script,
             te_file, SAMPLE_SHEET, CN_TABLES_DIR, OUTLIER_TABLES_DIR,
             CN_DIR, self.output, "average",
             "#6F0000,#009191,#6DB6FF,orange,#490092",
             "0.5", "FALSE"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"R clustering failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        # Check outputs
        haplo_table = os.path.join(self.output, "BARI_DM.haplotypeTable.tsv")
        assert os.path.isfile(haplo_table), f"Missing haplotype table. Files: {os.listdir(self.output)}"

        # Validate table structure
        df = pd.read_csv(haplo_table, sep="\t")
        assert "Cluster ID" in df.columns
        assert "Alleles" in df.columns
        assert df.shape[0] >= 1

    def test_r_output_plots_created(self):
        """Check that PNG plots are generated."""
        r_script = os.path.join(SCRIPT_DIR, "run_clustering.R")

        te_file = os.path.join(self.output, "single_te.tsv")
        with open(te_file, "w") as f:
            f.write("BARI_DM,TIR\n")

        subprocess.run(
            [_RSCRIPT, r_script,
             te_file, SAMPLE_SHEET, CN_TABLES_DIR, OUTLIER_TABLES_DIR,
             CN_DIR, self.output, "average",
             "#6F0000,#009191,#6DB6FF,orange,#490092",
             "0.5", "FALSE"],
            capture_output=True, text=True, timeout=120,
        )

        plots_dir = os.path.join(self.output, "PLOTS")
        assert os.path.isdir(plots_dir), "PLOTS directory not created"

        expected_plots = [
            "BARI_DM.pop.hmap.png",
            "BARI_DM.haplotype.hmap.png",
            "BARI_DM.dendro.png",
        ]
        for plot_name in expected_plots:
            path = os.path.join(plots_dir, plot_name)
            assert os.path.isfile(path), f"Missing plot: {plot_name}"
            assert os.path.getsize(path) > 0, f"Empty plot file: {plot_name}"

    def test_outlier_te_clustering(self):
        """Run R clustering for a TE that uses outlier-removed tables."""
        r_script = os.path.join(SCRIPT_DIR, "run_clustering.R")

        te_file = os.path.join(self.output, "single_te.tsv")
        with open(te_file, "w") as f:
            f.write("BLOOD_FULL,LTR\n")

        result = subprocess.run(
            [_RSCRIPT, r_script,
             te_file, SAMPLE_SHEET, CN_TABLES_DIR, OUTLIER_TABLES_DIR,
             CN_DIR, self.output, "average",
             "#6F0000,#009191,#6DB6FF,orange,#490092",
             "0.4", "TRUE"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"R clustering failed:\nstderr: {result.stderr}"

        haplo_table = os.path.join(self.output, "BLOOD_FULL.haplotypeTable.tsv")
        assert os.path.isfile(haplo_table)


@needs_all_deps
class TestFullPipeline:
    """End-to-end integration tests running the complete CLI pipeline."""

    @pytest.fixture(autouse=True)
    def setup_output(self, tmp_path):
        self.output = str(tmp_path / "full_output")
        os.makedirs(self.output, exist_ok=True)

    def test_full_pipeline_single_te(self, tmp_path):
        """Run the full pipeline on a single TE via subprocess."""
        # Create a single-TE list
        te_file = str(tmp_path / "single_te.tsv")
        with open(te_file, "w") as f:
            f.write("BARI_DM,TIR\n")

        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "run_pipeline.py"),
             "--te-list", te_file,
             "--sample-sheet", SAMPLE_SHEET,
             "--cn-dir", CN_DIR,
             "--output-dir", self.output,
             "--dist-cutoff", "0.5"],
            capture_output=True, text=True, timeout=300,
            cwd=SCRIPT_DIR,
        )
        assert result.returncode == 0, f"Pipeline failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        # Check R outputs
        haplo_table = os.path.join(self.output, "BARI_DM.haplotypeTable.tsv")
        assert os.path.isfile(haplo_table)

        plots_dir = os.path.join(self.output, "PLOTS")
        assert os.path.isdir(plots_dir)

    def test_full_pipeline_with_outliers(self, tmp_path):
        """Run pipeline with outlier JSON on a TE that needs outlier removal."""
        te_file = str(tmp_path / "single_te.tsv")
        with open(te_file, "w") as f:
            f.write("BLOOD_FULL,LTR\n")

        outlier_file = str(tmp_path / "outliers.json")
        with open(outlier_file, "w") as f:
            json.dump({"BLOOD_FULL": ["I01"]}, f)

        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "run_pipeline.py"),
             "--te-list", te_file,
             "--sample-sheet", SAMPLE_SHEET,
             "--cn-dir", CN_DIR,
             "--output-dir", self.output,
             "--outliers-json", outlier_file,
             "--dist-cutoff", "0.4"],
            capture_output=True, text=True, timeout=300,
            cwd=SCRIPT_DIR,
        )
        assert result.returncode == 0, f"Pipeline failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_full_pipeline_with_cutoffs_file(self, tmp_path):
        """Run pipeline with a per-TE cutoffs file."""
        te_file = str(tmp_path / "two_te.tsv")
        with open(te_file, "w") as f:
            f.write("BARI_DM,TIR\nBS,LINE\n")

        cutoffs_file = str(tmp_path / "cutoffs.tsv")
        with open(cutoffs_file, "w") as f:
            f.write("BARI_DM\t0.45\nBS\t0.35\n")

        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "run_pipeline.py"),
             "--te-list", te_file,
             "--sample-sheet", SAMPLE_SHEET,
             "--cn-dir", CN_DIR,
             "--output-dir", self.output,
             "--dist-cutoffs-file", cutoffs_file],
            capture_output=True, text=True, timeout=300,
            cwd=SCRIPT_DIR,
        )
        assert result.returncode == 0, f"Pipeline failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        for te in ["BARI_DM", "BS"]:
            haplo_table = os.path.join(self.output, f"{te}.haplotypeTable.tsv")
            assert os.path.isfile(haplo_table), f"Missing output for {te}"

    def test_skip_python_uses_existing_tables(self, tmp_path):
        """With --skip-python, pipeline should use pre-existing CN tables."""
        te_file = str(tmp_path / "single_te.tsv")
        with open(te_file, "w") as f:
            f.write("BARI_DM,TIR\n")

        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "run_pipeline.py"),
             "--te-list", te_file,
             "--sample-sheet", SAMPLE_SHEET,
             "--cn-dir", CN_DIR,
             "--output-dir", self.output,
             "--skip-python",
             "--cn-tables-dir", CN_TABLES_DIR,
             "--dist-cutoff", "0.5"],
            capture_output=True, text=True, timeout=300,
            cwd=SCRIPT_DIR,
        )
        assert result.returncode == 0, f"Pipeline failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        assert "Skipping Python preprocessing" in result.stdout


# ===========================================================================
# Output validation tests
# ===========================================================================

@needs_r_deps
class TestOutputValidation:
    """Validate the format and content of pipeline output files."""

    @pytest.fixture(autouse=True)
    def run_clustering(self, tmp_path):
        """Run R clustering once for BARI_DM to produce outputs."""
        self.output = str(tmp_path / "validation_output")
        os.makedirs(self.output, exist_ok=True)

        r_script = os.path.join(SCRIPT_DIR, "run_clustering.R")
        te_file = os.path.join(self.output, "te.tsv")
        with open(te_file, "w") as f:
            f.write("BARI_DM,TIR\n")

        result = subprocess.run(
            [_RSCRIPT, r_script,
             te_file, SAMPLE_SHEET, CN_TABLES_DIR, OUTLIER_TABLES_DIR,
             CN_DIR, self.output, "average",
             "#6F0000,#009191,#6DB6FF,orange,#490092",
             "0.5", "FALSE"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"Setup failed: {result.stderr}"

    def test_haplotype_table_columns(self):
        df = pd.read_csv(os.path.join(self.output, "BARI_DM.haplotypeTable.tsv"), sep="\t")
        required_cols = ["Cluster ID", "Alleles", "Cluster Size"]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

    def test_haplotype_table_has_freq_columns(self):
        df = pd.read_csv(os.path.join(self.output, "BARI_DM.haplotypeTable.tsv"), sep="\t")
        freq_cols = [c for c in df.columns if "FREQ" in c]
        assert len(freq_cols) > 0, "No frequency columns found"

    def test_haplotype_table_has_mean_cn_columns(self):
        df = pd.read_csv(os.path.join(self.output, "BARI_DM.haplotypeTable.tsv"), sep="\t")
        cn_cols = [c for c in df.columns if "MEAN CN" in c]
        assert len(cn_cols) > 0, "No mean CN columns found"

    def test_haplotype_table_cluster_ids_format(self):
        df = pd.read_csv(os.path.join(self.output, "BARI_DM.haplotypeTable.tsv"), sep="\t")
        for cid in df["Cluster ID"]:
            assert str(cid).startswith("Cluster_"), f"Unexpected cluster ID format: {cid}"

    def test_haplotype_table_alleles_format(self):
        df = pd.read_csv(os.path.join(self.output, "BARI_DM.haplotypeTable.tsv"), sep="\t")
        for alleles in df["Alleles"]:
            if pd.isna(alleles):
                continue
            for allele in str(alleles).split(","):
                parts = allele.strip().split("_")
                assert len(parts) == 2, f"Allele not in NT_pos format: {allele}"
                assert parts[0] in ("A", "T", "C", "G"), f"Unexpected NT: {parts[0]}"

    def test_cluster_cn_table_produced(self):
        cn_file = os.path.join(self.output, "BARI_DM_cluster_CN.tsv")
        assert os.path.isfile(cn_file)
        df = pd.read_csv(cn_file, sep="\t")
        assert "POP" in df.columns
        assert "SAMPLE_ID" in df.columns
        assert df.shape[0] == 85, f"Expected 85 rows, got {df.shape[0]}"

    def test_plot_files_are_valid_png(self):
        plots_dir = os.path.join(self.output, "PLOTS")
        for png_file in os.listdir(plots_dir):
            if not png_file.endswith(".png"):
                continue
            path = os.path.join(plots_dir, png_file)
            with open(path, "rb") as f:
                header = f.read(8)
            # PNG magic bytes
            assert header[:4] == b"\x89PNG", f"{png_file} is not a valid PNG"
