"""
Test quantification accuracy by comparing mokume's quantification implementations
with DirectLFQ and DIA-NN reported values.

This module contains two test suites:

1. TestSmallDiannSubset: Uses a small subset DIA-NN report from directLFQ test suite.
   Note: DIA-NN's PG.MaxLFQ is computed from ALL precursors, but this report only
   contains a subset. Direct comparison shows lower correlation (~0.55-0.61).

2. TestDiannWithSdrf: Uses a full DIA-NN report from PRIDE (PXD063291) with SDRF.
   This provides a more complete dataset for validation.

Test data sources:
- Small subset: https://github.com/MannLabs/directlfq/tree/main/test_data
- Full dataset: https://ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomes/pmultiqc/example-projects/PXD063291.zip
"""

import os
import tempfile
import zipfile
import urllib.request
import warnings
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Suppress warnings during tests
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
SMALL_DIANN_REPORT = TEST_DATA_DIR / "diann_test_report.tsv"

# PRIDE dataset URL
PRIDE_DATASET_URL = "https://ftp.pride.ebi.ac.uk/pub/databases/pride/resources/proteomes/pmultiqc/example-projects/PXD063291.zip"
PRIDE_CACHE_DIR = TEST_DATA_DIR / "pride_cache"


def load_small_diann_report():
    """Load small DIA-NN report from directLFQ test suite."""
    if not SMALL_DIANN_REPORT.exists():
        pytest.skip(f"Test data not found: {SMALL_DIANN_REPORT}")
    return pd.read_csv(SMALL_DIANN_REPORT, sep="\t")


def download_pride_dataset():
    """
    Download and cache the PRIDE PXD063291 dataset.
    Returns path to the extracted DIA-NN report.
    """
    PRIDE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = PRIDE_CACHE_DIR / "PXD063291.zip"

    # Check if already extracted
    extracted_files = list(PRIDE_CACHE_DIR.glob("*.tsv")) + list(PRIDE_CACHE_DIR.glob("**/diann-output/*.tsv"))
    if extracted_files:
        # Find the main report file
        for f in extracted_files:
            if "report" in f.name.lower() and f.suffix == ".tsv":
                return f
        # Return first TSV if no report found
        return extracted_files[0]

    # Download if not cached
    if not zip_path.exists():
        print(f"Downloading PRIDE dataset from {PRIDE_DATASET_URL}...")
        try:
            urllib.request.urlretrieve(PRIDE_DATASET_URL, zip_path)
        except Exception as e:
            pytest.skip(f"Failed to download PRIDE dataset: {e}")

    # Extract
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(PRIDE_CACHE_DIR)
    except Exception as e:
        pytest.skip(f"Failed to extract PRIDE dataset: {e}")

    # Find the DIA-NN report
    extracted_files = list(PRIDE_CACHE_DIR.rglob("*.tsv"))
    for f in extracted_files:
        if "report" in f.name.lower():
            return f

    if extracted_files:
        return extracted_files[0]

    pytest.skip("No TSV files found in PRIDE dataset")


def prepare_peptide_data(diann_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DIA-NN report to mokume peptide format.

    DIA-NN columns -> mokume columns:
    - Protein.Group -> ProteinName
    - Run -> SampleID
    - Precursor.Id -> PeptideSequence (using precursor as unique peptide identifier)
    - Precursor.Quantity -> NormIntensity
    """
    peptide_df = diann_df[["Protein.Group", "Run", "Precursor.Id", "Precursor.Quantity"]].copy()
    peptide_df = peptide_df.rename(columns={
        "Protein.Group": "ProteinName",
        "Run": "SampleID",
        "Precursor.Id": "PeptideSequence",
        "Precursor.Quantity": "NormIntensity",
    })

    # Remove rows with missing intensity
    peptide_df = peptide_df[peptide_df["NormIntensity"].notna()]
    peptide_df = peptide_df[peptide_df["NormIntensity"] > 0]

    return peptide_df


def extract_diann_maxlfq(diann_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract DIA-NN's MaxLFQ values (PG.MaxLFQ) per protein per sample.

    NOTE: These values are computed from ALL precursors by DIA-NN,
    not just those in this report. Direct comparison with mokume's
    values (computed from subset) may show lower correlation.
    """
    if "PG.MaxLFQ" not in diann_df.columns:
        return pd.DataFrame(columns=["ProteinName", "SampleID", "DIANNMaxLFQ"])

    maxlfq_df = diann_df[["Protein.Group", "Run", "PG.MaxLFQ"]].drop_duplicates()
    maxlfq_df = maxlfq_df.rename(columns={
        "Protein.Group": "ProteinName",
        "Run": "SampleID",
        "PG.MaxLFQ": "DIANNMaxLFQ",
    })
    maxlfq_df = maxlfq_df[maxlfq_df["DIANNMaxLFQ"].notna()]
    maxlfq_df = maxlfq_df[maxlfq_df["DIANNMaxLFQ"] > 0]
    return maxlfq_df


def compute_correlation(df1: pd.DataFrame, df2: pd.DataFrame,
                        col1: str, col2: str,
                        protein_col: str = "ProteinName",
                        sample_col: str = "SampleID") -> dict:
    """
    Compute correlation between two quantification results.
    """
    # Use suffixes to handle same column names
    merged = pd.merge(
        df1[[protein_col, sample_col, col1]],
        df2[[protein_col, sample_col, col2]],
        on=[protein_col, sample_col],
        how="inner",
        suffixes=("_x", "_y"),
    )

    if len(merged) == 0:
        return {"pearson": np.nan, "spearman": np.nan, "n": 0}

    # Determine actual column names after merge (handles same-name case)
    col1_actual = col1 if col1 in merged.columns else f"{col1}_x"
    col2_actual = col2 if col2 in merged.columns else f"{col2}_y"

    # Log transform for better correlation
    x = np.log2(merged[col1_actual].values + 1)
    y = np.log2(merged[col2_actual].values + 1)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return {"pearson": np.nan, "spearman": np.nan, "n": len(x)}

    pearson_r, _ = stats.pearsonr(x, y)
    spearman_r, _ = stats.spearmanr(x, y)

    return {
        "pearson": pearson_r,
        "spearman": spearman_r,
        "n": len(x),
        "mean_abs_log_diff": np.mean(np.abs(x - y)),
    }


def print_comparison_table(results: dict, intensity_cols: dict, dataset_name: str,
                           peptide_count: int, protein_count: int, sample_count: int,
                           precursor_count: int):
    """Print a formatted comparison table of all methods."""
    print("\n" + "=" * 90)
    print(f"QUANTIFICATION METHOD COMPARISON - {dataset_name}")
    print("=" * 90)
    print(f"Test Data:")
    print(f"  Peptide measurements: {peptide_count:,}")
    print(f"  Proteins: {protein_count}")
    print(f"  Samples: {sample_count}")
    print(f"  Precursors: {precursor_count}")
    print("-" * 90)
    print("CROSS-METHOD CORRELATIONS (Spearman)")
    print("-" * 90)

    method_names = list(results.keys())

    # Header
    print(f"{'':18}", end="")
    for name in method_names:
        print(f"{name[:14]:>15}", end="")
    print()
    print("-" * 90)

    # Rows
    for name1 in method_names:
        print(f"{name1:18}", end="")
        for name2 in method_names:
            if name1 == name2:
                print(f"{'1.000':>15}", end="")
            else:
                corr = compute_correlation(
                    results[name1], results[name2],
                    intensity_cols[name1], intensity_cols[name2]
                )
                if np.isnan(corr['spearman']):
                    print(f"{'N/A':>15}", end="")
                else:
                    print(f"{corr['spearman']:>15.3f}", end="")
        print()

    print("=" * 90)


def run_all_quantification_methods(peptide_df: pd.DataFrame, diann_maxlfq: pd.DataFrame = None):
    """
    Run all quantification methods and return results dict.

    Note: MaxLFQQuantification automatically uses DirectLFQ if available,
    falling back to built-in implementation otherwise. We test both paths
    when DirectLFQ is available.
    """
    from mokume.quantification import (
        MaxLFQQuantification,
        Top3Quantification,
        AllPeptidesQuantification,
        is_directlfq_available,
    )

    results = {}
    intensity_cols = {}

    # Add DIA-NN MaxLFQ if available
    if diann_maxlfq is not None and len(diann_maxlfq) > 0:
        results["DIA-NN MaxLFQ"] = diann_maxlfq
        intensity_cols["DIA-NN MaxLFQ"] = "DIANNMaxLFQ"

    # Mokume MaxLFQ (uses DirectLFQ if available, otherwise built-in)
    maxlfq = MaxLFQQuantification(min_peptides=1, threads=-1)
    results["Mokume MaxLFQ"] = maxlfq.quantify(
        peptide_df,
        protein_column="ProteinName",
        peptide_column="PeptideSequence",
        intensity_column="NormIntensity",
        sample_column="SampleID",
    )
    intensity_cols["Mokume MaxLFQ"] = "MaxLFQIntensity"

    # Top3
    top3 = Top3Quantification()
    results["Top3"] = top3.quantify(
        peptide_df,
        protein_column="ProteinName",
        peptide_column="PeptideSequence",
        intensity_column="NormIntensity",
        sample_column="SampleID",
    )
    intensity_cols["Top3"] = "Top3Intensity"

    # Sum
    sum_quant = AllPeptidesQuantification()
    results["Sum"] = sum_quant.quantify(
        peptide_df,
        protein_column="ProteinName",
        peptide_column="PeptideSequence",
        intensity_column="NormIntensity",
        sample_column="SampleID",
    )
    intensity_cols["Sum"] = "SumIntensity"

    # DirectLFQ standalone (for comparison when available)
    if is_directlfq_available():
        try:
            from mokume.quantification import DirectLFQQuantification
            directlfq = DirectLFQQuantification(min_nonan=1)
            results["DirectLFQ"] = directlfq.quantify(
                peptide_df,
                protein_column="ProteinName",
                peptide_column="PeptideSequence",
                intensity_column="NormIntensity",
                sample_column="SampleID",
            )
            intensity_cols["DirectLFQ"] = "DirectLFQIntensity"
        except Exception as e:
            print(f"DirectLFQ failed: {e}")

    return results, intensity_cols


class TestSmallDiannSubset:
    """
    Test suite using small DIA-NN subset from directLFQ test suite.

    IMPORTANT: This is a SUBSET dataset with only ~626 precursors.
    DIA-NN's PG.MaxLFQ was computed from ALL precursors, so correlation
    between computed values and DIA-NN MaxLFQ will be lower (~0.55-0.61).
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load test data."""
        self.diann_df = load_small_diann_report()
        self.peptide_df = prepare_peptide_data(self.diann_df)
        self.diann_maxlfq = extract_diann_maxlfq(self.diann_df)

    def test_small_diann_data_loaded(self):
        """Verify small DIA-NN subset data is loaded correctly."""
        assert len(self.diann_df) > 0, "DIA-NN report should not be empty"
        assert len(self.peptide_df) > 0, "Peptide data should not be empty"

        print(f"\n[Small DIA-NN Subset] Test data summary:")
        print(f"  Peptide measurements: {len(self.peptide_df):,}")
        print(f"  Unique proteins: {self.peptide_df['ProteinName'].nunique()}")
        print(f"  Unique samples: {self.peptide_df['SampleID'].nunique()}")
        print(f"  Unique precursors: {self.peptide_df['PeptideSequence'].nunique()}")

    def test_small_diann_maxlfq_produces_results(self):
        """Test that mokume MaxLFQ produces valid results on small subset."""
        from mokume.quantification import MaxLFQQuantification

        maxlfq = MaxLFQQuantification(min_peptides=1, threads=1)
        result = maxlfq.quantify(
            self.peptide_df,
            protein_column="ProteinName",
            peptide_column="PeptideSequence",
            intensity_column="NormIntensity",
            sample_column="SampleID",
        )

        assert len(result) > 0, "MaxLFQ should produce results"
        assert "MaxLFQIntensity" in result.columns
        assert result["MaxLFQIntensity"].notna().sum() > 0, "Should have non-NaN intensities"
        assert (result["MaxLFQIntensity"] > 0).all(), "All intensities should be positive"

        print(f"\n[Small DIA-NN Subset] MaxLFQ results:")
        print(f"  Protein-sample combinations: {len(result):,}")
        print(f"  Intensity range: {result['MaxLFQIntensity'].min():.2f} - {result['MaxLFQIntensity'].max():.2f}")

    def test_small_diann_maxlfq_parallelization(self):
        """Test that parallel MaxLFQ produces same results as single-threaded."""
        from mokume.quantification import MaxLFQQuantification

        maxlfq_single = MaxLFQQuantification(min_peptides=1, threads=1)
        result_single = maxlfq_single.quantify(
            self.peptide_df,
            protein_column="ProteinName",
            peptide_column="PeptideSequence",
            intensity_column="NormIntensity",
            sample_column="SampleID",
        )

        maxlfq_parallel = MaxLFQQuantification(min_peptides=1, threads=2)
        result_parallel = maxlfq_parallel.quantify(
            self.peptide_df,
            protein_column="ProteinName",
            peptide_column="PeptideSequence",
            intensity_column="NormIntensity",
            sample_column="SampleID",
        )

        corr = compute_correlation(
            result_single, result_parallel,
            "MaxLFQIntensity", "MaxLFQIntensity"
        )

        print(f"\n[Small DIA-NN Subset] Single-threaded vs Parallel MaxLFQ:")
        print(f"  Pearson correlation: {corr['pearson']:.4f}")

        assert corr["pearson"] > 0.9999, "Parallel should produce same results as single-threaded"

    def test_small_diann_full_comparison(self):
        """
        Full comparison of all quantification methods on small DIA-NN subset.

        This test prints the complete comparison table for reference.
        """
        results, intensity_cols = run_all_quantification_methods(
            self.peptide_df, self.diann_maxlfq
        )

        # Print comparison table
        print_comparison_table(
            results, intensity_cols,
            dataset_name="Small DIA-NN Subset (directLFQ test data)",
            peptide_count=len(self.peptide_df),
            protein_count=self.peptide_df['ProteinName'].nunique(),
            sample_count=self.peptide_df['SampleID'].nunique(),
            precursor_count=self.peptide_df['PeptideSequence'].nunique(),
        )

        print("\nNOTE: DIA-NN MaxLFQ shows lower correlation (~0.55-0.61) because:")
        print("  - DIA-NN computed MaxLFQ from ALL precursors")
        print("  - This report only contains a SUBSET (626 precursors)")
        print("  - Most proteins have only 1 precursor in this subset")

        # Validate that computed methods correlate well with each other
        corr_maxlfq_top3 = compute_correlation(
            results["Mokume MaxLFQ"], results["Top3"],
            intensity_cols["Mokume MaxLFQ"], intensity_cols["Top3"]
        )
        assert corr_maxlfq_top3["spearman"] > 0.9, \
            f"MaxLFQ and Top3 should correlate >0.9, got {corr_maxlfq_top3['spearman']:.3f}"

    def test_small_diann_directlfq_comparison(self):
        """Compare DirectLFQ with mokume MaxLFQ on small subset."""
        try:
            from mokume.quantification import DirectLFQQuantification, is_directlfq_available
        except ImportError:
            pytest.skip("DirectLFQ import not available")

        if not is_directlfq_available():
            pytest.skip("DirectLFQ not installed (pip install mokume[directlfq])")

        from mokume.quantification import MaxLFQQuantification

        directlfq = DirectLFQQuantification(min_nonan=1)
        result_directlfq = directlfq.quantify(
            self.peptide_df,
            protein_column="ProteinName",
            peptide_column="PeptideSequence",
            intensity_column="NormIntensity",
            sample_column="SampleID",
        )

        # Test default MaxLFQ (should use DirectLFQ when available)
        maxlfq = MaxLFQQuantification(min_peptides=1, threads=1)
        assert maxlfq.using_directlfq, "MaxLFQ should use DirectLFQ when available"

        result_maxlfq = maxlfq.quantify(
            self.peptide_df,
            protein_column="ProteinName",
            peptide_column="PeptideSequence",
            intensity_column="NormIntensity",
            sample_column="SampleID",
        )

        corr = compute_correlation(
            result_directlfq, result_maxlfq,
            "DirectLFQIntensity", "MaxLFQIntensity"
        )

        print(f"\n[Small DIA-NN Subset] DirectLFQ vs Mokume MaxLFQ:")
        print(f"  Using DirectLFQ backend: {maxlfq.using_directlfq}")
        print(f"  Pearson correlation:  {corr['pearson']:.4f}")
        print(f"  Spearman correlation: {corr['spearman']:.4f}")
        print(f"  Number of comparisons: {corr['n']:,}")

        assert corr["spearman"] > 0.99, "MaxLFQ with DirectLFQ should match DirectLFQ exactly"

    def test_small_diann_builtin_fallback(self):
        """Test the built-in MaxLFQ fallback implementation."""
        from mokume.quantification import MaxLFQQuantification

        # Force built-in implementation
        maxlfq_builtin = MaxLFQQuantification(min_peptides=1, threads=1, force_builtin=True)
        assert not maxlfq_builtin.using_directlfq, "Should use built-in when forced"
        assert maxlfq_builtin.name == "MaxLFQ (built-in)"

        result_builtin = maxlfq_builtin.quantify(
            self.peptide_df,
            protein_column="ProteinName",
            peptide_column="PeptideSequence",
            intensity_column="NormIntensity",
            sample_column="SampleID",
        )

        assert len(result_builtin) > 0, "Built-in MaxLFQ should produce results"
        assert "MaxLFQIntensity" in result_builtin.columns

        print(f"\n[Small DIA-NN Subset] Built-in MaxLFQ fallback:")
        print(f"  Protein-sample combinations: {len(result_builtin):,}")
        print(f"  Intensity range: {result_builtin['MaxLFQIntensity'].min():.2f} - {result_builtin['MaxLFQIntensity'].max():.2f}")


class TestDiannWithSdrf:
    """
    Test suite using full DIA-NN report from PRIDE (PXD063291).

    This dataset is downloaded from PRIDE FTP and cached locally.
    It provides a more complete dataset for validation compared to the small subset.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Download and load PRIDE dataset."""
        try:
            report_path = download_pride_dataset()
            self.diann_df = pd.read_csv(report_path, sep="\t")
            self.peptide_df = prepare_peptide_data(self.diann_df)
            self.diann_maxlfq = extract_diann_maxlfq(self.diann_df)
            self.report_path = report_path
        except Exception as e:
            pytest.skip(f"Failed to load PRIDE dataset: {e}")

    def test_pride_data_loaded(self):
        """Verify PRIDE dataset is loaded correctly."""
        assert len(self.diann_df) > 0, "DIA-NN report should not be empty"
        assert len(self.peptide_df) > 0, "Peptide data should not be empty"

        print(f"\n[PRIDE PXD063291] Test data summary:")
        print(f"  Source: {self.report_path}")
        print(f"  Peptide measurements: {len(self.peptide_df):,}")
        print(f"  Unique proteins: {self.peptide_df['ProteinName'].nunique()}")
        print(f"  Unique samples: {self.peptide_df['SampleID'].nunique()}")
        print(f"  Unique precursors: {self.peptide_df['PeptideSequence'].nunique()}")

    def test_pride_maxlfq_produces_results(self):
        """Test that mokume MaxLFQ produces valid results on PRIDE dataset."""
        from mokume.quantification import MaxLFQQuantification

        maxlfq = MaxLFQQuantification(min_peptides=1, threads=-1)
        result = maxlfq.quantify(
            self.peptide_df,
            protein_column="ProteinName",
            peptide_column="PeptideSequence",
            intensity_column="NormIntensity",
            sample_column="SampleID",
        )

        assert len(result) > 0, "MaxLFQ should produce results"
        assert "MaxLFQIntensity" in result.columns
        assert result["MaxLFQIntensity"].notna().sum() > 0, "Should have non-NaN intensities"

        print(f"\n[PRIDE PXD063291] MaxLFQ results:")
        print(f"  Protein-sample combinations: {len(result):,}")
        print(f"  Intensity range: {result['MaxLFQIntensity'].min():.2f} - {result['MaxLFQIntensity'].max():.2f}")

    def test_pride_full_comparison(self):
        """
        Full comparison of all quantification methods on PRIDE dataset.

        This test prints the complete comparison table for reference.
        Expected: Higher correlation with DIA-NN MaxLFQ since this is
        the complete dataset (not a subset).
        """
        results, intensity_cols = run_all_quantification_methods(
            self.peptide_df, self.diann_maxlfq
        )

        # Print comparison table
        print_comparison_table(
            results, intensity_cols,
            dataset_name="PRIDE PXD063291 (Full DIA-NN Report with SDRF)",
            peptide_count=len(self.peptide_df),
            protein_count=self.peptide_df['ProteinName'].nunique(),
            sample_count=self.peptide_df['SampleID'].nunique(),
            precursor_count=self.peptide_df['PeptideSequence'].nunique(),
        )

        # Validate that computed methods correlate well with each other
        corr_maxlfq_top3 = compute_correlation(
            results["Mokume MaxLFQ"], results["Top3"],
            intensity_cols["Mokume MaxLFQ"], intensity_cols["Top3"]
        )

        print(f"\nValidation: MaxLFQ vs Top3 Spearman = {corr_maxlfq_top3['spearman']:.3f}")

        assert corr_maxlfq_top3["spearman"] > 0.85, \
            f"MaxLFQ and Top3 should correlate >0.85, got {corr_maxlfq_top3['spearman']:.3f}"

        # If DIA-NN MaxLFQ is available, check correlation
        if "DIA-NN MaxLFQ" in results and len(results["DIA-NN MaxLFQ"]) > 0:
            corr_diann = compute_correlation(
                results["DIA-NN MaxLFQ"], results["Mokume MaxLFQ"],
                intensity_cols["DIA-NN MaxLFQ"], intensity_cols["Mokume MaxLFQ"]
            )
            print(f"Validation: DIA-NN MaxLFQ vs Mokume MaxLFQ Spearman = {corr_diann['spearman']:.3f}")


# Standalone test function for quick runs
def test_small_diann_summary():
    """
    Quick summary test for small DIA-NN subset.
    Run with: pytest tests/test_quantification_accuracy.py::test_small_diann_summary -v -s
    """
    diann_df = load_small_diann_report()
    peptide_df = prepare_peptide_data(diann_df)
    diann_maxlfq = extract_diann_maxlfq(diann_df)

    results, intensity_cols = run_all_quantification_methods(peptide_df, diann_maxlfq)

    print_comparison_table(
        results, intensity_cols,
        dataset_name="Small DIA-NN Subset (directLFQ test data)",
        peptide_count=len(peptide_df),
        protein_count=peptide_df['ProteinName'].nunique(),
        sample_count=peptide_df['SampleID'].nunique(),
        precursor_count=peptide_df['PeptideSequence'].nunique(),
    )


if __name__ == "__main__":
    print("Running small DIA-NN subset test...")
    test_small_diann_summary()
