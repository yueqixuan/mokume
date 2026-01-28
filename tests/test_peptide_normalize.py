import logging

import pytest
from mokume.normalization.peptide import (
    peptide_normalization,
    SQLFilterBuilder,
    Feature,
)
from pathlib import Path

TESTS_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def test_feature_assembly():
    """
    Test the peptide normalization process by setting up arguments for the
    `peptide_normalization` function and executing it. This test checks the
    function's ability to process a feature table from a parquet file and an
    SDRF file, applying various filtering and normalization steps, and saving
    the output to a CSV file. It ensures that the output file is removed before
    the test to avoid conflicts.

    The test uses the following parameters:
    - parquet: Path to the input parquet file containing feature data.
    - sdrf: Path to the SDRF file for experimental metadata.
    - min_aa: Minimum number of amino acids required for peptides.
    - min_unique: Minimum number of unique peptides required for proteins.
    - remove_ids: Path to a file with protein IDs to remove, if any.
    - remove_decoy_contaminants: Flag to remove decoy and contaminant proteins.
    - remove_low_frequency_peptides: Flag to remove low-frequency peptides.
    - output: Path to the output CSV file for normalized peptide intensities.
    - skip_normalization: Flag to skip the normalization process.
    - nmethod: Method for feature-level normalization.
    - pnmethod: Method for peptide-level normalization.
    - log2: Flag to apply log2 transformation to intensities.
    - save_parquet: Flag to save the output as a parquet file.
    """

    args = {
        "parquet": str(TESTS_DIR / "example/feature.parquet"),
        "sdrf": str(TESTS_DIR / "example/PXD017834-TMT.sdrf.tsv"),
        "min_aa": 7,
        "min_unique": 2,
        "remove_ids": None,
        "remove_decoy_contaminants": True,
        "remove_low_frequency_peptides": True,
        "output": str(TESTS_DIR / "example" / "out" / "PXD017834-peptides-norm.csv"),
        "skip_normalization": False,
        "nmethod": "median",
        "pnmethod": "none",
        "log2": True,
        "save_parquet": True,
    }
    logger.info(args)
    out = Path(args["output"])
    if out.exists():
        out.unlink()
    peptide_normalization(**args)


class TestSQLFilterBuilder:
    """Tests for the SQLFilterBuilder class."""

    def test_default_where_clause(self):
        """Test that default filter builder generates expected WHERE clause."""
        builder = SQLFilterBuilder()
        where_clause = builder.build_where_clause()

        # Should include intensity > 0
        assert "intensity > 0" in where_clause
        # Should include peptide length filter
        assert 'LENGTH("sequence") >= 7' in where_clause
        # Should include unique peptide filter
        assert '"unique" = 1' in where_clause
        # Should include contaminant filters
        assert "CONTAMINANT" in where_clause
        assert "DECOY" in where_clause
        assert "ENTRAP" in where_clause
        assert "NOT LIKE" in where_clause

    def test_custom_contaminant_patterns(self):
        """Test filter builder with custom contaminant patterns."""
        builder = SQLFilterBuilder(
            contaminant_patterns=["CONTAM", "REV_"],
            min_peptide_length=5,
        )
        where_clause = builder.build_where_clause()

        assert "CONTAM" in where_clause
        assert "REV_" in where_clause
        assert "DECOY" not in where_clause
        assert 'LENGTH("sequence") >= 5' in where_clause

    def test_disable_contaminant_filter(self):
        """Test that contaminant filter can be disabled."""
        builder = SQLFilterBuilder(remove_contaminants=False)
        where_clause = builder.build_where_clause()

        assert "CONTAMINANT" not in where_clause
        assert "DECOY" not in where_clause
        # Other filters should still be present
        assert "intensity > 0" in where_clause

    def test_min_intensity_threshold(self):
        """Test that min intensity threshold is applied."""
        builder = SQLFilterBuilder(min_intensity=1000.0)
        where_clause = builder.build_where_clause()

        assert "intensity >= 1000.0" in where_clause

    def test_disable_unique_requirement(self):
        """Test that unique peptide requirement can be disabled."""
        builder = SQLFilterBuilder(require_unique=False)
        where_clause = builder.build_where_clause()

        assert '"unique" = 1' not in where_clause


class TestFeatureWithFiltering:
    """Tests for Feature class with filter_builder."""

    @pytest.fixture
    def feature_path(self):
        """Path to test feature parquet file."""
        return str(TESTS_DIR / "example/feature.parquet")

    def test_feature_without_filter(self, feature_path):
        """Test Feature class works without filter_builder (backward compat)."""
        feature = Feature(feature_path)

        assert feature.filter_builder is None
        # Methods should still work
        samples = feature.get_unique_samples()
        assert len(samples) > 0

    def test_feature_with_filter_builder(self, feature_path):
        """Test Feature class accepts and stores filter_builder."""
        builder = SQLFilterBuilder(
            remove_contaminants=True,
            min_peptide_length=7,
        )
        feature = Feature(feature_path, filter_builder=builder)

        assert feature.filter_builder is not None
        assert feature.filter_builder.remove_contaminants is True
        assert feature.filter_builder.min_peptide_length == 7

    def test_get_median_map_with_filter(self, feature_path):
        """Test that get_median_map uses filter_builder when provided."""
        # Without filter
        feature_no_filter = Feature(feature_path)
        med_map_unfiltered = feature_no_filter.get_median_map()

        # With filter (excluding contaminants)
        builder = SQLFilterBuilder(remove_contaminants=True)
        feature_filtered = Feature(feature_path, filter_builder=builder)
        med_map_filtered = feature_filtered.get_median_map()

        # Both should return results
        assert len(med_map_unfiltered) > 0
        assert len(med_map_filtered) > 0

        # The samples should be the same (filtering applies to features, not samples)
        assert set(med_map_unfiltered.keys()) == set(med_map_filtered.keys())

    def test_get_low_frequency_peptides_with_filter(self, feature_path):
        """Test that get_low_frequency_peptides uses filter_builder."""
        # Without filter
        feature_no_filter = Feature(feature_path)
        low_freq_unfiltered = feature_no_filter.get_low_frequency_peptides()

        # With filter
        builder = SQLFilterBuilder(remove_contaminants=True)
        feature_filtered = Feature(feature_path, filter_builder=builder)
        low_freq_filtered = feature_filtered.get_low_frequency_peptides()

        # Both should return tuples
        assert isinstance(low_freq_unfiltered, tuple)
        assert isinstance(low_freq_filtered, tuple)

    def test_get_median_map_to_condition_with_filter(self, feature_path):
        """Test that get_median_map_to_condition uses filter_builder."""
        builder = SQLFilterBuilder(remove_contaminants=True)
        feature = Feature(feature_path, filter_builder=builder)

        med_map = feature.get_median_map_to_condition()

        # Should return dict of dicts
        assert isinstance(med_map, dict)
        for condition, samples in med_map.items():
            assert isinstance(samples, dict)


class TestPeptideNormalizationWithFilterConfig:
    """Tests for peptide_normalization with filter configuration."""

    def test_normalization_with_filter_config(self):
        """Test peptide normalization with PreprocessingFilterConfig."""
        from mokume.model.filters import PreprocessingFilterConfig

        # Create a filter config
        filter_config = PreprocessingFilterConfig(
            name="test_config",
            enabled=True,
        )
        # Set contaminant removal
        filter_config.protein.remove_contaminants = True
        filter_config.protein.remove_decoys = True

        args = {
            "parquet": str(TESTS_DIR / "example/feature.parquet"),
            "sdrf": str(TESTS_DIR / "example/PXD017834-TMT.sdrf.tsv"),
            "min_aa": 7,
            "min_unique": 2,
            "remove_ids": None,
            "remove_decoy_contaminants": True,
            "remove_low_frequency_peptides": False,
            "output": str(TESTS_DIR / "example" / "out" / "PXD017834-filtered-norm.csv"),
            "skip_normalization": False,
            "nmethod": "median",
            "pnmethod": "globalMedian",
            "log2": True,
            "save_parquet": False,
            "filter_config": filter_config,
        }

        out = Path(args["output"])
        if out.exists():
            out.unlink()

        # Should complete without errors
        peptide_normalization(**args)

        # Clean up
        if out.exists():
            out.unlink()
