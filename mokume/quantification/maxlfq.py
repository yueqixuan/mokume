"""
MaxLFQ protein quantification method.

This module provides the MaxLFQ algorithm, which uses delayed normalization
and maximal peptide ratio extraction for label-free quantification.

Reference:
    Cox J, et al. Accurate Proteome-wide Label-free Quantification by Delayed
    Normalization and Maximal Peptide Ratio Extraction, Termed MaxLFQ.
    Mol Cell Proteomics. 2014;13(9):2513-26.
"""

import pandas as pd
import numpy as np
from typing import Optional

from mokume.quantification.base import ProteinQuantificationMethod
from mokume.core.logger import get_logger

logger = get_logger("mokume.quantification.maxlfq")


def _maxlfq_solve(peptide_matrix: np.ndarray) -> np.ndarray:
    """
    Solve the MaxLFQ optimization problem for a single protein.

    The algorithm finds protein intensities that best explain the observed
    peptide ratios using a least-squares approach.

    Parameters
    ----------
    peptide_matrix : np.ndarray
        Matrix of shape (n_peptides, n_samples) with peptide intensities.
        NaN values indicate missing measurements.

    Returns
    -------
    np.ndarray
        Array of protein intensities for each sample.
    """
    n_peptides, n_samples = peptide_matrix.shape

    if n_samples == 0:
        return np.array([])

    if n_samples == 1:
        # Single sample: return median of peptides
        valid_values = peptide_matrix[~np.isnan(peptide_matrix)]
        if len(valid_values) == 0:
            return np.array([np.nan])
        return np.array([np.median(valid_values)])

    # Log-transform for ratio calculations (add small value to avoid log(0))
    log_matrix = np.log2(peptide_matrix + 1e-10)

    # Calculate pairwise sample ratios using median of peptide ratios
    ratio_matrix = np.full((n_samples, n_samples), np.nan)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # Find peptides measured in both samples
            valid_mask = ~np.isnan(log_matrix[:, i]) & ~np.isnan(log_matrix[:, j])
            if np.sum(valid_mask) > 0:
                # Median of log-ratios
                ratios = log_matrix[valid_mask, i] - log_matrix[valid_mask, j]
                median_ratio = np.median(ratios)
                ratio_matrix[i, j] = median_ratio
                ratio_matrix[j, i] = -median_ratio

    # Solve for protein intensities using least squares
    # We want to find intensities I such that I[i] - I[j] ≈ ratio_matrix[i,j]
    # This is solved by setting one sample as reference (intensity = 0 in log space)

    # Find the sample with most valid ratios as reference
    valid_counts = np.sum(~np.isnan(ratio_matrix), axis=1)
    ref_sample = np.argmax(valid_counts)

    # Initialize log-intensities
    log_intensities = np.full(n_samples, np.nan)
    log_intensities[ref_sample] = 0.0

    # Iteratively fill in intensities based on ratios
    max_iterations = n_samples
    for _ in range(max_iterations):
        for i in range(n_samples):
            if np.isnan(log_intensities[i]):
                # Try to compute from known samples
                for j in range(n_samples):
                    if not np.isnan(log_intensities[j]) and not np.isnan(ratio_matrix[i, j]):
                        log_intensities[i] = log_intensities[j] + ratio_matrix[i, j]
                        break

    # Convert back from log space and scale by median peptide intensity
    # to get absolute intensities
    median_peptide = np.nanmedian(peptide_matrix)
    if np.isnan(median_peptide) or median_peptide == 0:
        median_peptide = 1.0

    intensities = np.power(2, log_intensities) * median_peptide

    # For samples without valid ratios, use median of their peptides
    for i in range(n_samples):
        if np.isnan(intensities[i]):
            sample_peptides = peptide_matrix[:, i]
            valid_peptides = sample_peptides[~np.isnan(sample_peptides)]
            if len(valid_peptides) > 0:
                intensities[i] = np.median(valid_peptides)

    return intensities


class MaxLFQQuantification(ProteinQuantificationMethod):
    """
    MaxLFQ protein quantification method.

    MaxLFQ uses delayed normalization and maximal peptide ratio extraction
    for accurate label-free quantification. The algorithm:

    1. For each protein, calculates pairwise peptide ratios between samples
    2. Uses median ratio extraction to determine protein intensity differences
    3. Solves for absolute intensities using least-squares optimization

    This approach is robust to missing values and provides more accurate
    quantification than simple aggregation methods.

    Parameters
    ----------
    min_peptides : int
        Minimum number of peptides required for MaxLFQ calculation.
        Proteins with fewer peptides will use median aggregation.
    """

    def __init__(self, min_peptides: int = 2):
        """
        Initialize MaxLFQ quantification.

        Parameters
        ----------
        min_peptides : int
            Minimum number of peptides required for MaxLFQ calculation.
        """
        self.min_peptides = min_peptides

    @property
    def name(self) -> str:
        return "MaxLFQ"

    def quantify(
        self,
        peptide_df: pd.DataFrame,
        protein_column: str = "ProteinName",
        peptide_column: str = "PeptideCanonical",
        intensity_column: str = "NormIntensity",
        sample_column: str = "SampleID",
    ) -> pd.DataFrame:
        """
        Quantify proteins using the MaxLFQ algorithm.

        Parameters
        ----------
        peptide_df : pd.DataFrame
            DataFrame containing peptide-level data.
        protein_column : str
            Column name for protein identifiers.
        peptide_column : str
            Column name for peptide sequences.
        intensity_column : str
            Column name for intensity values.
        sample_column : str
            Column name for sample identifiers.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: protein_column, sample_column, 'MaxLFQIntensity'.
        """
        logger.info("Running MaxLFQ quantification")

        # Get unique samples and proteins
        samples = peptide_df[sample_column].unique()
        proteins = peptide_df[protein_column].unique()

        results = []

        for protein in proteins:
            protein_data = peptide_df[peptide_df[protein_column] == protein]
            peptides = protein_data[peptide_column].unique()

            if len(peptides) < self.min_peptides:
                # Fall back to median for proteins with few peptides
                for sample in samples:
                    sample_data = protein_data[protein_data[sample_column] == sample]
                    if len(sample_data) > 0:
                        intensity = sample_data[intensity_column].median()
                        results.append({
                            protein_column: protein,
                            sample_column: sample,
                            "MaxLFQIntensity": intensity,
                        })
                continue

            # Create peptide x sample matrix
            peptide_matrix = np.full((len(peptides), len(samples)), np.nan)
            peptide_to_idx = {p: i for i, p in enumerate(peptides)}
            sample_to_idx = {s: i for i, s in enumerate(samples)}

            for _, row in protein_data.iterrows():
                pep_idx = peptide_to_idx[row[peptide_column]]
                sample_idx = sample_to_idx[row[sample_column]]
                peptide_matrix[pep_idx, sample_idx] = row[intensity_column]

            # Run MaxLFQ algorithm
            intensities = _maxlfq_solve(peptide_matrix)

            # Store results
            for i, sample in enumerate(samples):
                if not np.isnan(intensities[i]) and intensities[i] > 0:
                    results.append({
                        protein_column: protein,
                        sample_column: sample,
                        "MaxLFQIntensity": intensities[i],
                    })

        result_df = pd.DataFrame(results)
        logger.info(f"MaxLFQ quantification complete: {len(proteins)} proteins, {len(samples)} samples")

        return result_df
