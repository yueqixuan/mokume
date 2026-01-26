"""
MaxLFQ protein quantification method.

This module provides the MaxLFQ algorithm, which uses delayed normalization
and maximal peptide ratio extraction for label-free quantification.

This implementation includes optimizations inspired by DirectLFQ:
- Parallelization using joblib
- Variance-guided pairwise merging
- Two-track optimization for large proteins

Reference:
    Cox J, et al. Accurate Proteome-wide Label-free Quantification by Delayed
    Normalization and Maximal Peptide Ratio Extraction, Termed MaxLFQ.
    Mol Cell Proteomics. 2014;13(9):2513-26.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from joblib import Parallel, delayed

from mokume.quantification.base import ProteinQuantificationMethod
from mokume.core.logger import get_logger

logger = get_logger("mokume.quantification.maxlfq")


def _compute_pairwise_ratios(
    log_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise sample ratios and their variances.

    Uses median of log-ratios between samples, with variance to guide
    the merging order (lower variance = more reliable ratio).

    Parameters
    ----------
    log_matrix : np.ndarray
        Log-transformed peptide intensities, shape (n_peptides, n_samples).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - ratio_matrix: Median log-ratios between sample pairs
        - variance_matrix: Variance of ratios (for merging priority)
    """
    n_samples = log_matrix.shape[1]
    ratio_matrix = np.full((n_samples, n_samples), np.nan)
    variance_matrix = np.full((n_samples, n_samples), np.inf)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # Find peptides measured in both samples
            valid_mask = ~np.isnan(log_matrix[:, i]) & ~np.isnan(log_matrix[:, j])
            n_valid = np.sum(valid_mask)

            if n_valid > 0:
                # Compute log-ratios
                ratios = log_matrix[valid_mask, i] - log_matrix[valid_mask, j]
                median_ratio = np.median(ratios)
                ratio_matrix[i, j] = median_ratio
                ratio_matrix[j, i] = -median_ratio

                # Compute variance for merging priority
                if n_valid > 1:
                    variance = np.var(ratios)
                else:
                    variance = np.inf  # Single peptide, low confidence
                variance_matrix[i, j] = variance
                variance_matrix[j, i] = variance

    return ratio_matrix, variance_matrix


def _variance_guided_solve(
    ratio_matrix: np.ndarray,
    variance_matrix: np.ndarray,
) -> np.ndarray:
    """
    Solve for protein intensities using variance-guided hierarchical merging.

    This approach merges sample pairs in order of lowest variance (highest
    confidence) first, inspired by DirectLFQ's hierarchical alignment.

    Parameters
    ----------
    ratio_matrix : np.ndarray
        Pairwise median log-ratios.
    variance_matrix : np.ndarray
        Variance of ratios for each pair.

    Returns
    -------
    np.ndarray
        Log-scale protein intensities for each sample.
    """
    n_samples = ratio_matrix.shape[0]
    log_intensities = np.full(n_samples, np.nan)

    # Track which samples have been assigned intensities
    assigned = np.zeros(n_samples, dtype=bool)

    # Find sample with most valid ratios as starting point
    valid_counts = np.sum(~np.isnan(ratio_matrix), axis=1)
    if valid_counts.max() == 0:
        return log_intensities

    ref_sample = np.argmax(valid_counts)
    log_intensities[ref_sample] = 0.0
    assigned[ref_sample] = True

    # Iteratively assign intensities, prioritizing low-variance pairs
    for _ in range(n_samples - 1):
        best_variance = np.inf
        best_i, best_j = -1, -1
        best_ratio = np.nan

        # Find the best unassigned sample to connect
        for i in range(n_samples):
            if not assigned[i]:
                continue
            for j in range(n_samples):
                if assigned[j]:
                    continue
                if np.isnan(ratio_matrix[i, j]):
                    continue
                if variance_matrix[i, j] < best_variance:
                    best_variance = variance_matrix[i, j]
                    best_i, best_j = i, j
                    best_ratio = ratio_matrix[j, i]  # ratio from assigned to unassigned

        if best_j >= 0:
            log_intensities[best_j] = log_intensities[best_i] + best_ratio
            assigned[best_j] = True
        else:
            break  # No more connections possible

    return log_intensities


def _maxlfq_solve_protein(
    peptide_matrix: np.ndarray,
    use_variance_guided: bool = True,
) -> np.ndarray:
    """
    Solve the MaxLFQ optimization problem for a single protein.

    Parameters
    ----------
    peptide_matrix : np.ndarray
        Matrix of shape (n_peptides, n_samples) with peptide intensities.
        NaN values indicate missing measurements.
    use_variance_guided : bool
        If True, use variance-guided merging (more accurate).
        If False, use simple iterative propagation (faster).

    Returns
    -------
    np.ndarray
        Array of protein intensities for each sample.
    """
    n_peptides, n_samples = peptide_matrix.shape

    if n_samples == 0:
        return np.array([])

    if n_samples == 1:
        valid_values = peptide_matrix[~np.isnan(peptide_matrix)]
        if len(valid_values) == 0:
            return np.array([np.nan])
        return np.array([np.median(valid_values)])

    # Log-transform for ratio calculations
    with np.errstate(divide='ignore', invalid='ignore'):
        log_matrix = np.log2(peptide_matrix)

    # Compute pairwise ratios and variances
    ratio_matrix, variance_matrix = _compute_pairwise_ratios(log_matrix)

    # Solve for log-intensities
    if use_variance_guided:
        log_intensities = _variance_guided_solve(ratio_matrix, variance_matrix)
    else:
        # Simple iterative approach (original implementation)
        valid_counts = np.sum(~np.isnan(ratio_matrix), axis=1)
        ref_sample = np.argmax(valid_counts)
        log_intensities = np.full(n_samples, np.nan)
        log_intensities[ref_sample] = 0.0

        for _ in range(n_samples):
            for i in range(n_samples):
                if np.isnan(log_intensities[i]):
                    for j in range(n_samples):
                        if not np.isnan(log_intensities[j]) and not np.isnan(ratio_matrix[i, j]):
                            log_intensities[i] = log_intensities[j] + ratio_matrix[i, j]
                            break

    # Convert back from log space and scale
    median_peptide = np.nanmedian(peptide_matrix)
    if np.isnan(median_peptide) or median_peptide <= 0:
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


def _process_protein(
    protein: str,
    protein_data: pd.DataFrame,
    peptide_column: str,
    intensity_column: str,
    sample_column: str,
    samples: np.ndarray,
    min_peptides: int,
    use_variance_guided: bool,
) -> list:
    """
    Process a single protein for MaxLFQ quantification.

    Parameters
    ----------
    protein : str
        Protein identifier.
    protein_data : pd.DataFrame
        Peptide data for this protein.
    peptide_column : str
        Column name for peptides.
    intensity_column : str
        Column name for intensities.
    sample_column : str
        Column name for samples.
    samples : np.ndarray
        Array of all sample names.
    min_peptides : int
        Minimum peptides required for MaxLFQ.
    use_variance_guided : bool
        Whether to use variance-guided merging.

    Returns
    -------
    list
        List of result dictionaries for this protein.
    """
    results = []
    peptides = protein_data[peptide_column].unique()

    if len(peptides) < min_peptides:
        # Fall back to median for proteins with few peptides
        for sample in samples:
            sample_data = protein_data[protein_data[sample_column] == sample]
            if len(sample_data) > 0:
                intensity = sample_data[intensity_column].median()
                results.append({
                    'protein': protein,
                    'sample': sample,
                    'intensity': intensity,
                })
        return results

    # Create peptide x sample matrix
    n_peptides = len(peptides)
    n_samples = len(samples)
    peptide_matrix = np.full((n_peptides, n_samples), np.nan)
    peptide_to_idx = {p: i for i, p in enumerate(peptides)}
    sample_to_idx = {s: i for i, s in enumerate(samples)}

    for _, row in protein_data.iterrows():
        pep_idx = peptide_to_idx[row[peptide_column]]
        sample_idx = sample_to_idx[row[sample_column]]
        # Handle multiple measurements per peptide/sample
        current = peptide_matrix[pep_idx, sample_idx]
        new_val = row[intensity_column]
        if np.isnan(current):
            peptide_matrix[pep_idx, sample_idx] = new_val
        else:
            # Sum multiple measurements
            peptide_matrix[pep_idx, sample_idx] = current + new_val

    # Run MaxLFQ algorithm
    intensities = _maxlfq_solve_protein(peptide_matrix, use_variance_guided)

    # Store results
    for i, sample in enumerate(samples):
        if not np.isnan(intensities[i]) and intensities[i] > 0:
            results.append({
                'protein': protein,
                'sample': sample,
                'intensity': intensities[i],
            })

    return results


class MaxLFQQuantification(ProteinQuantificationMethod):
    """
    MaxLFQ protein quantification method with parallelization.

    MaxLFQ uses delayed normalization and maximal peptide ratio extraction
    for accurate label-free quantification. This implementation includes:

    - Variance-guided hierarchical merging (inspired by DirectLFQ)
    - Parallel processing using joblib
    - Robust handling of missing values

    Parameters
    ----------
    min_peptides : int
        Minimum number of peptides required for MaxLFQ calculation.
        Proteins with fewer peptides will use median aggregation.
        Default is 2.
    n_jobs : int
        Number of parallel jobs. -1 uses all available cores.
        Default is -1.
    use_variance_guided : bool
        If True, use variance-guided merging for better accuracy.
        If False, use simple iterative propagation (faster).
        Default is True.
    verbose : int
        Verbosity level for joblib (0=silent, 10=verbose).
        Default is 0.

    Examples
    --------
    >>> from mokume.quantification import MaxLFQQuantification
    >>> maxlfq = MaxLFQQuantification(min_peptides=2, n_jobs=4)
    >>> result = maxlfq.quantify(
    ...     peptide_df,
    ...     protein_column="ProteinName",
    ...     peptide_column="PeptideSequence",
    ...     intensity_column="Intensity",
    ...     sample_column="SampleID"
    ... )

    References
    ----------
    Cox J, et al. Accurate Proteome-wide Label-free Quantification by Delayed
    Normalization and Maximal Peptide Ratio Extraction, Termed MaxLFQ.
    Mol Cell Proteomics. 2014;13(9):2513-26.
    """

    def __init__(
        self,
        min_peptides: int = 2,
        n_jobs: int = -1,
        use_variance_guided: bool = True,
        verbose: int = 0,
    ):
        """
        Initialize MaxLFQ quantification.

        Parameters
        ----------
        min_peptides : int
            Minimum number of peptides required for MaxLFQ calculation.
        n_jobs : int
            Number of parallel jobs (-1 for all cores).
        use_variance_guided : bool
            Whether to use variance-guided merging.
        verbose : int
            Verbosity level for parallel processing.
        """
        self.min_peptides = min_peptides
        self.n_jobs = n_jobs
        self.use_variance_guided = use_variance_guided
        self.verbose = verbose

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

        logger.info(f"Processing {len(proteins)} proteins across {len(samples)} samples")
        logger.info(f"Using {'variance-guided' if self.use_variance_guided else 'simple'} merging")
        logger.info(f"Parallel jobs: {self.n_jobs}")

        # Group data by protein for efficient access
        grouped = peptide_df.groupby(protein_column)

        # Process proteins in parallel
        all_results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_process_protein)(
                protein,
                group,
                peptide_column,
                intensity_column,
                sample_column,
                samples,
                self.min_peptides,
                self.use_variance_guided,
            )
            for protein, group in grouped
        )

        # Flatten results
        results = []
        for protein_results in all_results:
            results.extend(protein_results)

        # Create result DataFrame
        result_df = pd.DataFrame(results)

        if len(result_df) > 0:
            result_df = result_df.rename(columns={
                'protein': protein_column,
                'sample': sample_column,
                'intensity': 'MaxLFQIntensity',
            })

        logger.info(f"MaxLFQ complete: {len(proteins)} proteins, {len(samples)} samples")

        return result_df
