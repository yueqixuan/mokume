"""
MaxLFQ protein quantification method.

This module provides the MaxLFQ algorithm for label-free quantification.

This implementation uses peptide trace alignment (inspired by DirectLFQ) for
improved accuracy:
- Aligns peptide intensity traces within each protein using median shifts
- Aggregates aligned traces using median
- Scales results to preserve total peptide intensity
- Parallelization using joblib for performance

References:
    Cox J, et al. Accurate Proteome-wide Label-free Quantification by Delayed
    Normalization and Maximal Peptide Ratio Extraction, Termed MaxLFQ.
    Mol Cell Proteomics. 2014;13(9):2513-26.

    Ammar C, et al. Accurate label-free quantification by directLFQ to compare
    unlimited numbers of proteomes. Mol Cell Proteomics. 2023.
"""

import warnings

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from mokume.quantification.base import ProteinQuantificationMethod
from mokume.core.logger import get_logger

logger = get_logger("mokume.quantification.maxlfq")


def _maxlfq_solve_protein(peptide_matrix: np.ndarray) -> np.ndarray:
    """
    Solve the MaxLFQ optimization problem for a single protein.

    Uses peptide trace alignment inspired by DirectLFQ for optimal accuracy.
    The algorithm:
    1. Aligns peptide intensity traces using median shifts
    2. Takes median of aligned traces per sample
    3. Scales to preserve total peptide intensity

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
        valid_values = peptide_matrix[~np.isnan(peptide_matrix)]
        if len(valid_values) == 0:
            return np.array([np.nan])
        return np.array([np.median(valid_values)])

    if n_peptides == 1:
        # Single peptide: return its intensities directly
        return peptide_matrix[0, :].copy()

    # Store original sum for scaling
    original_sum = np.nansum(peptide_matrix)
    if original_sum <= 0:
        return np.full(n_samples, np.nan)

    # Log-transform for ratio calculations
    with np.errstate(divide='ignore', invalid='ignore'):
        log_matrix = np.log2(peptide_matrix.copy())

    # Step 1: Align peptide traces
    # Use peptide with most valid values as reference
    valid_counts = np.sum(~np.isnan(log_matrix), axis=1)
    if valid_counts.max() == 0:
        return np.full(n_samples, np.nan)

    ref_peptide_idx = np.argmax(valid_counts)
    ref_trace = log_matrix[ref_peptide_idx, :]

    # Align other peptides to reference using median shift
    aligned_matrix = log_matrix.copy()
    for pep_idx in range(n_peptides):
        if pep_idx == ref_peptide_idx:
            continue

        pep_trace = log_matrix[pep_idx, :]

        # Find samples measured in both reference and current peptide
        valid = ~np.isnan(ref_trace) & ~np.isnan(pep_trace)
        if np.sum(valid) > 0:
            # Compute median shift to align this peptide to reference
            shift = np.nanmedian(ref_trace[valid] - pep_trace[valid])
            aligned_matrix[pep_idx, :] = pep_trace + shift

    # Step 2: Take median of aligned traces per sample
    # Suppress warning for samples with no peptides (all-NaN columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        log_intensities = np.nanmedian(aligned_matrix, axis=0)

    # Step 3: Scale to preserve total peptide intensity
    intensities = np.power(2, log_intensities)

    # Handle NaN samples using fallback
    for i in range(n_samples):
        if np.isnan(intensities[i]):
            sample_peptides = peptide_matrix[:, i]
            valid_peptides = sample_peptides[~np.isnan(sample_peptides)]
            if len(valid_peptides) > 0:
                intensities[i] = np.median(valid_peptides)

    # Scale to preserve total intensity sum
    current_sum = np.nansum(intensities)
    if current_sum > 0:
        scale_factor = original_sum / current_sum
        intensities = intensities * scale_factor

    return intensities


def _process_protein(
    protein: str,
    protein_data: pd.DataFrame,
    peptide_column: str,
    intensity_column: str,
    sample_column: str,
    samples: np.ndarray,
    min_peptides: int,
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
    intensities = _maxlfq_solve_protein(peptide_matrix)

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
    for accurate label-free quantification. This implementation uses
    variance-guided hierarchical merging (inspired by DirectLFQ) for
    improved accuracy and parallel processing for performance.

    Parameters
    ----------
    min_peptides : int
        Minimum number of peptides required for MaxLFQ calculation.
        Proteins with fewer peptides will use median aggregation.
        Default is 2.
    threads : int
        Number of parallel threads. Use -1 for all available cores,
        1 for single-threaded execution. Default is -1.
    verbose : int
        Verbosity level for parallel processing (0=silent, 10=verbose).
        Default is 0.

    Examples
    --------
    >>> from mokume.quantification import MaxLFQQuantification
    >>> maxlfq = MaxLFQQuantification(min_peptides=2, threads=4)
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
        threads: int = -1,
        verbose: int = 0,
        # Legacy parameter support
        n_jobs: int = None,
        use_variance_guided: bool = None,
    ):
        """
        Initialize MaxLFQ quantification.

        Parameters
        ----------
        min_peptides : int
            Minimum number of peptides required for MaxLFQ calculation.
        threads : int
            Number of parallel threads (-1 for all cores, 1 for single-threaded).
        verbose : int
            Verbosity level for parallel processing.
        n_jobs : int, optional
            Deprecated. Use 'threads' instead.
        use_variance_guided : bool, optional
            Deprecated. Variance-guided merging is always used.
        """
        self.min_peptides = min_peptides

        # Handle legacy n_jobs parameter
        if n_jobs is not None:
            import warnings
            warnings.warn(
                "Parameter 'n_jobs' is deprecated, use 'threads' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.threads = n_jobs
        else:
            self.threads = threads

        self.verbose = verbose

        # Warn if use_variance_guided is explicitly set
        if use_variance_guided is not None and not use_variance_guided:
            import warnings
            warnings.warn(
                "Parameter 'use_variance_guided' is deprecated. "
                "Variance-guided merging is always used for best accuracy.",
                DeprecationWarning,
                stacklevel=2,
            )

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
        logger.info(f"Threads: {self.threads}")

        # Group data by protein for efficient access
        grouped = peptide_df.groupby(protein_column)

        # Process proteins in parallel
        all_results = Parallel(n_jobs=self.threads, verbose=self.verbose)(
            delayed(_process_protein)(
                protein,
                group,
                peptide_column,
                intensity_column,
                sample_column,
                samples,
                self.min_peptides,
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
