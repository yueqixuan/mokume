"""
All peptides protein quantification method.

This module provides a quantification method that sums all peptide
intensities for each protein.
"""

import pandas as pd

from mokume.quantification.base import ProteinQuantificationMethod


class AllPeptidesQuantification(ProteinQuantificationMethod):
    """
    All peptides protein quantification method.

    Calculates protein abundance as the sum of all peptide intensities
    for each protein in each sample.
    """

    @property
    def name(self) -> str:
        return "AllPeptides"

    def quantify(
        self,
        peptide_df: pd.DataFrame,
        protein_column: str = "ProteinName",
        peptide_column: str = "PeptideCanonical",
        intensity_column: str = "NormIntensity",
        sample_column: str = "SampleID",
    ) -> pd.DataFrame:
        """
        Quantify proteins using sum of all peptide intensities.

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
            DataFrame with columns: protein_column, sample_column, 'SumIntensity'.
        """
        result = (
            peptide_df.groupby([protein_column, sample_column])[intensity_column]
            .sum()
            .reset_index()
        )
        result.columns = [protein_column, sample_column, "SumIntensity"]
        return result
