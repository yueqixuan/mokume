"""
TopN protein quantification method.

This module provides the TopN quantification method, which calculates
protein abundance as the average of the N most intense peptides.
"""

import pandas as pd

from mokume.quantification.base import ProteinQuantificationMethod


class TopNQuantification(ProteinQuantificationMethod):
    """
    TopN protein quantification method.

    Calculates protein abundance as the average of the N most intense
    peptides for each protein in each sample.
    """

    def __init__(self, n: int = 3):
        """
        Initialize TopN quantification.

        Parameters
        ----------
        n : int
            Number of top peptides to use for quantification.
        """
        self.n = n

    @property
    def name(self) -> str:
        return f"Top{self.n}"

    def quantify(
        self,
        peptide_df: pd.DataFrame,
        protein_column: str = "ProteinName",
        peptide_column: str = "PeptideCanonical",
        intensity_column: str = "NormIntensity",
        sample_column: str = "SampleID",
    ) -> pd.DataFrame:
        """
        Quantify proteins using the TopN method.

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
            DataFrame with columns: protein_column, sample_column, 'TopNIntensity'.
        """
        # Sort by intensity and take top N per protein per sample
        result = (
            peptide_df.sort_values(intensity_column, ascending=False)
            .groupby([protein_column, sample_column])
            .head(self.n)
            .groupby([protein_column, sample_column])[intensity_column]
            .mean()
            .reset_index()
        )
        result.columns = [protein_column, sample_column, f"Top{self.n}Intensity"]
        return result
