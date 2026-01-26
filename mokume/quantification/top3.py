"""
Top3 protein quantification method.

This module provides the Top3 quantification method, which calculates
protein abundance as the average of the three most intense peptides.
"""

import pandas as pd

from mokume.quantification.base import ProteinQuantificationMethod


class Top3Quantification(ProteinQuantificationMethod):
    """
    Top3 protein quantification method.

    Calculates protein abundance as the average of the three most intense
    peptides for each protein in each sample.
    """

    @property
    def name(self) -> str:
        return "Top3"

    def quantify(
        self,
        peptide_df: pd.DataFrame,
        protein_column: str = "ProteinName",
        peptide_column: str = "PeptideCanonical",
        intensity_column: str = "NormIntensity",
        sample_column: str = "SampleID",
    ) -> pd.DataFrame:
        """
        Quantify proteins using the Top3 method.

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
            DataFrame with columns: protein_column, sample_column, 'Top3Intensity'.
        """
        # Sort by intensity and take top 3 per protein per sample
        result = (
            peptide_df.sort_values(intensity_column, ascending=False)
            .groupby([protein_column, sample_column])
            .head(3)
            .groupby([protein_column, sample_column])[intensity_column]
            .mean()
            .reset_index()
        )
        result.columns = [protein_column, sample_column, "Top3Intensity"]
        return result
