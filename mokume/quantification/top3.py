"""
Top3 protein quantification method.

This module provides the Top3 quantification method, which calculates
protein abundance as the average of the three most intense peptides.
"""

from typing import Optional

import pandas as pd

from mokume.quantification.base import ProteinQuantificationMethod
from mokume.core.constants import (
    PROTEIN_NAME,
    PEPTIDE_CANONICAL,
    NORM_INTENSITY,
    SAMPLE_ID,
)


class Top3Quantification(ProteinQuantificationMethod):
    """
    Top3 protein quantification method.

    Calculates protein abundance as the average of the three most intense
    peptides for each protein in each sample (or run if run_column is provided).
    """

    @property
    def name(self) -> str:
        return "Top3"

    def quantify(
        self,
        peptide_df: pd.DataFrame,
        protein_column: str = PROTEIN_NAME,
        peptide_column: str = PEPTIDE_CANONICAL,
        intensity_column: str = NORM_INTENSITY,
        sample_column: str = SAMPLE_ID,
        run_column: Optional[str] = None,
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
        run_column : str, optional
            Column name for run identifiers. If provided, quantification
            is performed at the run level instead of sample level.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: protein_column, sample_column,
            (run_column if provided), 'Top3Intensity'.
        """
        # Determine grouping columns based on aggregation level
        if run_column is not None and run_column in peptide_df.columns:
            group_cols = [protein_column, sample_column, run_column]
        else:
            group_cols = [protein_column, sample_column]

        # Sort by intensity and take top 3 per protein per group
        result = (
            peptide_df.sort_values(intensity_column, ascending=False)
            .groupby(group_cols)
            .head(3)
            .groupby(group_cols)[intensity_column]
            .mean()
            .reset_index()
        )

        # Rename intensity column
        result = result.rename(columns={intensity_column: "Top3Intensity"})
        return result
