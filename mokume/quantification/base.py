"""
Base class for protein quantification methods.

This module provides an abstract base class that all quantification methods
should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class ProteinQuantificationMethod(ABC):
    """
    Abstract base class for protein quantification methods.

    All quantification methods (iBAQ, Top3, TopN, MaxLFQ, etc.) should
    inherit from this class and implement the required methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the quantification method."""
        pass

    @abstractmethod
    def quantify(
        self,
        peptide_df: pd.DataFrame,
        protein_column: str = "ProteinName",
        peptide_column: str = "PeptideCanonical",
        intensity_column: str = "NormIntensity",
        sample_column: str = "SampleID",
    ) -> pd.DataFrame:
        """
        Quantify proteins from peptide intensities.

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
            DataFrame containing protein-level quantification values.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
