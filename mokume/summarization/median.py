"""Median summarization strategy."""

import pandas as pd
from mokume.summarization.base import SummarizationStrategy


class MedianSummarization(SummarizationStrategy):
    """Use median of peptide intensities."""

    @property
    def name(self) -> str:
        return "Median"

    def aggregate(self, values: pd.Series) -> float:
        return values.median()
