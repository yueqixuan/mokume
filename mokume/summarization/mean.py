"""Mean summarization strategy."""

import pandas as pd
from mokume.summarization.base import SummarizationStrategy


class MeanSummarization(SummarizationStrategy):
    """Use mean of peptide intensities."""

    @property
    def name(self) -> str:
        return "Mean"

    def aggregate(self, values: pd.Series) -> float:
        return values.mean()
