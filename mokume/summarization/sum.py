"""Sum summarization strategy."""

import pandas as pd
from mokume.summarization.base import SummarizationStrategy


class SumSummarization(SummarizationStrategy):
    """Use sum of peptide intensities."""

    @property
    def name(self) -> str:
        return "Sum"

    def aggregate(self, values: pd.Series) -> float:
        return values.sum()
