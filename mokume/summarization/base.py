"""
Base class for summarization strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd


class SummarizationStrategy(ABC):
    """Abstract base class for intensity summarization strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the summarization strategy."""
        pass

    @abstractmethod
    def aggregate(self, values: pd.Series) -> float:
        """Aggregate a series of values."""
        pass
