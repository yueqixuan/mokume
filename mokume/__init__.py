"""
mokume - A comprehensive proteomics quantification library.

This package provides tools for processing and analyzing proteomics data using
multiple protein quantification methods including iBAQ (intensity-based absolute
quantification), Top3, TopN, and MaxLFQ.
"""

import warnings

# Suppress numpy matrix deprecation warning
warnings.filterwarnings(
    "ignore", category=PendingDeprecationWarning, module="numpy.matrixlib.defmatrix"
)

__version__ = "0.1.0"

# Import logging configuration
from mokume.core.logging_config import initialize_logging

# Initialize logging with default settings
# Users can override these settings by calling initialize_logging with their own settings
initialize_logging()
