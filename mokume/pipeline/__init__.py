"""
Unified pipelines for proteomics data processing.

This module provides high-level pipelines that combine multiple processing
steps into single, easy-to-use functions.
"""

from mokume.pipeline.features_to_proteins import (
    PipelineConfig,
    QuantificationPipeline,
    features_to_proteins,
)

__all__ = [
    "PipelineConfig",
    "QuantificationPipeline",
    "features_to_proteins",
]
