"""
Normalization implementations for the mokume package.

This module provides implementations for feature-level, peptide-level,
protein-level, and hierarchical sample normalization.
"""

from mokume.normalization.feature import (
    normalize_runs,
    normalize_sample,
    normalize_replicates,
)
from mokume.normalization.peptide import (
    peptide_normalization,
    analyse_sdrf,
    remove_contaminants_entrapments_decoys,
    remove_protein_by_ids,
    Feature,
)
from mokume.normalization.hierarchical import (
    HierarchicalSampleNormalizer,
    HierarchicalIonAligner,
    DistanceMetric,
)

__all__ = [
    # Feature normalization
    "normalize_runs",
    "normalize_sample",
    "normalize_replicates",
    # Peptide normalization
    "peptide_normalization",
    "analyse_sdrf",
    "remove_contaminants_entrapments_decoys",
    "remove_protein_by_ids",
    "Feature",
    # Hierarchical normalization (DirectLFQ-style, native mokume)
    "HierarchicalSampleNormalizer",
    "HierarchicalIonAligner",
    "DistanceMetric",
]
