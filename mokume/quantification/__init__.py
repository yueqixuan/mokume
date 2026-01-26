"""
Protein quantification methods for the mokume package.

This module provides implementations for various protein quantification
methods including iBAQ, Top3, TopN, MaxLFQ, and AllPeptides.
"""

from mokume.quantification.base import ProteinQuantificationMethod
from mokume.quantification.ibaq import (
    peptides_to_protein,
    normalize_ibaq,
    extract_fasta,
    PeptideProteinMapper,
    ConcentrationWeightByProteomicRuler,
)
from mokume.quantification.top3 import Top3Quantification
from mokume.quantification.topn import TopNQuantification
from mokume.quantification.maxlfq import MaxLFQQuantification
from mokume.quantification.all_peptides import AllPeptidesQuantification

__all__ = [
    # Base class
    "ProteinQuantificationMethod",
    # iBAQ
    "peptides_to_protein",
    "normalize_ibaq",
    "extract_fasta",
    "PeptideProteinMapper",
    "ConcentrationWeightByProteomicRuler",
    # Other quantification methods
    "Top3Quantification",
    "TopNQuantification",
    "MaxLFQQuantification",
    "AllPeptidesQuantification",
]


def get_quantification_method(method: str, **kwargs) -> ProteinQuantificationMethod:
    """
    Get a quantification method instance by name.

    Parameters
    ----------
    method : str
        Name of the quantification method. One of:
        'top3', 'topn', 'maxlfq', 'all', 'sum'.
    **kwargs
        Additional arguments passed to the quantification method constructor.

    Returns
    -------
    ProteinQuantificationMethod
        An instance of the requested quantification method.

    Raises
    ------
    ValueError
        If the method name is not recognized.
    """
    method_lower = method.lower()

    if method_lower == "top3":
        return Top3Quantification()
    elif method_lower == "topn":
        n = kwargs.get("n", 3)
        return TopNQuantification(n=n)
    elif method_lower == "maxlfq":
        min_peptides = kwargs.get("min_peptides", 2)
        return MaxLFQQuantification(min_peptides=min_peptides)
    elif method_lower in ("all", "sum", "allpeptides"):
        return AllPeptidesQuantification()
    else:
        raise ValueError(
            f"Unknown quantification method: {method}. "
            f"Available methods: top3, topn, maxlfq, all"
        )
