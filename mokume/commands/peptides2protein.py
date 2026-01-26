"""
CLI command for computing protein quantification values.
"""

import logging

import click
import pandas as pd

from mokume.quantification.ibaq import peptides_to_protein
from mokume.quantification import (
    get_quantification_method,
    Top3Quantification,
    TopNQuantification,
    MaxLFQQuantification,
    AllPeptidesQuantification,
)
from mokume.model.organism import OrganismDescription
from mokume.core.constants import (
    PROTEIN_NAME,
    SAMPLE_ID,
    CONDITION,
    NORM_INTENSITY,
    PEPTIDE_CANONICAL,
    is_parquet,
)

logger = logging.getLogger(__name__)


QUANTIFICATION_METHODS = ["ibaq", "top3", "topn", "maxlfq", "sum"]


@click.command("peptides2protein", short_help="Compute protein quantification values")
@click.option(
    "-f",
    "--fasta",
    help="Protein database to compute IBAQ values (required for ibaq method)",
    type=click.Path(exists=True),
)
@click.option(
    "-p",
    "--peptides",
    help="Peptide identifications with intensities following the peptide intensity output",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--method",
    help="Quantification method to use",
    type=click.Choice(QUANTIFICATION_METHODS, case_sensitive=False),
    default="ibaq",
)
@click.option(
    "-e",
    "--enzyme",
    help="Enzyme used during the analysis of the dataset (default: Trypsin)",
    default="Trypsin",
)
@click.option(
    "-n",
    "--normalize",
    help="Normalize quantification values",
    is_flag=True,
)
@click.option("--min_aa", help="Minimum number of amino acids to consider a peptide", default=7)
@click.option("--max_aa", help="Maximum number of amino acids to consider a peptide", default=30)
@click.option("-t", "--tpa", help="Whether calculate TPA (iBAQ method only)", is_flag=True)
@click.option("-r", "--ruler", help="Whether to use ProteomicRuler (iBAQ method only)", is_flag=True)
@click.option("-i", "--ploidy", help="Ploidy number (default: 2)", default=2)
@click.option(
    "-m",
    "--organism",
    help="Organism source of the data (default: human)",
    type=click.Choice(
        sorted(map(str.lower, OrganismDescription.registered_organisms())), case_sensitive=False
    ),
    default="human",
)
@click.option(
    "-c", "--cpc", help="Cellular protein concentration(g/L) (default: 200)", default=200
)
@click.option("-o", "--output", help="Output file with the proteins and quantification values")
@click.option(
    "--verbose",
    help="Print additional information about the distributions of the intensities",
    is_flag=True,
)
@click.option(
    "--qc_report",
    help="PDF file to store multiple QC images (iBAQ method only)",
    default="QCprofile.pdf",
)
@click.option(
    "--topn_n",
    help="Number of top peptides to use for TopN method (default: 3)",
    default=3,
    type=int,
)
@click.pass_context
def peptides2protein(
    click_context,
    fasta: str,
    peptides: str,
    method: str,
    enzyme: str,
    normalize: bool,
    min_aa: int,
    max_aa: int,
    tpa: bool,
    ruler: bool,
    organism: str,
    ploidy: int,
    cpc: float,
    output: str,
    verbose: bool,
    qc_report: str,
    topn_n: int,
) -> None:
    """
    Compute protein quantification values from peptide intensity data.

    This command processes peptide identifications and computes protein
    quantification values using various methods:

    \b
    - ibaq: Intensity-Based Absolute Quantification (default)
    - top3: Average of the 3 most intense peptides
    - topn: Average of the N most intense peptides (use --topn_n)
    - maxlfq: MaxLFQ delayed normalization algorithm
    - sum: Sum of all peptide intensities

    For the iBAQ method, a FASTA file is required. Other methods can work
    without a FASTA file.
    """
    method_lower = method.lower()

    if method_lower == "ibaq":
        # Use the existing iBAQ implementation
        if not fasta:
            raise click.UsageError("The --fasta option is required for the iBAQ method")

        peptides_to_protein(
            fasta=fasta,
            peptides=peptides,
            enzyme=enzyme,
            normalize=normalize,
            min_aa=min_aa,
            max_aa=max_aa,
            tpa=tpa,
            ruler=ruler,
            ploidy=ploidy,
            cpc=cpc,
            organism=organism,
            output=output,
            verbose=verbose,
            qc_report=qc_report,
        )
    else:
        # Use the generic quantification methods
        logger.info(f"Using {method} quantification method")

        # Load peptide data
        if is_parquet(peptides):
            peptide_df = pd.read_parquet(peptides)
        else:
            peptide_df = pd.read_csv(peptides)

        # Get the quantification method
        if method_lower == "topn":
            quant_method = get_quantification_method(method, n=topn_n)
        else:
            quant_method = get_quantification_method(method)

        # Determine column names (try to auto-detect)
        protein_col = PROTEIN_NAME if PROTEIN_NAME in peptide_df.columns else "ProteinName"
        sample_col = SAMPLE_ID if SAMPLE_ID in peptide_df.columns else "SampleID"
        intensity_col = NORM_INTENSITY if NORM_INTENSITY in peptide_df.columns else "NormIntensity"
        peptide_col = PEPTIDE_CANONICAL if PEPTIDE_CANONICAL in peptide_df.columns else "PeptideSequence"

        # Check for required columns
        for col, name in [(protein_col, "protein"), (sample_col, "sample"), (intensity_col, "intensity")]:
            if col not in peptide_df.columns:
                raise click.UsageError(f"Could not find {name} column '{col}' in peptide file")

        # Run quantification
        result_df = quant_method.quantify(
            peptide_df,
            protein_column=protein_col,
            peptide_column=peptide_col,
            intensity_column=intensity_col,
            sample_column=sample_col,
        )

        # Add condition if available
        if CONDITION in peptide_df.columns:
            condition_map = peptide_df[[sample_col, CONDITION]].drop_duplicates().set_index(sample_col)[CONDITION]
            result_df[CONDITION] = result_df[sample_col].map(condition_map)

        # Normalize if requested
        if normalize:
            intensity_col_out = result_df.columns[-1]  # The quantification output column
            result_df[f"{intensity_col_out}Norm"] = result_df.groupby(sample_col)[intensity_col_out].transform(
                lambda x: x / x.sum()
            )

        # Save output
        if output:
            if output.endswith(".parquet"):
                result_df.to_parquet(output, index=False)
            else:
                result_df.to_csv(output, sep="\t", index=False)
            logger.info(f"Results saved to {output}")
        else:
            print(result_df.to_string())
