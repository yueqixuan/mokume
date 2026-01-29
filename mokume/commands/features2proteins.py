"""
CLI command for unified features to proteins pipeline.

This command provides a single-step workflow from feature-level parquet
files to protein intensities, supporting multiple quantification methods.
"""

import click

from mokume.model.normalization import FeatureNormalizationMethod, PeptideNormalizationMethod


# Build choices for sample normalization (including hierarchical)
SAMPLE_NORM_CHOICES = [p.name.lower() for p in PeptideNormalizationMethod]


@click.command("features2proteins", short_help="Quantify proteins from feature parquet file.")
@click.option(
    "-p",
    "--parquet",
    help="Parquet file (quantms.io/qpx format)",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output",
    help="Output file for protein intensities",
    required=True,
    type=click.Path(),
)
@click.option(
    "-s",
    "--sdrf",
    help="SDRF file for sample metadata",
    default=None,
    type=click.Path(exists=True),
)
@click.option(
    "--quant-method",
    "quant_method",
    help="Quantification method: directlfq, ibaq, maxlfq, topn, sum, median",
    type=click.Choice(
        ["directlfq", "ibaq", "maxlfq", "topn", "sum", "median"],
        case_sensitive=False,
    ),
    default="maxlfq",
    show_default=True,
)
@click.option(
    "--topn",
    "topn_peptides",
    help="Number of top peptides for TopN quantification (used when --quant-method=topn)",
    type=int,
    default=3,
    show_default=True,
)
# Filtering options
@click.option(
    "--min-aa",
    "min_aa",
    help="Minimum number of amino acids for peptides",
    type=int,
    default=7,
    show_default=True,
)
@click.option(
    "--min-unique",
    "min_unique",
    help="Minimum number of unique peptides per protein",
    type=int,
    default=2,
    show_default=True,
)
@click.option(
    "--remove-contaminants/--keep-contaminants",
    "remove_contaminants",
    help="Remove contaminants and decoys",
    default=True,
    show_default=True,
)
# Normalization options (ignored for directlfq)
@click.option(
    "--run-normalization",
    "run_normalization",
    help="Run/technical replicate normalization (ignored for directlfq)",
    type=click.Choice([f.name.lower() for f in FeatureNormalizationMethod], case_sensitive=False),
    default="median",
    show_default=True,
)
@click.option(
    "--sample-normalization",
    "sample_normalization",
    help="Sample normalization method (ignored for directlfq). "
         "Use 'hierarchical' for DirectLFQ-style clustering-based normalization.",
    type=click.Choice(SAMPLE_NORM_CHOICES, case_sensitive=False),
    default="globalmedian",
    show_default=True,
)
@click.option(
    "--normalization-proteins",
    "normalization_proteins",
    help="File with protein IDs to use for normalization (one per line)",
    type=click.Path(exists=True),
    default=None,
)
# Method-specific options
@click.option(
    "--fasta",
    "fasta_file",
    help="FASTA file (required for iBAQ)",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--ion-alignment",
    "ion_alignment",
    help="Ion alignment method for MaxLFQ",
    type=click.Choice(["none", "hierarchical"], case_sensitive=False),
    default=None,
)
# DirectLFQ-specific options
@click.option(
    "--directlfq-cores",
    "directlfq_cores",
    help="Number of CPU cores for DirectLFQ",
    type=int,
    default=None,
)
@click.option(
    "--directlfq-min-nonan",
    "directlfq_min_nonan",
    help="Minimum non-NaN values for DirectLFQ",
    type=int,
    default=1,
    show_default=True,
)
# Optional exports
@click.option(
    "--export-peptides",
    "export_peptides",
    help="Export normalized peptides to this file (for debugging/analysis)",
    type=click.Path(),
    default=None,
)
@click.option(
    "--export-ions",
    "export_ions",
    help="Export normalized ions to this file (DirectLFQ only)",
    type=click.Path(),
    default=None,
)
@click.pass_context
def features2proteins(
    ctx,
    parquet: str,
    output: str,
    sdrf: str,
    quant_method: str,
    topn_peptides: int,
    min_aa: int,
    min_unique: int,
    remove_contaminants: bool,
    run_normalization: str,
    sample_normalization: str,
    normalization_proteins: str,
    fasta_file: str,
    ion_alignment: str,
    directlfq_cores: int,
    directlfq_min_nonan: int,
    export_peptides: str,
    export_ions: str,
) -> None:
    """
    Quantify proteins directly from feature parquet file.

    This is the recommended unified command that handles the full pipeline
    from features to proteins in one step.

    \b
    QUANTIFICATION METHODS:
      directlfq  - DirectLFQ (uses directlfq package for everything)
      ibaq       - Intensity-Based Absolute Quantification (requires --fasta)
      maxlfq     - MaxLFQ algorithm
      topn       - Top N peptides per protein (use --topn to set N, default 3)
      sum        - Sum of all peptides
      median     - Median of peptides

    \b
    NORMALIZATION:
      When using directlfq, the DirectLFQ package handles all normalization.
      For other methods, mokume applies normalization:
      - Run normalization: normalizes technical replicates within samples
      - Sample normalization: normalizes samples relative to each other
        Use 'hierarchical' for DirectLFQ-style clustering normalization
        combined with other quantification methods (e.g., iBAQ).

    \b
    EXAMPLES:
      # DirectLFQ quantification (uses directlfq package)
      mokume features2proteins -p data.parquet -o proteins.csv --quant-method directlfq

      # iBAQ with hierarchical normalization (best of both worlds)
      mokume features2proteins -p data.parquet -o proteins.csv \\
        --quant-method ibaq --sample-normalization hierarchical --fasta uniprot.fasta

      # MaxLFQ with default normalization
      mokume features2proteins -p data.parquet -o proteins.csv --quant-method maxlfq

      # TopN quantification (default top3)
      mokume features2proteins -p data.parquet -o proteins.csv --quant-method topn

      # Top5 quantification
      mokume features2proteins -p data.parquet -o proteins.csv --quant-method topn --topn 5
    """
    from mokume.pipeline import features_to_proteins as run_pipeline

    # Validate iBAQ requires fasta
    if quant_method.lower() == "ibaq" and not fasta_file:
        raise click.UsageError("iBAQ quantification requires --fasta option")

    # Info about DirectLFQ ignoring normalization settings
    if quant_method.lower() == "directlfq":
        if run_normalization != "median" or sample_normalization != "globalmedian":
            click.echo(
                "Note: DirectLFQ handles its own normalization. "
                "--run-normalization and --sample-normalization are ignored.",
                err=True,
            )

    # Handle topn method - construct the method name with N
    effective_quant_method = quant_method
    if quant_method.lower() == "topn":
        effective_quant_method = f"top{topn_peptides}"
        click.echo(f"Using Top{topn_peptides} quantification method")

    # Run the pipeline
    run_pipeline(
        parquet=parquet,
        output=output,
        sdrf=sdrf,
        quant_method=effective_quant_method,
        min_aa=min_aa,
        min_unique_peptides=min_unique,
        remove_contaminants=remove_contaminants,
        run_normalization=run_normalization,
        sample_normalization=sample_normalization,
        normalization_proteins_file=normalization_proteins,
        fasta_file=fasta_file,
        ion_alignment=ion_alignment,
        directlfq_num_cores=directlfq_cores,
        export_peptides=export_peptides,
        export_ions=export_ions,
    )

    click.echo(f"Protein intensities saved to: {output}")
