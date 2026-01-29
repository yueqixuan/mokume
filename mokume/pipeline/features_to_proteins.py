"""
Unified pipeline: features → proteins in one step.

This module provides the main `features_to_proteins` function and
`QuantificationPipeline` class that handle the full proteomics
quantification workflow from feature-level parquet files to
protein intensities.

The pipeline automatically handles:
- Loading and filtering parquet data
- Normalization (run-level and sample-level)
- Protein quantification using various methods
- Optional intermediate exports (peptides, ions)

When DirectLFQ is selected as the quantification method, the pipeline
delegates ALL processing (normalization + quantification) to the
directlfq package for reproducibility.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

from mokume.normalization.peptide import (
    Feature,
    SQLFilterBuilder,
    analyse_sdrf,
    remove_contaminants_entrapments_decoys,
    apply_initial_filtering,
    get_peptidoform_normalize_intensities,
    sum_peptidoform_intensities,
    reformat_quantms_feature_table_quant_labels,
)
from mokume.normalization.hierarchical import HierarchicalSampleNormalizer
from mokume.model.normalization import (
    FeatureNormalizationMethod,
    PeptideNormalizationMethod,
)
from mokume.model.labeling import QuantificationCategory
from mokume.core.constants import (
    PROTEIN_NAME,
    PEPTIDE_CANONICAL,
    NORM_INTENSITY,
    SAMPLE_ID,
    INTENSITY,
    PARQUET_COLUMNS,
    AGGREGATION_LEVEL_SAMPLE,
)
from mokume.core.logger import get_logger

logger = get_logger("mokume.pipeline")


@dataclass
class PipelineConfig:
    """
    Configuration for the quantification pipeline.

    Attributes
    ----------
    parquet : str
        Path to the input parquet file (quantms.io/qpx format).
    sdrf : str, optional
        Path to SDRF file for sample metadata.
    quant_method : str
        Quantification method: directlfq, ibaq, maxlfq, top3, top5, sum, median.
    min_aa : int
        Minimum amino acid length for peptides.
    min_unique_peptides : int
        Minimum unique peptides per protein.
    remove_contaminants : bool
        Whether to remove contaminants and decoys.
    run_normalization : str
        Run/tech-rep normalization method: none, median, mean, max, etc.
    sample_normalization : str
        Sample normalization method: none, globalMedian, conditionMedian, hierarchical.
    normalization_proteins_file : str, optional
        File with protein IDs to use for normalization (one per line).
    fasta_file : str, optional
        FASTA file path (required for iBAQ).
    ion_alignment : str, optional
        Ion alignment method for MaxLFQ: none, hierarchical.
    directlfq_num_cores : int, optional
        Number of cores for DirectLFQ.
    directlfq_min_nonan : int
        Minimum non-NaN values for DirectLFQ.
    directlfq_num_samples_quadratic : int
        Samples threshold for quadratic optimization.
    export_peptides : str, optional
        Path to export normalized peptides.
    export_ions : str, optional
        Path to export normalized ions (DirectLFQ only).
    """

    parquet: str
    sdrf: Optional[str] = None
    quant_method: str = "maxlfq"

    # Filtering
    min_aa: int = 7
    min_unique_peptides: int = 2
    remove_contaminants: bool = True

    # Normalization
    run_normalization: str = "median"
    sample_normalization: str = "globalMedian"
    normalization_proteins_file: Optional[str] = None

    # Method-specific
    fasta_file: Optional[str] = None
    ion_alignment: Optional[str] = None

    # DirectLFQ-specific
    directlfq_num_cores: Optional[int] = None
    directlfq_min_nonan: int = 1
    directlfq_num_samples_quadratic: int = 50

    # Optional exports
    export_peptides: Optional[str] = None
    export_ions: Optional[str] = None


class QuantificationPipeline:
    """
    Unified pipeline: features → proteins.

    This class handles the full proteomics quantification workflow from
    feature-level parquet files to protein intensities.

    The pipeline adapts based on the quantification method:
    - DirectLFQ: Delegates ALL processing to the directlfq package
    - Others (iBAQ, MaxLFQ, etc.): Uses mokume's native implementations

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration object.

    Examples
    --------
    >>> from mokume.pipeline import QuantificationPipeline, PipelineConfig
    >>>
    >>> # DirectLFQ quantification
    >>> config = PipelineConfig(
    ...     parquet="data.parquet",
    ...     quant_method="directlfq",
    ... )
    >>> pipeline = QuantificationPipeline(config)
    >>> proteins = pipeline.run()
    >>>
    >>> # iBAQ with hierarchical normalization
    >>> config = PipelineConfig(
    ...     parquet="data.parquet",
    ...     quant_method="ibaq",
    ...     sample_normalization="hierarchical",
    ...     fasta_file="uniprot.fasta",
    ... )
    >>> pipeline = QuantificationPipeline(config)
    >>> proteins = pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate configuration and check for required parameters."""
        if not Path(self.config.parquet).exists():
            raise FileNotFoundError(f"Parquet file not found: {self.config.parquet}")

        if self.config.quant_method.lower() == "ibaq" and not self.config.fasta_file:
            raise ValueError("iBAQ quantification requires --fasta-file")

        if self.config.fasta_file and not Path(self.config.fasta_file).exists():
            raise FileNotFoundError(f"FASTA file not found: {self.config.fasta_file}")

    def run(self) -> pd.DataFrame:
        """
        Execute the full pipeline.

        Returns
        -------
        pd.DataFrame
            Protein intensities matrix (proteins × samples).
        """
        quant_method = self.config.quant_method.lower()
        logger.info(f"Starting pipeline with quant_method={quant_method}")

        if quant_method == "directlfq":
            return self._run_directlfq_pipeline()
        else:
            return self._run_mokume_pipeline()

    def _run_directlfq_pipeline(self) -> pd.DataFrame:
        """
        Run pipeline using DirectLFQ package.

        DirectLFQ handles ALL processing (normalization + quantification)
        for reproducibility with the official implementation.
        """
        try:
            import directlfq.lfq_manager as lfq_manager
            import directlfq.utils as lfq_utils
            import directlfq.protein_intensity_estimation as lfq_estimation
            import directlfq.normalization as lfq_norm
            import directlfq.config as lfq_config
        except ImportError:
            raise ImportError(
                "DirectLFQ quantification requires the directlfq package.\n"
                "Install with: pip install directlfq\n"
                "Or: pip install mokume[directlfq]"
            )

        logger.info("Loading and filtering data for DirectLFQ...")
        filtered_df = self._load_and_filter_for_directlfq()

        logger.info(f"Filtered data: {len(filtered_df)} features")

        # Convert to DirectLFQ format (wide, log2)
        logger.info("Converting to DirectLFQ format...")
        directlfq_input = self._convert_to_directlfq_format(filtered_df)

        logger.info(f"DirectLFQ input shape: {directlfq_input.shape}")

        # Configure DirectLFQ
        lfq_config.set_global_protein_and_ion_id(protein_id="protein", quant_id="ion")
        lfq_config.set_compile_normalized_ion_table(
            self.config.export_ions is not None
        )

        # Run DirectLFQ normalization
        logger.info("Running DirectLFQ sample normalization...")
        normed_df = lfq_norm.NormalizationManagerSamplesOnSelectedProteins(
            directlfq_input,
            num_samples_quadratic=self.config.directlfq_num_samples_quadratic,
        ).complete_dataframe

        # Run DirectLFQ protein estimation
        logger.info("Running DirectLFQ protein estimation...")
        protein_df, ion_df = lfq_estimation.estimate_protein_intensities(
            normed_df,
            min_nonan=self.config.directlfq_min_nonan,
            num_samples_quadratic=10,
            num_cores=self.config.directlfq_num_cores,
        )

        # Export ions if requested
        if self.config.export_ions and ion_df is not None:
            logger.info(f"Exporting ions to {self.config.export_ions}")
            ion_df.to_csv(self.config.export_ions)

        logger.info(f"DirectLFQ complete: {len(protein_df)} proteins")

        return protein_df

    def _run_mokume_pipeline(self) -> pd.DataFrame:
        """
        Run pipeline using mokume's native implementations.

        This handles all quantification methods except DirectLFQ.
        """
        logger.info("Loading and filtering data...")
        peptide_df = self._load_and_process_peptides()

        logger.info(f"Processed peptides: {len(peptide_df)} rows")

        # Export peptides if requested
        if self.config.export_peptides:
            logger.info(f"Exporting peptides to {self.config.export_peptides}")
            peptide_df.to_csv(self.config.export_peptides, index=False)

        # Quantify proteins
        logger.info(f"Quantifying proteins with method: {self.config.quant_method}")
        protein_df = self._quantify(peptide_df)

        logger.info(f"Quantification complete: {len(protein_df)} proteins")

        return protein_df

    def _load_and_filter_for_directlfq(self) -> pd.DataFrame:
        """Load and filter data for DirectLFQ processing."""
        filter_builder = SQLFilterBuilder(
            remove_contaminants=self.config.remove_contaminants,
            min_peptide_length=self.config.min_aa,
            require_unique=True,
        )

        feature = Feature(self.config.parquet, filter_builder=filter_builder)

        if self.config.sdrf:
            feature.enrich_with_sdrf(self.config.sdrf)

        # Build query with filters
        where_clause = filter_builder.build_where_clause()
        query = f"""
            SELECT
                pg_accessions,
                sequence,
                sample_accession,
                intensity
            FROM parquet_db
            WHERE {where_clause}
        """

        df = feature.parquet_db.sql(query).df()

        # Parse protein accessions
        df["protein"] = df["pg_accessions"].apply(
            lambda x: x[0].split("|")[1] if x and "|" in str(x[0]) else (x[0] if x else "")
        )

        # Filter by min unique peptides
        peptide_counts = df.groupby("protein")["sequence"].nunique()
        valid_proteins = peptide_counts[
            peptide_counts >= self.config.min_unique_peptides
        ].index
        df = df[df["protein"].isin(valid_proteins)]

        return df

    def _convert_to_directlfq_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to DirectLFQ expected format (wide, log2, MultiIndex)."""
        # Pivot to wide format
        wide = df.pivot_table(
            index=["protein", "sequence"],
            columns="sample_accession",
            values="intensity",
            aggfunc="sum",
        )

        # Replace 0 with NaN and log2 transform
        wide = wide.replace(0, np.nan)
        wide = np.log2(wide)

        # Set index names for DirectLFQ
        wide.index.names = ["protein", "ion"]

        return wide

    def _load_and_process_peptides(self) -> pd.DataFrame:
        """Load data and apply normalization for mokume quantification methods."""
        filter_builder = SQLFilterBuilder(
            remove_contaminants=self.config.remove_contaminants,
            min_peptide_length=self.config.min_aa,
            require_unique=True,
        )

        feature = Feature(self.config.parquet, filter_builder=filter_builder)

        if self.config.sdrf:
            feature.enrich_with_sdrf(self.config.sdrf)
            technical_repetitions, label, sample_names, choice = analyse_sdrf(
                self.config.sdrf
            )
        else:
            (
                technical_repetitions,
                label,
                sample_names,
                choice,
            ) = feature.experimental_inference

        # Get normalization factors if needed
        med_map = {}
        sample_norm = self.config.sample_normalization.lower()

        if sample_norm == "globalmedian":
            med_map = feature.get_median_map()
        elif sample_norm == "conditionmedian":
            med_map = feature.get_median_map_to_condition()
        elif sample_norm == "hierarchical":
            # Hierarchical normalization is applied after loading all data
            pass

        # Process samples
        all_peptides = []

        for samples, df in feature.iter_samples():
            df.dropna(subset=["pg_accessions"], inplace=True)

            for sample in samples:
                dataset_df = df[df["sample_accession"] == sample].copy()
                dataset_df = dataset_df[dataset_df["unique"] == 1]
                dataset_df = dataset_df[PARQUET_COLUMNS]

                dataset_df = reformat_quantms_feature_table_quant_labels(
                    dataset_df, label, choice
                )
                dataset_df = apply_initial_filtering(
                    dataset_df, self.config.min_aa, AGGREGATION_LEVEL_SAMPLE
                )

                # Filter by min unique peptides
                dataset_df = dataset_df.groupby(PROTEIN_NAME).filter(
                    lambda x: len(set(x[PEPTIDE_CANONICAL]))
                    >= self.config.min_unique_peptides
                )

                if self.config.remove_contaminants:
                    dataset_df = remove_contaminants_entrapments_decoys(dataset_df)

                dataset_df.rename(columns={INTENSITY: NORM_INTENSITY}, inplace=True)

                # Apply run normalization
                run_norm = self.config.run_normalization.lower()
                if run_norm not in ("none", "") and technical_repetitions > 1:
                    run_method = FeatureNormalizationMethod.from_str(run_norm)
                    dataset_df = run_method(dataset_df, technical_repetitions)

                dataset_df = get_peptidoform_normalize_intensities(dataset_df)
                dataset_df = sum_peptidoform_intensities(
                    dataset_df, AGGREGATION_LEVEL_SAMPLE
                )

                # Apply sample normalization (non-hierarchical)
                if sample_norm in ("globalmedian", "conditionmedian"):
                    peptide_method = PeptideNormalizationMethod.from_str(sample_norm)
                    dataset_df = peptide_method(dataset_df, sample, med_map)

                all_peptides.append(dataset_df)

        # Combine all peptides
        combined_df = pd.concat(all_peptides, ignore_index=True)

        # Apply hierarchical normalization if selected
        if sample_norm == "hierarchical":
            combined_df = self._apply_hierarchical_normalization(combined_df)

        return combined_df

    def _apply_hierarchical_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply hierarchical sample normalization."""
        logger.info("Applying hierarchical sample normalization...")

        # Convert to wide format for normalization
        wide = df.pivot_table(
            index=[PROTEIN_NAME, PEPTIDE_CANONICAL],
            columns=SAMPLE_ID,
            values=NORM_INTENSITY,
            aggfunc="sum",
        )

        # Log2 transform for normalization
        wide = wide.replace(0, np.nan)
        wide_log2 = np.log2(wide)

        # Load selected proteins if specified
        selected_proteins = None
        if self.config.normalization_proteins_file:
            with open(self.config.normalization_proteins_file) as f:
                selected_proteins = [line.strip() for line in f if line.strip()]
            logger.info(f"Using {len(selected_proteins)} selected proteins for normalization")

        # Apply hierarchical normalization
        normalizer = HierarchicalSampleNormalizer(
            num_samples_quadratic=self.config.directlfq_num_samples_quadratic,
            selected_proteins=selected_proteins,
        )

        normalized_log2 = normalizer.fit_transform(wide_log2)

        # Convert back to linear scale
        normalized_wide = 2 ** normalized_log2

        # Convert back to long format
        normalized_long = normalized_wide.reset_index().melt(
            id_vars=[PROTEIN_NAME, PEPTIDE_CANONICAL],
            var_name=SAMPLE_ID,
            value_name=NORM_INTENSITY,
        )

        # Remove NaN rows
        normalized_long = normalized_long.dropna(subset=[NORM_INTENSITY])

        logger.info(f"Hierarchical normalization complete: {len(normalized_long)} rows")

        return normalized_long

    def _quantify(self, peptide_df: pd.DataFrame) -> pd.DataFrame:
        """Apply protein quantification method."""
        import re
        from mokume.quantification import get_quantification_method, TopNQuantification

        quant_method = self.config.quant_method.lower()

        if quant_method == "ibaq":
            return self._quantify_ibaq(peptide_df)
        elif quant_method in ("maxlfq", "sum", "all"):
            # Use the existing quantification infrastructure
            method = get_quantification_method(quant_method)
            result = method.quantify(
                peptide_df,
                protein_column=PROTEIN_NAME,
                peptide_column=PEPTIDE_CANONICAL,
                intensity_column=NORM_INTENSITY,
                sample_column=SAMPLE_ID,
            )
            # Convert long format to wide format
            return self._to_wide_format(result, quant_method)
        elif quant_method.startswith("top"):
            # Handle topN where N can be any number (top3, top5, top10, etc.)
            match = re.match(r"top(\d+)", quant_method)
            if match:
                n = int(match.group(1))
            else:
                n = 3  # Default to top3
            method = TopNQuantification(n=n)
            result = method.quantify(
                peptide_df,
                protein_column=PROTEIN_NAME,
                peptide_column=PEPTIDE_CANONICAL,
                intensity_column=NORM_INTENSITY,
                sample_column=SAMPLE_ID,
            )
            # Convert long format to wide format
            return self._to_wide_format(result, quant_method)
        elif quant_method == "median":
            return self._quantify_median(peptide_df)
        else:
            raise ValueError(f"Unknown quantification method: {quant_method}")

    def _to_wide_format(self, long_df: pd.DataFrame, method_name: str) -> pd.DataFrame:
        """Convert long format quantification results to wide format."""
        import re

        # Determine the intensity column name based on method
        intensity_col_map = {
            "maxlfq": "MaxLFQIntensity",
            "sum": "SumIntensity",
            "all": "SumIntensity",
        }

        # Handle topN methods dynamically
        if method_name.startswith("top"):
            match = re.match(r"top(\d+)", method_name)
            if match:
                n = match.group(1)
                intensity_col = f"Top{n}Intensity"
            else:
                intensity_col = "Top3Intensity"
        else:
            intensity_col = intensity_col_map.get(method_name, "intensity")

        # Find the actual intensity column in the DataFrame
        # First check for exact match, then look for any TopN column
        if intensity_col not in long_df.columns:
            # Look for any TopNIntensity column
            for col in long_df.columns:
                if re.match(r"Top\d+Intensity", col):
                    intensity_col = col
                    break
            else:
                # Fall back to other known columns
                candidates = [
                    "intensity",
                    NORM_INTENSITY,
                    "MaxLFQIntensity",
                    "SumIntensity",
                ]
                for col in candidates:
                    if col in long_df.columns:
                        intensity_col = col
                        break
                else:
                    # Last resort: use the last numeric column
                    numeric_cols = long_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        intensity_col = numeric_cols[-1]

        logger.debug(f"Using intensity column: {intensity_col}")

        # Pivot to wide format
        wide_df = long_df.pivot(
            index=PROTEIN_NAME,
            columns=SAMPLE_ID,
            values=intensity_col,
        )

        return wide_df.reset_index()

    def _quantify_ibaq(self, peptide_df: pd.DataFrame) -> pd.DataFrame:
        """Quantify using iBAQ method."""
        from mokume.quantification.ibaq import extract_fasta

        # Get unique proteins
        proteins = peptide_df[PROTEIN_NAME].unique().tolist()

        logger.info(f"Computing iBAQ for {len(proteins)} proteins using FASTA...")

        # Extract theoretical peptide counts from FASTA
        unique_peptide_counts, mw_dict, found_proteins = extract_fasta(
            fasta=self.config.fasta_file,
            enzyme="Trypsin",
            proteins=proteins,
            min_aa=self.config.min_aa,
            max_aa=50,
            tpa=False,
        )

        logger.info(f"Found {len(found_proteins)} proteins in FASTA")

        # Filter peptide_df to found proteins only
        peptide_df = peptide_df[peptide_df[PROTEIN_NAME].isin(found_proteins)]

        # Sum intensities per protein per sample
        protein_intensities = (
            peptide_df.groupby([PROTEIN_NAME, SAMPLE_ID])[NORM_INTENSITY]
            .sum()
            .reset_index()
        )

        # Calculate iBAQ = sum(intensity) / num_theoretical_peptides
        def calc_ibaq(row):
            protein = row[PROTEIN_NAME]
            num_peptides = unique_peptide_counts.get(protein, 1)
            return row[NORM_INTENSITY] / num_peptides if num_peptides > 0 else 0

        protein_intensities["iBAQ"] = protein_intensities.apply(calc_ibaq, axis=1)

        # Pivot to wide format
        result_wide = protein_intensities.pivot(
            index=PROTEIN_NAME,
            columns=SAMPLE_ID,
            values="iBAQ",
        )

        logger.info(f"iBAQ complete: {len(result_wide)} proteins")

        return result_wide.reset_index()

    def _quantify_median(self, peptide_df: pd.DataFrame) -> pd.DataFrame:
        """Quantify using median of peptides."""
        result = (
            peptide_df.groupby([PROTEIN_NAME, SAMPLE_ID])[NORM_INTENSITY]
            .median()
            .reset_index()
        )

        # Pivot to wide format
        result_wide = result.pivot(
            index=PROTEIN_NAME, columns=SAMPLE_ID, values=NORM_INTENSITY
        )

        return result_wide.reset_index()


def features_to_proteins(
    parquet: str,
    output: str,
    sdrf: Optional[str] = None,
    quant_method: str = "maxlfq",
    min_aa: int = 7,
    min_unique_peptides: int = 2,
    remove_contaminants: bool = True,
    run_normalization: str = "median",
    sample_normalization: str = "globalMedian",
    normalization_proteins_file: Optional[str] = None,
    fasta_file: Optional[str] = None,
    ion_alignment: Optional[str] = None,
    directlfq_num_cores: Optional[int] = None,
    export_peptides: Optional[str] = None,
    export_ions: Optional[str] = None,
) -> pd.DataFrame:
    """
    Quantify proteins directly from feature parquet file.

    This is the main entry point for the unified pipeline that handles
    the full workflow from features to proteins in one step.

    Parameters
    ----------
    parquet : str
        Path to the input parquet file (quantms.io/qpx format).
    output : str
        Path for the output protein intensities file.
    sdrf : str, optional
        Path to SDRF file for sample metadata.
    quant_method : str
        Quantification method. Options:
        - 'directlfq': Uses DirectLFQ package (normalization + quantification)
        - 'ibaq': Intensity-Based Absolute Quantification
        - 'maxlfq': MaxLFQ algorithm
        - 'top3': Top 3 peptides per protein
        - 'top5': Top 5 peptides per protein
        - 'sum': Sum of all peptides
        - 'median': Median of peptides
    min_aa : int
        Minimum amino acid length for peptides. Default: 7.
    min_unique_peptides : int
        Minimum unique peptides per protein. Default: 2.
    remove_contaminants : bool
        Whether to remove contaminants and decoys. Default: True.
    run_normalization : str
        Run/technical replicate normalization method. Options:
        none, median, mean, max, global, max_min, IQR.
        Ignored when quant_method='directlfq'.
    sample_normalization : str
        Sample-to-sample normalization method. Options:
        - 'none': No normalization
        - 'globalMedian': Sample median / global median
        - 'conditionMedian': Condition-specific median
        - 'hierarchical': DirectLFQ-style hierarchical clustering
        Ignored when quant_method='directlfq'.
    normalization_proteins_file : str, optional
        File with protein IDs to use for normalization (one per line).
        Useful for normalizing on housekeeping genes.
    fasta_file : str, optional
        FASTA file path. Required for iBAQ quantification.
    ion_alignment : str, optional
        Ion alignment method for MaxLFQ: none, hierarchical.
    directlfq_num_cores : int, optional
        Number of cores for DirectLFQ parallel processing.
    export_peptides : str, optional
        Path to export normalized peptides (for debugging/analysis).
    export_ions : str, optional
        Path to export normalized ions (DirectLFQ only).

    Returns
    -------
    pd.DataFrame
        Protein intensities matrix.

    Examples
    --------
    >>> # DirectLFQ quantification
    >>> proteins = features_to_proteins(
    ...     parquet="data.parquet",
    ...     output="proteins.csv",
    ...     quant_method="directlfq",
    ... )

    >>> # iBAQ with hierarchical normalization
    >>> proteins = features_to_proteins(
    ...     parquet="data.parquet",
    ...     output="proteins.csv",
    ...     quant_method="ibaq",
    ...     sample_normalization="hierarchical",
    ...     fasta_file="uniprot.fasta",
    ... )
    """
    config = PipelineConfig(
        parquet=parquet,
        sdrf=sdrf,
        quant_method=quant_method,
        min_aa=min_aa,
        min_unique_peptides=min_unique_peptides,
        remove_contaminants=remove_contaminants,
        run_normalization=run_normalization,
        sample_normalization=sample_normalization,
        normalization_proteins_file=normalization_proteins_file,
        fasta_file=fasta_file,
        ion_alignment=ion_alignment,
        directlfq_num_cores=directlfq_num_cores,
        export_peptides=export_peptides,
        export_ions=export_ions,
    )

    pipeline = QuantificationPipeline(config)
    protein_df = pipeline.run()

    # Save output
    protein_df.to_csv(output, index=False)
    logger.info(f"Protein intensities saved to {output}")

    return protein_df
