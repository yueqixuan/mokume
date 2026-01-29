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
from mokume.postprocessing.batch_correction import (
    is_batch_correction_available,
    detect_batches,
    extract_covariates_from_sdrf,
    apply_batch_correction,
)
from mokume.model.batch_correction import BatchDetectionMethod

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
    batch_correction : bool
        Whether to apply batch correction after quantification.
    batch_method : str
        Batch detection method: sample_prefix, run, column.
    batch_column : str, optional
        Column name for explicit batch assignment.
    batch_covariates : list, optional
        SDRF columns to use as covariates (biological signal to preserve).
        Example: ["characteristics[sex]", "characteristics[organism part]"]
    batch_parametric : bool
        Use parametric estimation for ComBat.
    batch_mean_only : bool
        Only adjust batch means, not individual effects.
    batch_ref : int, optional
        Reference batch ID.
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

    # Batch correction (applied after protein quantification)
    batch_correction: bool = False
    batch_method: str = "sample_prefix"
    batch_column: Optional[str] = None
    batch_covariates: Optional[list] = None
    batch_parametric: bool = True
    batch_mean_only: bool = False
    batch_ref: Optional[int] = None


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
            protein_df = self._run_directlfq_pipeline()
        else:
            protein_df = self._run_mokume_pipeline()

        # Apply batch correction if configured
        if self.config.batch_correction:
            protein_df = self._apply_batch_correction(protein_df)

        return protein_df

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

    def _apply_batch_correction(self, protein_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply batch correction after protein quantification.

        Batch = technical variation to REMOVE (from runs/files)
        Covariates = biological signal to PRESERVE (from SDRF characteristics)

        Parameters
        ----------
        protein_df : pd.DataFrame
            Protein intensities (proteins × samples format, with protein column).

        Returns
        -------
        pd.DataFrame
            Batch-corrected protein intensities.
        """
        if not is_batch_correction_available():
            raise ImportError(
                "Batch correction requires inmoose package. "
                "Install with: pip install mokume[batch-correction]"
            )

        # Identify protein column and sample columns
        if PROTEIN_NAME in protein_df.columns:
            protein_col = PROTEIN_NAME
        elif "protein" in protein_df.columns:
            protein_col = "protein"
        else:
            protein_col = protein_df.columns[0]

        sample_cols = [c for c in protein_df.columns if c != protein_col]

        if len(sample_cols) < 2:
            logger.warning("Not enough samples for batch correction, skipping")
            return protein_df

        # Create intensity matrix for batch correction (features × samples)
        intensity_matrix = protein_df.set_index(protein_col)[sample_cols]

        # 1. Detect batches (technical variation to remove)
        try:
            batch_method = BatchDetectionMethod.from_str(self.config.batch_method)
        except ValueError:
            logger.warning(
                f"Unknown batch method '{self.config.batch_method}', "
                "using sample_prefix"
            )
            batch_method = BatchDetectionMethod.SAMPLE_PREFIX

        batch_indices = detect_batches(
            sample_ids=sample_cols,
            method=batch_method,
            batch_column_values=(
                self._get_batch_column_values(sample_cols)
                if self.config.batch_column else None
            ),
        )

        unique_batches = len(set(batch_indices))
        logger.info(f"Detected {unique_batches} batches for batch correction")

        if unique_batches < 2:
            logger.warning("Only 1 batch detected, skipping batch correction")
            return protein_df

        # Check minimum samples per batch
        from collections import Counter
        batch_counts = Counter(batch_indices)
        min_samples = min(batch_counts.values())
        if min_samples < 2:
            logger.warning(
                f"Some batches have fewer than 2 samples (min={min_samples}), "
                "skipping batch correction"
            )
            return protein_df

        # 2. Extract covariates from SDRF (biological signal to preserve)
        covariates = None
        if self.config.sdrf and self.config.batch_covariates:
            covariates = extract_covariates_from_sdrf(
                self.config.sdrf,
                sample_cols,
                self.config.batch_covariates,
            )
            if covariates:
                logger.info(
                    f"Extracted {len(self.config.batch_covariates)} "
                    f"covariates to preserve biological signal"
                )

        # 3. Apply ComBat batch correction
        logger.info("Applying ComBat batch correction...")
        try:
            corrected_matrix = apply_batch_correction(
                df=intensity_matrix,
                batch=batch_indices,
                covs=covariates,
                kwargs={
                    "par_prior": self.config.batch_parametric,
                    "mean_only": self.config.batch_mean_only,
                    "ref_batch": self.config.batch_ref,
                },
            )

            # Reconstruct DataFrame with protein column
            corrected_df = corrected_matrix.reset_index()
            corrected_df = corrected_df.rename(columns={"index": protein_col})

            logger.info(
                f"Batch correction complete: {len(corrected_df)} proteins, "
                f"{len(sample_cols)} samples"
            )
            return corrected_df

        except Exception as e:
            logger.error(f"Batch correction failed: {e}")
            logger.warning("Returning uncorrected protein intensities")
            return protein_df

    def _get_batch_column_values(self, sample_ids: list) -> Optional[list]:
        """Get batch values from SDRF for explicit batch column."""
        if not self.config.sdrf or not self.config.batch_column:
            return None

        try:
            import pandas as pd
            sdrf = pd.read_csv(self.config.sdrf, sep="\t")
            sdrf.columns = [c.lower() for c in sdrf.columns]

            batch_col = self.config.batch_column.lower()
            if batch_col not in sdrf.columns:
                logger.warning(f"Batch column '{self.config.batch_column}' not in SDRF")
                return None

            # Map sample IDs to batch values
            sample_to_batch = dict(zip(sdrf["source name"], sdrf[batch_col]))
            return [sample_to_batch.get(s, "unknown") for s in sample_ids]

        except Exception as e:
            logger.warning(f"Failed to extract batch column: {e}")
            return None


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
    # Batch correction parameters
    batch_correction: bool = False,
    batch_method: str = "sample_prefix",
    batch_column: Optional[str] = None,
    batch_covariates: Optional[list] = None,
    batch_parametric: bool = True,
    batch_mean_only: bool = False,
    batch_ref: Optional[int] = None,
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
    batch_correction : bool
        Whether to apply batch correction after quantification. Default: False.
    batch_method : str
        Batch detection method: sample_prefix, run, column. Default: sample_prefix.
    batch_column : str, optional
        Column name for explicit batch assignment (when batch_method='column').
    batch_covariates : list, optional
        SDRF columns to use as covariates (biological signal to preserve).
        Example: ["characteristics[sex]", "characteristics[organism part]"]
    batch_parametric : bool
        Use parametric estimation for ComBat. Default: True.
    batch_mean_only : bool
        Only adjust batch means, not individual effects. Default: False.
    batch_ref : int, optional
        Reference batch ID.

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

    >>> # MaxLFQ with batch correction and covariates
    >>> proteins = features_to_proteins(
    ...     parquet="data.parquet",
    ...     output="proteins.csv",
    ...     sdrf="experiment.sdrf.tsv",
    ...     quant_method="maxlfq",
    ...     batch_correction=True,
    ...     batch_covariates=["characteristics[sex]", "characteristics[tissue]"],
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
        # Batch correction
        batch_correction=batch_correction,
        batch_method=batch_method,
        batch_column=batch_column,
        batch_covariates=batch_covariates,
        batch_parametric=batch_parametric,
        batch_mean_only=batch_mean_only,
        batch_ref=batch_ref,
    )

    pipeline = QuantificationPipeline(config)
    protein_df = pipeline.run()

    # Save output
    protein_df.to_csv(output, index=False)
    logger.info(f"Protein intensities saved to {output}")

    return protein_df
