"""
Peptide-level normalization implementations.

This module provides functions for peptide-level normalization including
feature normalization and the main peptide_normalization function.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np
import duckdb

from mokume.model.labeling import QuantificationCategory, IsobaricLabel
from mokume.model.normalization import FeatureNormalizationMethod, PeptideNormalizationMethod
from mokume.core.constants import (
    BIOREPLICATE,
    TECHREPLICATE,
    CHANNEL,
    CONDITION,
    FRACTION,
    INTENSITY,
    NORM_INTENSITY,
    PEPTIDE_CANONICAL,
    PEPTIDE_CHARGE,
    PEPTIDE_SEQUENCE,
    PROTEIN_NAME,
    RUN,
    SAMPLE_ID,
    PARQUET_COLUMNS,
    parquet_map,
    AGGREGATION_LEVEL_SAMPLE,
    AGGREGATION_LEVEL_RUN,
)

from mokume.core.write_queue import WriteParquetTask, WriteCSVTask
from mokume.core.logger import get_logger, log_execution_time

if TYPE_CHECKING:
    from mokume.model.filters import PreprocessingFilterConfig

# Get a logger for this module
logger = get_logger("mokume.peptide_normalization")


@dataclass
class SQLFilterBuilder:
    """Builds SQL WHERE clauses for filtering parquet data at query time.

    This class is used to ensure that normalization factors (median maps, peptide
    frequencies, IRS scaling) are computed on filtered data that excludes contaminants,
    decoys, and other artifacts.

    Attributes
    ----------
    remove_contaminants : bool
        Whether to exclude rows containing contaminant patterns in pg_accessions.
    contaminant_patterns : list[str]
        List of substring patterns to identify contaminants (e.g., ["CONTAMINANT", "DECOY"]).
    min_intensity : float
        Minimum intensity threshold (0.0 means only filter intensity > 0).
    min_peptide_length : int
        Minimum peptide sequence length.
    require_unique : bool
        Whether to require unique peptides only (unique = 1).
    """

    remove_contaminants: bool = True
    contaminant_patterns: list[str] = field(
        default_factory=lambda: ["CONTAMINANT", "ENTRAP", "DECOY"]
    )
    min_intensity: float = 0.0
    min_peptide_length: int = 7
    require_unique: bool = True

    def build_where_clause(self) -> str:
        """Build SQL WHERE clause string for DuckDB queries.

        Returns
        -------
        str
            A SQL WHERE clause (without the WHERE keyword) that can be used
            in DuckDB queries to filter the parquet data.
        """
        conditions = []

        # Always filter intensity > 0
        conditions.append("intensity > 0")

        # Min intensity threshold
        if self.min_intensity > 0:
            conditions.append(f"intensity >= {self.min_intensity}")

        # Peptide length filter
        if self.min_peptide_length > 0:
            conditions.append(f'LENGTH("sequence") >= {self.min_peptide_length}')

        # Unique peptides only
        if self.require_unique:
            conditions.append('"unique" = 1')

        # Contaminant/decoy filter - cast pg_accessions array to text for LIKE matching
        if self.remove_contaminants and self.contaminant_patterns:
            pattern_conditions = []
            for pattern in self.contaminant_patterns:
                # Escape any SQL special characters in the pattern
                safe_pattern = pattern.replace("'", "''")
                pattern_conditions.append(
                    f"pg_accessions::text NOT LIKE '%{safe_pattern}%'"
                )
            conditions.append(f"({' AND '.join(pattern_conditions)})")

        return " AND ".join(conditions) if conditions else "1=1"


def parse_uniprot_accession(uniprot_id: str) -> str:
    """
    Parse a UniProt accession string to extract and return the core accession numbers.

    Parameters
    ----------
    uniprot_id : str
        A string containing one or more UniProt accessions.

    Returns
    -------
    str
        A semicolon-separated string of core accession numbers.
    """
    uniprot_list = uniprot_id.split(";")
    result_uniprot_list = []
    for accession in uniprot_list:
        if accession.count("|") == 2:
            accession = accession.split("|")[1]
        result_uniprot_list.append(accession)
    return ";".join(result_uniprot_list)


def get_canonical_peptide(peptide_sequence: str) -> str:
    """
    Remove modifications and special characters from a peptide sequence.

    Parameters
    ----------
    peptide_sequence : str
        The peptide sequence to be cleaned.

    Returns
    -------
    str
        The cleaned canonical peptide sequence.
    """
    clean_peptide = re.sub(r"[\(\[].*?[\)\]]", "", peptide_sequence)
    clean_peptide = clean_peptide.replace(".", "").replace("-", "")
    return clean_peptide


def analyse_sdrf(
    sdrf_path: str,
) -> tuple[int, QuantificationCategory, list[str], Optional[IsobaricLabel]]:
    """
    Analyzes an SDRF file to determine quantification details.

    Parameters
    ----------
    sdrf_path : str
        The file path to the SDRF file.

    Returns
    -------
    tuple[int, QuantificationCategory, list[str], Optional[IsobaricLabel]]
        A tuple containing the number of technical repetitions, the quantification category,
        a list of unique sample names, and the isobaric label scheme if applicable.
    """
    sdrf_df = pd.read_csv(sdrf_path, sep="\t")
    sdrf_df.columns = [i.lower() for i in sdrf_df.columns]

    labels = set(sdrf_df["comment[label]"])
    # Determine label type
    label, channel_set = QuantificationCategory.classify(labels)
    if label in (QuantificationCategory.TMT, QuantificationCategory.ITRAQ):
        choice_df = (
            pd.DataFrame.from_dict(channel_set.channels(), orient="index", columns=[CHANNEL])
            .reset_index()
            .rename(columns={"index": "comment[label]"})
        )
        sdrf_df = sdrf_df.merge(choice_df, on="comment[label]", how="left")
    sample_names = sdrf_df["source name"].unique().tolist()
    technical_repetitions = len(sdrf_df["comment[technical replicate]"].unique())
    return technical_repetitions, label, sample_names, channel_set


def remove_contaminants_entrapments_decoys(
    dataset: pd.DataFrame, protein_field=PROTEIN_NAME
) -> pd.DataFrame:
    """
    Remove rows from the dataset that contain contaminants, entrapments, or decoys.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame containing protein data.
    protein_field : str
        The column name in the DataFrame to check for contaminants.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the contaminants, entrapments, and decoys removed.
    """
    contaminants = ["CONTAMINANT", "ENTRAP", "DECOY"]
    cregex = "|".join(contaminants)
    return dataset[~dataset[protein_field].str.contains(cregex)]


def remove_protein_by_ids(
    dataset: pd.DataFrame, protein_file: str, protein_field=PROTEIN_NAME
) -> pd.DataFrame:
    """
    Remove proteins from a dataset based on a list of protein IDs.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset containing protein information.
    protein_file : str
        Path to the file containing protein IDs to be removed.
    protein_field : str
        The field in the dataset to check for protein IDs.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the specified proteins removed.
    """
    contaminants_reader = open(protein_file, "r")
    contaminants = contaminants_reader.read().split("\n")
    contaminants = [cont for cont in contaminants if cont.strip()]
    cregex = "|".join(contaminants)
    return dataset[~dataset[protein_field].str.contains(cregex, regex=True)]


def reformat_quantms_feature_table_quant_labels(
    data_df: pd.DataFrame, label: QuantificationCategory, choice: Optional[IsobaricLabel]
) -> pd.DataFrame:
    """
    Reformats a DataFrame containing quantification labels for QuantMS features.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input DataFrame containing quantification data.
    label : QuantificationCategory
        The quantification category (e.g., LFQ, TMT, ITRAQ).
    choice : Optional[IsobaricLabel]
        The isobaric label scheme, if applicable.

    Returns
    -------
    pd.DataFrame
        The reformatted DataFrame with updated column names and channel information.
    """
    data_df = data_df.rename(columns=parquet_map)
    data_df[PROTEIN_NAME] = data_df[PROTEIN_NAME].str.join(";")
    if label == QuantificationCategory.LFQ:
        data_df.drop(CHANNEL, inplace=True, axis=1)
    else:
        data_df[CHANNEL] = data_df[CHANNEL].map(choice.channels())

    return data_df


def apply_initial_filtering(
    data_df: pd.DataFrame,
    min_aa: int,
    aggregation_level: str = AGGREGATION_LEVEL_SAMPLE,
) -> pd.DataFrame:
    """
    Apply initial filtering to a DataFrame containing peptide data.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input DataFrame containing peptide data.
    min_aa : int
        The minimum number of amino acids required for peptides.
    aggregation_level : str
        Level at which to aggregate intensities. Options:
        - "sample": Aggregate at sample level (default)
        - "run": Aggregate at run level, preserving run information

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame with relevant columns.
    """
    # Remove 0 intensity signals from the data
    data_df = data_df[data_df[INTENSITY] > 0]

    data_df = data_df[(data_df["Condition"] != "Empty") | (data_df["Condition"].isnull())]

    # "Run" is NA for reference files not found in the SDRF file.
    if data_df[RUN].isna().any():
        missing_files = data_df.loc[
            data_df[RUN].isna(), "Reference"
        ].drop_duplicates().tolist()

        logger.warning(
            f"Reference files {missing_files} are not present in the SDRF file. Skipping calculation."
        )
        data_df.dropna(subset=[RUN], inplace=True)

    # Filter peptides with less amino acids than min_aa (default: 7)
    data_df.loc[:, "len"] = data_df[PEPTIDE_CANONICAL].apply(len)
    data_df = data_df[data_df["len"] >= min_aa]
    data_df.drop(["len"], inplace=True, axis=1)
    data_df[PROTEIN_NAME] = data_df[PROTEIN_NAME].apply(parse_uniprot_accession)
    if FRACTION not in data_df.columns:
        data_df[FRACTION] = 1

    # Try to extract technical replicate from run name
    try:
        if data_df[RUN].str.contains("_").all():
            # Get the last part after underscore (e.g., "S1_Brain_2" -> "2")
            last_parts = data_df[RUN].str.split("_").str.get(-1)
            data_df[TECHREPLICATE] = last_parts.astype("int")
        else:
            data_df[TECHREPLICATE] = data_df[RUN].astype("int")
    except (ValueError, TypeError):
        # Fall back to using run index
        unique_runs = data_df[RUN].unique().tolist()
        run_to_index = {run: i + 1 for i, run in enumerate(unique_runs)}
        data_df[TECHREPLICATE] = data_df[RUN].map(run_to_index)

    # Define columns to keep based on aggregation level
    columns_to_keep = [
        PROTEIN_NAME,
        PEPTIDE_SEQUENCE,
        PEPTIDE_CANONICAL,
        PEPTIDE_CHARGE,
        INTENSITY,
        CONDITION,
        TECHREPLICATE,
        BIOREPLICATE,
        FRACTION,
        SAMPLE_ID,
    ]

    # Include RUN column for run-level aggregation
    if aggregation_level == AGGREGATION_LEVEL_RUN:
        columns_to_keep.append(RUN)

    data_df = data_df[columns_to_keep]
    data_df[CONDITION] = pd.Categorical(data_df[CONDITION])
    data_df[SAMPLE_ID] = pd.Categorical(data_df[SAMPLE_ID])

    return data_df


def merge_fractions(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fractions in the dataset by grouping and aggregating normalized intensity.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame containing peptide data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with merged fractions and maximum normalized intensity.
    """
    dataset.dropna(subset=[NORM_INTENSITY], inplace=True)
    dataset = dataset.groupby(
        [
            PROTEIN_NAME,
            PEPTIDE_SEQUENCE,
            PEPTIDE_CANONICAL,
            PEPTIDE_CHARGE,
            CONDITION,
            BIOREPLICATE,
            TECHREPLICATE,
            SAMPLE_ID,
        ],
        observed=True,
    ).agg({NORM_INTENSITY: "max"})
    dataset.reset_index(inplace=True)
    return dataset


def get_peptidoform_normalize_intensities(
    dataset: pd.DataFrame, higher_intensity: bool = True
) -> pd.DataFrame:
    """
    Normalize peptide intensities in a dataset by selecting the highest intensity.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame containing peptide data.
    higher_intensity : bool
        If True, selects the row with the highest normalized intensity for each group.

    Returns
    -------
    pd.DataFrame
        A DataFrame with normalized intensities.
    """
    dataset.dropna(subset=[NORM_INTENSITY], inplace=True)
    if higher_intensity:
        dataset = dataset.loc[
            dataset.groupby(
                [PEPTIDE_SEQUENCE, PEPTIDE_CHARGE, SAMPLE_ID, CONDITION, BIOREPLICATE],
                observed=True,
            )[NORM_INTENSITY].idxmax()
        ]
    dataset.reset_index(drop=True, inplace=True)
    return dataset


def sum_peptidoform_intensities(
    dataset: pd.DataFrame,
    aggregation_level: str = AGGREGATION_LEVEL_SAMPLE,
) -> pd.DataFrame:
    """
    Aggregate normalized intensities for each unique peptidoform.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input DataFrame containing peptidoform data with normalized intensities.
    aggregation_level : str
        Level at which to aggregate intensities. Options:
        - "sample": Aggregate at sample level (default, original behavior)
        - "run": Aggregate at run level, preserving run information

    Returns
    -------
    pd.DataFrame
        A DataFrame with summed normalized intensities for each unique peptidoform entry.
    """
    dataset.dropna(subset=[NORM_INTENSITY], inplace=True)

    # Define columns based on aggregation level
    base_columns = [
        PROTEIN_NAME,
        PEPTIDE_CANONICAL,
        SAMPLE_ID,
        BIOREPLICATE,
        CONDITION,
        NORM_INTENSITY,
    ]

    groupby_columns = [
        PROTEIN_NAME,
        PEPTIDE_CANONICAL,
        SAMPLE_ID,
        BIOREPLICATE,
        CONDITION,
    ]

    # If run-level aggregation, include RUN/TECHREPLICATE columns
    if aggregation_level == AGGREGATION_LEVEL_RUN:
        if RUN in dataset.columns:
            base_columns.insert(-1, RUN)
            groupby_columns.append(RUN)
        if TECHREPLICATE in dataset.columns:
            base_columns.insert(-1, TECHREPLICATE)
            groupby_columns.append(TECHREPLICATE)

    dataset = dataset[[c for c in base_columns if c in dataset.columns]]

    dataset.loc[:, NORM_INTENSITY] = dataset.groupby(
        [c for c in groupby_columns if c in dataset.columns],
        observed=True,
    )[NORM_INTENSITY].transform("sum")
    dataset = dataset.drop_duplicates()
    dataset.reset_index(inplace=True, drop=True)
    return dataset


class Feature:
    """
    Represents a feature in a proteomics dataset, providing methods for data manipulation
    and analysis using a DuckDB database connection to a Parquet file.

    This class expects the quantms.io/qpx wide format where intensities are stored
    in a nested array: intensities[{sample_accession, channel, intensity}, ...]

    Attributes
    ----------
    filter_builder : Optional[SQLFilterBuilder]
        If provided, this filter builder will be used to apply SQL-level filtering
        when computing normalization factors (median maps, peptide frequencies, etc.).
        This ensures normalization is computed on clean data without contaminants/decoys.
    """

    labels: Optional[list[str]]
    label: Optional[QuantificationCategory]
    choice: Optional[IsobaricLabel]
    technical_repetitions: Optional[int]
    filter_builder: Optional[SQLFilterBuilder]

    def __init__(
        self, database_path: str, filter_builder: Optional[SQLFilterBuilder] = None
    ):
        if os.path.exists(database_path):
            self.parquet_db = duckdb.connect()

            # Create raw view from parquet
            self.parquet_db.execute(
                "CREATE VIEW parquet_db_raw AS SELECT * FROM parquet_scan('{}')".format(
                    database_path
                )
            )

            # UNNEST intensities array to create long format view
            # This converts the wide format (nested intensities) to long format
            self.parquet_db.execute("""
                CREATE VIEW parquet_db AS
                SELECT
                    sequence,
                    peptidoform,
                    pg_accessions,
                    precursor_charge,
                    reference_file_name,
                    "unique",
                    unnest.sample_accession,
                    unnest.channel,
                    unnest.intensity,
                    -- Defaults (can be enriched with SDRF later)
                    reference_file_name as run,
                    unnest.sample_accession as condition,
                    1 as biological_replicate,
                    '1' as fraction
                FROM parquet_db_raw, UNNEST(intensities) as unnest
                WHERE unnest.intensity IS NOT NULL AND unnest.intensity > 0
            """)

            self.samples = self.get_unique_samples()
            self.filter_builder = filter_builder
        else:
            raise FileNotFoundError(f"the file {database_path} does not exist.")

    def enrich_with_sdrf(self, sdrf_path: str) -> None:
        """Enrich parquet data with SDRF metadata (condition, biological_replicate, etc.).

        Parameters
        ----------
        sdrf_path : str
            Path to the SDRF file containing sample metadata.
        """
        sdrf_df = pd.read_csv(sdrf_path, sep="\t")
        sdrf_df.columns = [c.lower() for c in sdrf_df.columns]

        # Find the condition column (try factor value first, then characteristics)
        condition_col = None
        for col in sdrf_df.columns:
            if 'factor value' in col:
                condition_col = col
                break
        if condition_col is None:
            for col in sdrf_df.columns:
                if 'organism part' in col and 'characteristics' in col:
                    condition_col = col
                    break

        # Prepare SDRF mapping
        sdrf_mapping = pd.DataFrame({
            'sdrf_sample_accession': sdrf_df['source name'],
            'sdrf_condition': (
                sdrf_df[condition_col] if condition_col else sdrf_df['source name']
            ),
            'sdrf_biological_replicate': sdrf_df.get(
                'characteristics[biological replicate]', 1
            ),
            'sdrf_fraction': sdrf_df.get('comment[fraction identifier]', '1'),
        })

        self.parquet_db.register('sdrf_mapping', sdrf_mapping)

        # Create intermediate view for unnested data
        self.parquet_db.execute("""
            CREATE OR REPLACE VIEW parquet_db_unnested AS
            SELECT
                sequence,
                peptidoform,
                pg_accessions,
                precursor_charge,
                reference_file_name,
                "unique",
                unnest.sample_accession,
                unnest.channel,
                unnest.intensity,
                reference_file_name as run
            FROM parquet_db_raw, UNNEST(intensities) as unnest
            WHERE unnest.intensity IS NOT NULL AND unnest.intensity > 0
        """)

        # Recreate main view with SDRF data joined
        self.parquet_db.execute("DROP VIEW IF EXISTS parquet_db")
        self.parquet_db.execute("""
            CREATE VIEW parquet_db AS
            SELECT
                p.sequence,
                p.peptidoform,
                p.pg_accessions,
                p.precursor_charge,
                p.reference_file_name,
                p."unique",
                p.sample_accession,
                p.channel,
                p.intensity,
                p.run,
                COALESCE(s.sdrf_condition, p.sample_accession) as condition,
                COALESCE(CAST(s.sdrf_biological_replicate AS INTEGER), 1) as biological_replicate,
                COALESCE(CAST(s.sdrf_fraction AS VARCHAR), '1') as fraction
            FROM parquet_db_unnested p
            LEFT JOIN sdrf_mapping s ON p.sample_accession = s.sdrf_sample_accession
        """)

        logger.info("Enriched parquet data with SDRF metadata from %s", sdrf_path)

    @staticmethod
    def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names in the given DataFrame."""
        return df.rename(
            {"protein_accessions": "pg_accessions", "charge": "precursor_charge"}, axis=1
        )

    @property
    def experimental_inference(
        self,
    ) -> tuple[int, QuantificationCategory, list[str], Optional[IsobaricLabel]]:
        """Infers experimental details from the dataset."""
        self.labels = self.get_unique_labels()
        self.label, self.choice = QuantificationCategory.classify(self.labels)
        self.technical_repetitions = self.get_unique_tec_reps()
        return len(self.technical_repetitions), self.label, self.samples, self.choice

    def get_low_frequency_peptides(self, percentage: float = 0.2) -> tuple:
        """Identifies peptides that occur with low frequency across samples.

        If a filter_builder is set on this Feature instance, it will be used to
        exclude contaminants, decoys, and other artifacts from the frequency
        calculation.

        Parameters
        ----------
        percentage : float
            The frequency threshold. Peptides appearing in less than this
            fraction of samples are considered low frequency. Default is 0.2 (20%).

        Returns
        -------
        tuple
            A tuple of (protein_accession, sequence) pairs for low frequency peptides.
        """
        where_clause = (
            self.filter_builder.build_where_clause()
            if self.filter_builder
            else "1=1"
        )

        f_table = self.parquet_db.sql(
            f"""
            SELECT "sequence", "pg_accessions", COUNT(DISTINCT sample_accession) as "count"
            FROM parquet_db
            WHERE {where_clause}
            GROUP BY "sequence", "pg_accessions"
            """
        ).df()
        f_table.dropna(subset=["pg_accessions"], inplace=True)
        try:
            f_table["pg_accessions"] = f_table["pg_accessions"].apply(lambda x: x[0].split("|")[1])
        except IndexError:
            f_table["pg_accessions"] = f_table["pg_accessions"].apply(lambda x: x[0])
        except Exception as e:
            raise ValueError(
                "Some errors occurred when parsing pg_accessions column in feature parquet!"
            ) from e
        f_table.set_index(["sequence", "pg_accessions"], inplace=True)
        f_table.drop(
            f_table[f_table["count"] >= (percentage * len(self.samples))].index,
            inplace=True,
        )
        f_table.reset_index(inplace=True)
        return tuple(zip(f_table["pg_accessions"], f_table["sequence"]))

    @staticmethod
    def csv2parquet(csv):
        """Converts a CSV file to a Parquet file using DuckDB."""
        parquet_path = os.path.splitext(csv)[0] + ".parquet"
        duckdb.read_csv(csv).to_parquet(parquet_path)

    def get_report_from_database(self, samples: list, columns: list = None):
        """Retrieves a standardized report from the database for specified samples."""
        cols = ",".join(columns) if columns is not None else "*"
        database = self.parquet_db.sql(
            """SELECT {} FROM parquet_db WHERE sample_accession IN {}""".format(
                cols, tuple(samples)
            )
        )
        report = database.df()
        return Feature.standardize_df(report)

    def iter_samples(
        self, sample_num: int = 20, columns: list = None
    ) -> Iterator[tuple[list[str], pd.DataFrame]]:
        """Iterates over samples in batches."""
        ref_list = [
            self.samples[i : i + sample_num] for i in range(0, len(self.samples), sample_num)
        ]
        for refs in ref_list:
            batch_df = self.get_report_from_database(refs, columns)
            yield refs, batch_df

    def get_unique_samples(self) -> list[str]:
        """Retrieves a list of unique sample accessions from the Parquet database."""
        unique = self.parquet_db.sql("SELECT DISTINCT sample_accession FROM parquet_db").df()
        return unique["sample_accession"].tolist()

    def get_unique_labels(self) -> list[str]:
        """Retrieves a list of unique channel labels from the Parquet database."""
        unique = self.parquet_db.sql("SELECT DISTINCT channel FROM parquet_db").df()
        return unique["channel"].tolist()

    def get_unique_tec_reps(self) -> list[int]:
        """Retrieves a list of unique technical repetition identifiers.

        Attempts to extract technical replicate numbers from run names in order:
        1. If run name contains '_', try to extract the last part as an integer
        2. If run name is numeric, use it directly
        3. Fall back to sequential integers (1, 2, 3, ...) based on unique runs
        """
        unique = self.parquet_db.sql("SELECT DISTINCT run FROM parquet_db").df()

        try:
            # Try to extract last part after underscore as tech rep
            if unique["run"].str.contains("_").all():
                # Get the last part after splitting by underscore
                last_parts = unique["run"].str.split("_").str.get(-1)
                unique["run"] = last_parts.astype("int")
            else:
                # Try to convert directly to int
                unique["run"] = unique["run"].astype("int")
        except (ValueError, TypeError):
            # Fall back to sequential integers
            unique["run"] = list(range(1, len(unique) + 1))

        return unique["run"].tolist()

    def get_median_map(self) -> dict[str, float]:
        """Computes a median intensity map for samples.

        If a filter_builder is set on this Feature instance, it will be used to
        exclude contaminants, decoys, and other artifacts from the median
        calculation. This ensures normalization factors are computed on clean data.

        Returns
        -------
        dict[str, float]
            A dictionary mapping sample accessions to their normalization factors
            (sample median / global median).
        """
        where_clause = (
            self.filter_builder.build_where_clause()
            if self.filter_builder
            else "1=1"
        )

        # Use SQL aggregation with filtering for efficiency
        result = self.parquet_db.sql(
            f"""
            SELECT sample_accession, MEDIAN(intensity) as median_intensity
            FROM parquet_db
            WHERE {where_clause}
            GROUP BY sample_accession
            """
        ).df()

        med_map = dict(zip(result["sample_accession"], result["median_intensity"]))
        global_med = np.median(list(med_map.values()))

        for sample, med in med_map.items():
            med_map[sample] = med / global_med

        return med_map

    def get_report_condition_from_database(self, cons: list, columns: list = None) -> pd.DataFrame:
        """Retrieves a standardized report from the database for specified conditions."""
        cols = ",".join(columns) if columns is not None else "*"
        database = self.parquet_db.sql(
            f"""SELECT {cols} FROM parquet_db WHERE condition IN {tuple(cons)}"""
        )
        report = database.df()
        return Feature.standardize_df(report)

    def iter_conditions(
        self, conditions: int = 10, columns: list = None
    ) -> Iterator[tuple[list[str], pd.DataFrame]]:
        """Iterates over experimental conditions in batches."""
        condition_list = self.get_unique_conditions()
        ref_list = [
            condition_list[i : i + conditions] for i in range(0, len(condition_list), conditions)
        ]
        for refs in ref_list:
            batch_df = self.get_report_condition_from_database(refs, columns)
            yield refs, batch_df

    def get_unique_conditions(self) -> list[str]:
        """Retrieves a list of unique experimental conditions from the Parquet database."""
        unique = self.parquet_db.sql("SELECT DISTINCT condition FROM parquet_db").df()
        return unique["condition"].tolist()

    def get_median_map_to_condition(self) -> dict[str, dict[str, float]]:
        """Computes a median intensity map for each experimental condition.

        If a filter_builder is set on this Feature instance, it will be used to
        exclude contaminants, decoys, and other artifacts from the median
        calculation. This ensures normalization factors are computed on clean data.

        Returns
        -------
        dict[str, dict[str, float]]
            A nested dictionary mapping conditions to sample normalization factors.
            For each condition, samples are normalized to the condition mean.
        """
        where_clause = (
            self.filter_builder.build_where_clause()
            if self.filter_builder
            else "1=1"
        )

        # Use SQL aggregation with filtering for efficiency
        result = self.parquet_db.sql(
            f"""
            SELECT condition, sample_accession, MEDIAN(intensity) as median_intensity
            FROM parquet_db
            WHERE {where_clause}
            GROUP BY condition, sample_accession
            """
        ).df()

        med_map = {}
        for condition in result["condition"].unique():
            cond_data = result[result["condition"] == condition]
            meds = pd.Series(
                cond_data["median_intensity"].values,
                index=cond_data["sample_accession"].values,
            )
            meds = meds / meds.mean()
            med_map[condition] = meds.to_dict()

        return med_map

    def get_irs_scaling_factors(
        self,
        irs_channel: str,
        irs_stat: str = "median",
        irs_scope: str = "global",
    ) -> dict[int, float]:
        """Compute IRS (Internal Reference Scaling) factors with filtering applied.

        If a filter_builder is set on this Feature instance, it will be used to
        exclude contaminants, decoys, and other artifacts from the IRS calculation.

        Parameters
        ----------
        irs_channel : str
            The channel label to use as internal reference (e.g., "126").
        irs_stat : str
            Statistic to use for computing reference values: "median" or "mean".
        irs_scope : str
            Scope of normalization: "global", "by_mixture", or "two_stage".

        Returns
        -------
        dict[int, float]
            Dictionary mapping technical replicate indices to scaling factors.
        """
        stat_fn = "median" if (irs_stat or "").lower() == "median" else "avg"

        # Build filter conditions for contaminants only (not unique peptide requirement)
        # since IRS uses specific channel which may have different characteristics
        filter_conditions = ["intensity > 0"]

        if self.filter_builder and self.filter_builder.remove_contaminants:
            for pattern in self.filter_builder.contaminant_patterns:
                safe_pattern = pattern.replace("'", "''")
                filter_conditions.append(
                    f"pg_accessions::text NOT LIKE '%{safe_pattern}%'"
                )

        if self.filter_builder and self.filter_builder.min_intensity > 0:
            filter_conditions.append(
                f"intensity >= {self.filter_builder.min_intensity}"
            )

        # Add channel filter
        filter_conditions.append(f"channel = '{irs_channel}'")
        where_clause = " AND ".join(filter_conditions)

        irs_df = self.parquet_db.sql(
            f"""
            SELECT run, {stat_fn}(intensity) as irs_value, mixture, techreplicate as techrep_guess
            FROM (
                SELECT *,
                       CASE WHEN position('_' in run) > 0 THEN CAST(split_part(run, '_', 2) AS INTEGER)
                            ELSE CAST(run AS INTEGER) END AS techreplicate
                FROM parquet_db
                WHERE {where_clause}
            )
            GROUP BY run, mixture, techrep_guess
            """
        ).df()

        irs_scale_by_techrep: dict[int, float] = {}

        if len(irs_df.index) > 0:
            irs_df = irs_df[irs_df["irs_value"] > 0]

            if irs_scope.lower() == "by_mixture":
                transform_fn = "median" if stat_fn == "median" else "mean"
                irs_df["mixture_center"] = irs_df.groupby("mixture")[
                    "irs_value"
                ].transform(transform_fn)
                irs_df["scale"] = irs_df["mixture_center"] / irs_df["irs_value"]
            elif irs_scope.lower() == "two_stage":
                transform_fn = "median" if stat_fn == "median" else "mean"
                irs_df["mixture_center"] = irs_df.groupby("mixture")[
                    "irs_value"
                ].transform(transform_fn)
                irs_df["scale_stage1"] = (
                    irs_df["mixture_center"] / irs_df["irs_value"]
                )
                mixture_center_df = irs_df[["mixture", "mixture_center"]].drop_duplicates()
                if stat_fn == "median":
                    global_center = mixture_center_df["mixture_center"].median()
                else:
                    global_center = mixture_center_df["mixture_center"].mean()
                mixture_center_df["scale_stage2"] = (
                    global_center / mixture_center_df["mixture_center"]
                )
                irs_df = irs_df.merge(
                    mixture_center_df, on="mixture", how="left", suffixes=("", "_mix")
                )
                irs_df["scale"] = irs_df["scale_stage1"] * irs_df["scale_stage2"]
            else:
                # Global scope
                if stat_fn == "median":
                    global_center = irs_df["irs_value"].median()
                else:
                    global_center = irs_df["irs_value"].mean()
                irs_df["scale"] = global_center / irs_df["irs_value"]

            irs_scale_by_techrep = dict(
                zip(irs_df["techrep_guess"].tolist(), irs_df["scale"].tolist())
            )

        return irs_scale_by_techrep


@log_execution_time(logger)
def peptide_normalization(
    parquet: str,
    sdrf: str,
    min_aa: int,
    min_unique: int,
    remove_ids: str,
    remove_decoy_contaminants: bool,
    remove_low_frequency_peptides: bool,
    output: str,
    skip_normalization: bool,
    nmethod: str,
    pnmethod: str,
    log2: bool,
    save_parquet: bool,
    irs_channel: str = None,
    irs_autodetect_regex: str = None,
    irs_stat: str = "median",
    irs_scope: str = "global",
    aggregation_level: str = AGGREGATION_LEVEL_SAMPLE,
    filter_config: Optional["PreprocessingFilterConfig"] = None,
) -> None:
    """
    Perform peptide normalization on a proteomics dataset.

    Parameters
    ----------
    parquet : str
        Path to the Parquet file containing the dataset.
    sdrf : str
        Path to the SDRF file for quantification details.
    min_aa : int
        Minimum number of amino acids required for peptides.
    min_unique : int
        Minimum number of unique peptides per protein.
    remove_ids : str
        Path to a file with protein IDs to remove.
    remove_decoy_contaminants : bool
        Whether to remove decoys and contaminants.
    remove_low_frequency_peptides : bool
        Whether to remove low-frequency peptides.
    output : str
        Path to the output file for saving results.
    skip_normalization : bool
        Whether to skip normalization steps.
    nmethod : str
        Method for feature-level normalization.
    pnmethod : str
        Method for peptide-level normalization.
    log2 : bool
        Whether to apply log2 transformation to intensities.
    save_parquet : bool
        Whether to save results in Parquet format.
    irs_channel : str, optional
        IRS reference channel label for TMT/ITRAQ normalization.
    irs_autodetect_regex : str, optional
        Regex to autodetect pooled/reference sample in SDRF.
    irs_stat : str, optional
        Statistic for IRS per-run metric (median or mean).
    irs_scope : str, optional
        IRS scaling scope (global, by_mixture, or two_stage).
    aggregation_level : str, optional
        Level at which to aggregate intensities. Options:
        - "sample": Aggregate at sample level (default, original behavior)
        - "run": Aggregate at run level, preserving run information for
          downstream quantification. This is useful when you want to perform
          quantification (e.g., MaxLFQ) at the run level first, similar to
          DIA-NN's approach.
    filter_config : PreprocessingFilterConfig, optional
        Configuration for preprocessing filters. If provided, filters will be
        applied to the data during processing.
    """

    if os.path.exists(output):
        raise FileExistsError("The output file already exists.")

    if parquet is None:
        raise FileNotFoundError("The file does not exist.")

    feature_normalization = FeatureNormalizationMethod.from_str(nmethod)
    peptide_normalized = PeptideNormalizationMethod.from_str(pnmethod)

    logger.info("Loading data from %s...", parquet)

    # Create filter builder for pre-computations (median maps, peptide frequencies)
    # This ensures normalization factors are computed on clean data without
    # contaminants, decoys, and other artifacts
    if filter_config is not None and filter_config.enabled:
        filter_builder = SQLFilterBuilder(
            remove_contaminants=(
                filter_config.protein.remove_contaminants
                or filter_config.protein.remove_decoys
            ),
            contaminant_patterns=filter_config.protein.contaminant_patterns,
            min_intensity=filter_config.intensity.min_intensity,
            min_peptide_length=min_aa,
            require_unique=True,
        )
    else:
        filter_builder = SQLFilterBuilder(
            remove_contaminants=remove_decoy_contaminants,
            contaminant_patterns=["CONTAMINANT", "ENTRAP", "DECOY"],
            min_intensity=0.0,
            min_peptide_length=min_aa,
            require_unique=True,
        )

    feature = Feature(parquet, filter_builder=filter_builder)

    if sdrf:
        feature.enrich_with_sdrf(sdrf)
        technical_repetitions, label, sample_names, choice = analyse_sdrf(sdrf)
    else:
        technical_repetitions, label, sample_names, choice = feature.experimental_inference

    if remove_low_frequency_peptides:
        low_frequency_peptides = feature.get_low_frequency_peptides()

    med_map = {}
    if not skip_normalization and peptide_normalized == PeptideNormalizationMethod.GlobalMedian:
        med_map = feature.get_median_map()
    elif (
        not skip_normalization and peptide_normalized == PeptideNormalizationMethod.ConditionMedian
    ):
        med_map = feature.get_median_map_to_condition()

    # Incremental CSV writing
    write_csv = True
    if write_csv:
        write_csv_task = WriteCSVTask(output)
        write_csv_task.start()

    # Incremental Parquet writing
    if save_parquet:
        writer_parquet_task = WriteParquetTask(output)
        writer_parquet_task.start()

    # IRS normalization pre-computation
    irs_scale_by_techrep: dict[int, float] = {}
    try:
        if label in (QuantificationCategory.TMT, QuantificationCategory.ITRAQ):
            if irs_channel is None and irs_autodetect_regex and sdrf:
                sdrf_df = pd.read_csv(sdrf, sep="\t")
                sdrf_df.columns = [i.lower() for i in sdrf_df.columns]
                ref_mask = sdrf_df["source name"].str.contains(irs_autodetect_regex, case=False, na=False)
                ref_labels = sdrf_df.loc[ref_mask, "comment[label]"]
                if not ref_labels.empty:
                    irs_channel = ref_labels.mode().iloc[0]
                else:
                    logger.warning("IRS autodetect regex '%s' found no pooled sample; skipping IRS.", irs_autodetect_regex)

            if irs_channel is not None:
                irs_scale_by_techrep = feature.get_irs_scaling_factors(
                    irs_channel=irs_channel,
                    irs_stat=irs_stat,
                    irs_scope=irs_scope,
                )
                if not irs_scale_by_techrep:
                    logger.warning(
                        "IRS channel '%s' not found in dataset; skipping IRS normalization.",
                        irs_channel,
                    )
    except Exception as e:
        logger.warning("IRS normalization pre-computation failed: %s", e)

    # Initialize filter pipeline if config provided
    filter_pipeline = None
    if filter_config is not None and filter_config.enabled:
        from mokume.preprocessing.filters import get_filter_pipeline

        filter_pipeline = get_filter_pipeline(filter_config)
        if len(filter_pipeline) > 0:
            logger.info(
                "Filter pipeline '%s' initialized with %d filters",
                filter_config.name,
                len(filter_pipeline),
            )

    for samples, df in feature.iter_samples():
        df.dropna(subset=["pg_accessions"], inplace=True)
        for sample in samples:
            logger.info("%s: Data preprocessing...", str(sample).upper())
            dataset_df = df[df["sample_accession"] == sample].copy()

            dataset_df = dataset_df[dataset_df["unique"] == 1]
            dataset_df = dataset_df[PARQUET_COLUMNS]

            dataset_df = reformat_quantms_feature_table_quant_labels(dataset_df, label, choice)

            dataset_df = apply_initial_filtering(dataset_df, min_aa, aggregation_level)

            dataset_df = dataset_df.groupby(PROTEIN_NAME).filter(
                lambda x: len(set(x[PEPTIDE_CANONICAL])) >= min_unique
            )

            if remove_decoy_contaminants:
                dataset_df = remove_contaminants_entrapments_decoys(dataset_df)

            if remove_ids is not None:
                dataset_df = remove_protein_by_ids(dataset_df, remove_ids)
            dataset_df.rename(columns={INTENSITY: NORM_INTENSITY}, inplace=True)

            # Apply filter pipeline if configured
            if filter_pipeline is not None and len(filter_pipeline) > 0:
                initial_count = len(dataset_df)
                dataset_df, filter_results = filter_pipeline.apply(dataset_df)
                if filter_config.log_filtered_counts:
                    for result in filter_results:
                        if result.removed_count > 0:
                            logger.info(
                                "%s: %s removed %d items (%.1f%%)",
                                str(sample).upper(),
                                result.filter_name,
                                result.removed_count,
                                result.removal_rate * 100,
                            )
                    total_removed = initial_count - len(dataset_df)
                    if total_removed > 0:
                        logger.info(
                            "%s: Filter pipeline removed %d/%d items total",
                            str(sample).upper(),
                            total_removed,
                            initial_count,
                        )

            if (
                not skip_normalization
                and nmethod not in ("none", None)
                and technical_repetitions > 1
            ):
                start_time = time.time()
                logger.info(
                    "%s: Normalizing intensities of features using method %s...",
                    str(sample).upper(),
                    nmethod,
                )
                dataset_df = feature_normalization(dataset_df, technical_repetitions)
                elapsed = time.time() - start_time
                logger.info(
                    "%s: Number of features after normalization: %d (completed in %.2f seconds)",
                    str(sample).upper(),
                    len(dataset_df.index),
                    elapsed,
                )

            if irs_scale_by_techrep:
                if TECHREPLICATE in dataset_df.columns:
                    scale_series = dataset_df[TECHREPLICATE].map(irs_scale_by_techrep).fillna(1.0)
                    dataset_df.loc[:, NORM_INTENSITY] = dataset_df[NORM_INTENSITY] * scale_series
                else:
                    logger.warning(
                        "%s: TECHREPLICATE column not present; cannot apply IRS scaling to sample %s",
                        str(sample).upper(),
                        sample,
                    )

            dataset_df = get_peptidoform_normalize_intensities(dataset_df)
            logger.info(
                "%s: Number of peptides after peptidoform selection: %d",
                str(sample).upper(),
                len(dataset_df.index),
            )

            if len(dataset_df[FRACTION].unique().tolist()) > 1:
                start_time = time.time()
                logger.info("%s: Merging features across fractions...", str(sample).upper())
                dataset_df = merge_fractions(dataset_df)
                elapsed = time.time() - start_time
                logger.info(
                    "%s: Number of features after merging fractions: %d (completed in %.2f seconds)",
                    str(sample).upper(),
                    len(dataset_df.index),
                    elapsed,
                )

            if not skip_normalization:
                dataset_df = peptide_normalized(dataset_df, sample, med_map)

            if remove_low_frequency_peptides and len(sample_names) > 1:
                dataset_df.set_index([PROTEIN_NAME, PEPTIDE_CANONICAL], drop=True, inplace=True)
                dataset_df = dataset_df[
                    ~dataset_df.index.isin(low_frequency_peptides)
                ].reset_index()
                logger.info(
                    "%s: Peptides after removing low frequency peptides: %d",
                    str(sample).upper(),
                    len(dataset_df.index),
                )

            start_time = time.time()
            logger.info("%s: Summing all peptidoforms per %s...", str(sample).upper(), aggregation_level)
            dataset_df = sum_peptidoform_intensities(dataset_df, aggregation_level)
            elapsed = time.time() - start_time
            logger.info(
                "%s: Number of peptides after selection: %d (completed in %.2f seconds)",
                str(sample).upper(),
                len(dataset_df.index),
                elapsed,
            )

            if log2:
                dataset_df[NORM_INTENSITY] = np.log2(dataset_df[NORM_INTENSITY])

            logger.info("%s: Saving the normalized peptide intensities...", str(sample).upper())

            if save_parquet:
                writer_parquet_task.write(dataset_df)
            if write_csv:
                write_csv_task.write(dataset_df)

    if write_csv:
        write_csv_task.close()
    if save_parquet:
        writer_parquet_task.close()
