"""
Peptide-level normalization implementations.

This module provides functions for peptide-level normalization including
feature normalization and the main peptide_normalization function.
"""

import os
import re
import time
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

    if data_df[RUN].str.contains("_").all():
        data_df[TECHREPLICATE] = data_df[RUN].str.split("_").str.get(1)
        data_df[TECHREPLICATE] = data_df[TECHREPLICATE].astype("int")
    else:
        data_df[TECHREPLICATE] = data_df[RUN].astype("int")

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
    """

    labels: Optional[list[str]]
    label: Optional[QuantificationCategory]
    choice: Optional[IsobaricLabel]
    technical_repetitions: Optional[int]

    def __init__(self, database_path: str):
        if os.path.exists(database_path):
            self.parquet_db = duckdb.connect()
            self.parquet_db = self.parquet_db.execute(
                "CREATE VIEW parquet_db AS SELECT * FROM parquet_scan('{}')".format(database_path)
            )
            self.samples = self.get_unique_samples()
        else:
            raise FileNotFoundError(f"the file {database_path} does not exist.")

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

    @property
    def low_frequency_peptides(self, percentage=0.2) -> tuple:
        """Identifies peptides that occur with low frequency across samples."""
        f_table = self.parquet_db.sql(
            """
            SELECT "sequence", "pg_accessions", COUNT(DISTINCT sample_accession) as "count" from parquet_db
            GROUP BY "sequence","pg_accessions"
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
        """Retrieves a list of unique technical repetition identifiers."""
        unique = self.parquet_db.sql("SELECT DISTINCT run FROM parquet_db").df()
        try:
            if unique["run"].str.contains("_").all():
                unique["run"] = unique["run"].str.split("_").str.get(1)
                unique["run"] = unique["run"].astype("int")
            else:
                unique["run"] = unique["run"].astype("int")
        except ValueError as e:
            raise ValueError(
                f"Some errors occurred when getting technical repetitions: {e}"
            ) from e

        return unique["run"].tolist()

    def get_median_map(self) -> dict[str, float]:
        """Computes a median intensity map for samples."""
        med_map: dict[str, float] = {}
        for _, batch_df in self.iter_samples(1000, ["sample_accession", "intensity"]):
            meds = batch_df.groupby(["sample_accession"])["intensity"].median()
            med_map.update(meds.to_dict())
        global_med = np.median([med for med in med_map.values()])
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
        """Computes a median intensity map for each experimental condition."""
        med_map = {}
        for cons, batch_df in self.iter_conditions(
            1000, ["condition", "sample_accession", "intensity"]
        ):
            for con in cons:
                meds = (
                    batch_df[batch_df["condition"] == con]
                    .groupby(["sample_accession"])["intensity"]
                    .median()
                )
                meds = meds / meds.mean()
                med_map[con] = meds.to_dict()
        return med_map


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
    feature = Feature(parquet)

    if sdrf:
        technical_repetitions, label, sample_names, choice = analyse_sdrf(sdrf)
    else:
        technical_repetitions, label, sample_names, choice = feature.experimental_inference

    if remove_low_frequency_peptides:
        low_frequency_peptides = feature.low_frequency_peptides

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
                stat_fn = "median" if (irs_stat or "").lower() == "median" else "avg"
                irs_df = feature.parquet_db.sql(
                    f"""
                    SELECT run, {stat_fn}(intensity) as irs_value, mixture, techreplicate as techrep_guess
                    FROM (
                        SELECT *,
                               CASE WHEN position('_' in run) > 0 THEN CAST(split_part(run, '_', 2) AS INTEGER)
                                    ELSE CAST(run AS INTEGER) END AS techreplicate
                        FROM parquet_db
                        WHERE channel = '{irs_channel}'
                    )
                    GROUP BY run, mixture, techrep_guess
                    """
                ).df()
                if len(irs_df.index) > 0:
                    irs_df = irs_df[irs_df["irs_value"] > 0]

                    if irs_scope.lower() == "by_mixture":
                        irs_df["mixture_center"] = irs_df.groupby("mixture")["irs_value"].transform("median" if stat_fn == "median" else "mean")
                        irs_df["scale"] = irs_df["mixture_center"] / irs_df["irs_value"]
                    elif irs_scope.lower() == "two_stage":
                        irs_df["mixture_center"] = irs_df.groupby("mixture")["irs_value"].transform("median" if stat_fn == "median" else "mean")
                        irs_df["scale_stage1"] = irs_df["mixture_center"] / irs_df["irs_value"]
                        mixture_center_df = irs_df[["mixture", "mixture_center"]].drop_duplicates()
                        global_center = (mixture_center_df["mixture_center"].median() if stat_fn == "median" else mixture_center_df["mixture_center"].mean())
                        mixture_center_df["scale_stage2"] = global_center / mixture_center_df["mixture_center"]
                        irs_df = irs_df.merge(mixture_center_df, on="mixture", how="left", suffixes=("", "_mix"))
                        irs_df["scale"] = irs_df["scale_stage1"] * irs_df["scale_stage2"]
                    else:
                        global_center = (irs_df["irs_value"].median() if stat_fn == "median" else irs_df["irs_value"].mean())
                        irs_df["scale"] = global_center / irs_df["irs_value"]

                    irs_scale_by_techrep = dict(zip(irs_df["techrep_guess"].tolist(), irs_df["scale"].tolist()))
                else:
                    logger.warning(
                        "IRS channel '%s' not found in dataset; skipping IRS normalization.", irs_channel,
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
