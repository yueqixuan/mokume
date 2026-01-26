"""
Batch correction utilities for the mokume package.
"""

import logging
import warnings
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings(
    "ignore", category=PendingDeprecationWarning, module="numpy.matrixlib.defmatrix"
)

from sklearn.cluster._hdbscan import hdbscan
from sklearn.decomposition import PCA

from mokume.core.constants import IBAQ_NORMALIZED, SAMPLE_ID, PROTEIN_NAME

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def compute_pca(df, n_components=5) -> pd.DataFrame:
    """Compute principal components for a given dataframe."""
    pca = PCA(n_components=n_components)
    pca.fit(df)
    df_pca = pca.transform(df)
    df_pca = pd.DataFrame(
        df_pca, index=df.index, columns=[f"PC{i}" for i in range(1, n_components + 1)]
    )
    return df_pca


def get_batch_info_from_sample_names(sample_list: List[str]) -> List[int]:
    """Get batch indices from sample names."""
    samples = [s.split("-")[0] for s in sample_list]
    batches = list(set(samples))
    index = {i: batches.index(i) for i in batches}
    return [index[i] for i in samples]


def remove_single_sample_batches(df: pd.DataFrame, batch: list) -> pd.DataFrame:
    """Remove batches with only one sample."""
    batch_dict = dict(zip(df.columns, batch))
    single_sample_batch = [
        k for k, v in batch_dict.items() if list(batch_dict.values()).count(v) == 1
    ]
    df_single_batches_removed = df.drop(single_sample_batch, axis=1)
    return df_single_batches_removed


def plot_pca(
    df_pca,
    output_file,
    x_col="PC1",
    y_col="PC2",
    hue_col="batch",
    palette="Set2",
    title="PCA plot",
    figsize=(8, 6),
):
    """Plot a PCA scatter plot and save it to a file."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df_pca, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")


class TooFewSamplesInBatch(ValueError):
    def __init__(self, batches):
        super().__init__(
            f"Batches must contain at least two samples, the following batch factors did not: {batches}"
        )


def apply_batch_correction(
    df: pd.DataFrame,
    batch: List[int],
    covs: Optional[List[int]] = None,
    kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Apply batch correction using pycombat."""
    if kwargs is None:
        kwargs = {}

    if len(df.columns) != len(batch):
        raise ValueError(
            f"The number of samples should match the number of batch "
            f"indices. There were {len(batch)} batch indices and {len(df.columns)} samples"
        )

    if any([batch.count(i) < 2 for i in set(batch)]):
        short_batches = [i for i in set(batch) if batch.count(i) < 2]
        raise TooFewSamplesInBatch(short_batches)

    if covs:
        if len(df.columns) != len(covs):
            raise ValueError(
                f"The number of samples should match the number of covariates. "
                f"There were {len(covs)} batch indices and {len(df.columns)} samples"
            )

    from inmoose.pycombat import pycombat_norm

    df_co = pycombat_norm(counts=df, batch=batch, covar_mod=covs, **kwargs)
    return df_co


def find_clusters(df, min_cluster_size, min_samples) -> pd.DataFrame:
    """Compute clusters for a given dataframe using HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=True,
        cluster_selection_epsilon=0.01,
    )
    clusterer.fit(df)
    df["cluster"] = clusterer.labels_
    return df


def iterative_outlier_removal(
    df: pd.DataFrame,
    batch: List[int],
    n_components: int = 5,
    min_cluster_size: int = 10,
    min_samples: int = 10,
    n_iter: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Iteratively remove outliers using PCA and HDBSCAN clustering."""
    batch_dict = dict(zip(df.columns, batch))

    for i in range(n_iter):
        logger.info("Running iteration: {}".format(i + 1))

        df_pca = compute_pca(df.T, n_components=n_components)
        df_clusters = find_clusters(
            df_pca, min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        logger.info(df_clusters)

        outliers = df_clusters[df_clusters["cluster"] == -1].index.tolist()
        df_filtered_outliers = df.drop(outliers, axis=1)
        logger.info(f"Number of outliers in iteration {i + 1}: {len(outliers)}")
        logger.info(f"Outliers in iteration {i + 1}: {str(outliers)}")

        batch_dict = {col: batch_dict[col] for col in df_filtered_outliers.columns}
        df = df_filtered_outliers

        if verbose:
            plot_pca(
                df_clusters,
                output_file=f"iterative_outlier_removal_{i + 1}.png",
                x_col="PC1",
                y_col="PC2",
                hue_col="cluster",
                title=f"Iteration {i + 1}: Number of outliers: {len(outliers)}",
            )

        if len(outliers) == 0:
            break

    return df
