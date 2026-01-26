"""
CLI command for t-SNE visualization.
"""

import glob
import logging

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mokume.core.constants import PROTEIN_NAME, SAMPLE_ID, IBAQ_LOG


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

    plt.rcParams["figure.figsize"] = (12, 6)
    fig, ax = plt.subplots()
    xi = np.arange(1, n_components + 1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker="o", linestyle="--", color="b")
    plt.xlabel("Number of Components")
    plt.xticks(np.arange(0, n_components, step=1))
    plt.ylabel("Cumulative variance (%)")
    plt.title("The number of components needed to explain variance")

    plt.axhline(y=0.95, color="r", linestyle="-")
    plt.text(0.5, 0.85, "95% cut-off threshold", color="red", fontsize=16)
    ax.grid(axis="x")
    plt.show()

    return df_pca


def compute_tsne(df_pca, n_components=2, perplexity=30, learning_rate=200, n_iter=2000):
    """Compute t-SNE components from PCA components."""
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
    )
    tsne_results = tsne.fit_transform(np.asarray(df_pca))
    tsne_cols = [f"tSNE{i + 1}" for i in range(n_components)]
    df_tsne = pd.DataFrame(data=tsne_results, columns=tsne_cols)
    df_tsne.index = df_pca.index
    return df_tsne


def plot_tsne(df, x_col, y_col, hue_col, file_name):
    """Generate and save a t-SNE scatter plot from a DataFrame."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df, ax=ax, markers=["o", "+", "x"])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col} with {hue_col} information")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=8)
    plt.subplots_adjust(right=0.8)
    plt.savefig(file_name)


@click.command()
@click.option("-f", "--folder", help="Folder that contains all the protein files", required=True)
@click.option(
    "-o",
    "--pattern",
    help="Protein file pattern",
    required=False,
    default="proteins.tsv",
)
def tsne_visualization(folder: str, pattern: str):
    """Generate a t-SNE visualization for protein data from specified files."""
    files = glob.glob(f"{folder}/*{pattern}")
    dfs = []

    for f in files:
        reanalysis = (f.split("/")[-1].split("_")[0]).replace("-proteins.tsv", "")
        dfs += [
            pd.read_csv(f, usecols=[PROTEIN_NAME, SAMPLE_ID, IBAQ_LOG], sep=",").assign(
                reanalysis=reanalysis
            )
        ]

    total_proteins = pd.concat(dfs, ignore_index=True)

    normalize_df = pd.pivot_table(
        total_proteins,
        index=[SAMPLE_ID, "reanalysis"],
        columns=PROTEIN_NAME,
        values=IBAQ_LOG,
    )
    normalize_df = normalize_df.fillna(0)
    df_pca = compute_pca(normalize_df, n_components=30)
    df_tsne = compute_tsne(df_pca)

    batch = df_tsne.index.get_level_values("reanalysis").tolist()
    df_tsne["batch"] = batch

    plot_tsne(df_tsne, "tSNE1", "tSNE2", "batch", "5.tsne_plot_with_batch_information.pdf")
    logger.info(total_proteins.shape)


if __name__ == "__main__":
    tsne_visualization()
