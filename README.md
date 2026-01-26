# mokume

[![Python application](https://github.com/bigbio/mokume/actions/workflows/python-app.yml/badge.svg)](https://github.com/bigbio/mokume/actions/workflows/python-app.yml)
[![Upload Python Package](https://github.com/bigbio/mokume/actions/workflows/python-publish.yml/badge.svg)](https://github.com/bigbio/mokume/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/mokume.svg)](https://badge.fury.io/py/mokume)
![PyPI - Downloads](https://img.shields.io/pypi/dm/mokume)

## Why "mokume"?

The name comes from [mokume-gane](https://en.wikipedia.org/wiki/Mokume-gane) (木目金), a Japanese metalworking technique that fuses multiple metal layers into distinctive patterns - similar to how this library melds peptide intensities into unified protein expression profiles.

## Overview

**mokume** is a comprehensive proteomics quantification library that supports multiple protein quantification methods including iBAQ, Top3, TopN, MaxLFQ, and DirectLFQ. It provides feature/peptide normalization, batch correction, and various summarization strategies for the quantms ecosystem. This library is an evolution of [ibaqpy](https://github.com/bigbio/ibaqpy), extended to support a broader range of protein quantification methods beyond iBAQ.

## Installation

```bash
pip install mokume
```

With optional DirectLFQ support:

```bash
pip install mokume[directlfq]
```

Or install from source:

```bash
git clone https://github.com/bigbio/mokume
cd mokume
pip install .
```

Using conda:

```bash
mamba env create -f environment.yaml
conda activate mokume
pip install .
```

## Library Structure

```
mokume/
├── core/                    # Core utilities and constants
│   ├── constants.py         # Column names, mappings, utility functions
│   ├── logger.py            # Logging utilities
│   └── write_queue.py       # Async file writing
│
├── model/                   # Data models and enums
│   ├── labeling.py          # TMT/iTRAQ/LFQ labeling types
│   ├── normalization.py     # Normalization method enums
│   ├── organism.py          # Organism metadata (histones, genome size)
│   ├── quantification.py    # Quantification method enums
│   └── summarization.py     # Summarization strategy enums
│
├── normalization/           # Normalization implementations
│   ├── feature.py           # Feature-level normalization
│   ├── peptide.py           # Peptide-level normalization pipeline
│   └── protein.py           # Protein-level normalization
│
├── quantification/          # Protein quantification methods
│   ├── base.py              # Abstract base class
│   ├── ibaq.py              # iBAQ implementation
│   ├── top3.py              # Top3 quantification
│   ├── topn.py              # TopN quantification
│   ├── maxlfq.py            # MaxLFQ algorithm (parallelized)
│   ├── directlfq.py         # DirectLFQ wrapper (optional)
│   └── all_peptides.py      # Sum of all peptides
│
├── summarization/           # Intensity summarization strategies
│   ├── base.py              # Abstract base class
│   ├── median.py            # Median summarization
│   ├── mean.py              # Mean summarization
│   └── sum.py               # Sum summarization
│
├── imputation/              # Missing value handling
│   └── methods.py           # Imputation implementations
│
├── postprocessing/          # Data reshaping and correction
│   ├── reshape.py           # Pivot operations (wide/long format)
│   ├── batch_correction.py  # ComBat batch correction
│   └── combiner.py          # Multi-file combining
│
├── io/                      # Input/Output utilities
│   ├── parquet.py           # Parquet/TSV reading, AnnData creation
│   └── fasta.py             # FASTA file handling
│
├── commands/                # CLI commands
│   ├── features2peptides.py # Feature to peptide conversion
│   ├── peptides2protein.py  # Protein quantification
│   ├── batch_correct.py     # Batch correction
│   └── visualize.py         # t-SNE visualization
│
└── data/                    # Static data resources
    ├── organisms.py         # Organism histone data
    └── organisms.json       # Organism registry
```

## Quantification Methods

| Method | Description | Requires FASTA | Class | Optional |
|--------|-------------|----------------|-------|----------|
| **iBAQ** | Intensity-Based Absolute Quantification | Yes | `peptides_to_protein()` | No |
| **Top3** | Average of 3 most intense peptides | No | `Top3Quantification` | No |
| **TopN** | Average of N most intense peptides | No | `TopNQuantification` | No |
| **MaxLFQ** | Delayed normalization with parallelization | No | `MaxLFQQuantification` | No |
| **DirectLFQ** | Intensity traces with hierarchical alignment | No | `DirectLFQQuantification` | Yes* |
| **Sum** | Sum of all peptide intensities | No | `AllPeptidesQuantification` | No |

*DirectLFQ requires optional install: `pip install mokume[directlfq]`

## CLI Usage

### Peptides to Protein Quantification

```bash
# Using iBAQ (default) - requires FASTA
mokume peptides2protein --method ibaq \
    -f proteome.fasta \
    -p peptides.csv \
    -o proteins-ibaq.tsv

# Using Top3 - no FASTA required
mokume peptides2protein --method top3 \
    -p peptides.csv \
    -o proteins-top3.tsv

# Using TopN with N=5
mokume peptides2protein --method topn --topn_n 5 \
    -p peptides.csv \
    -o proteins-top5.tsv

# Using MaxLFQ with parallelization
mokume peptides2protein --method maxlfq \
    --n_jobs 4 \
    -p peptides.csv \
    -o proteins-maxlfq.tsv

# Using DirectLFQ (requires: pip install mokume[directlfq])
mokume peptides2protein --method directlfq \
    -p peptides.csv \
    -o proteins-directlfq.tsv

# Using Sum of all peptides
mokume peptides2protein --method sum \
    -p peptides.csv \
    -o proteins-sum.tsv
```

### Full iBAQ with TPA and ProteomicRuler

```bash
mokume peptides2protein \
    -f proteome.fasta \
    -p peptides.csv \
    -e Trypsin \
    --normalize \
    --tpa \
    --ruler \
    --ploidy 2 \
    --cpc 200 \
    --organism human \
    --output proteins-ibaq.tsv \
    --verbose \
    --qc_report QC.pdf
```

### Features to Peptides

```bash
mokume features2peptides \
    -p features.parquet \
    -s experiment.sdrf.tsv \
    --remove_decoy_contaminants \
    --remove_low_frequency_peptides \
    --nmethod median \
    --pnmethod globalMedian \
    --output peptides-norm.csv
```

### Batch Correction

```bash
mokume correct-batches \
    -f ibaq_folder/ \
    -p "*ibaq.tsv" \
    -o corrected_ibaq.tsv \
    --export_anndata
```

### t-SNE Visualization

```bash
mokume tsne-visualization \
    -f protein_folder/ \
    -o proteins.tsv
```

## Python API

### Quantification Methods

```python
import pandas as pd
from mokume.quantification import (
    Top3Quantification,
    TopNQuantification,
    MaxLFQQuantification,
    AllPeptidesQuantification,
    get_quantification_method,
    is_directlfq_available,
    peptides_to_protein,  # iBAQ function
)

# Load peptide data
peptides = pd.read_csv("peptides.csv")

# --- Top3 Quantification ---
top3 = Top3Quantification()
result = top3.quantify(
    peptides,
    protein_column="ProteinName",
    peptide_column="PeptideSequence",
    intensity_column="NormIntensity",
    sample_column="SampleID",
)

# --- TopN Quantification (configurable N) ---
topn = TopNQuantification(n=5)
result = topn.quantify(peptides, protein_column="ProteinName", ...)

# --- MaxLFQ Quantification (parallelized, variance-guided) ---
maxlfq = MaxLFQQuantification(
    min_peptides=2,
    n_jobs=4,              # Use 4 parallel cores
    use_variance_guided=True,  # Smarter merging (inspired by DirectLFQ)
)
result = maxlfq.quantify(peptides, protein_column="ProteinName", ...)

# --- DirectLFQ Quantification (optional dependency) ---
if is_directlfq_available():
    from mokume.quantification import DirectLFQQuantification
    directlfq = DirectLFQQuantification(min_nonan=2)
    result = directlfq.quantify(peptides, protein_column="ProteinName", ...)

# --- Sum of All Peptides ---
sum_quant = AllPeptidesQuantification()
result = sum_quant.quantify(peptides, protein_column="ProteinName", ...)

# --- Factory Function ---
method = get_quantification_method("maxlfq", min_peptides=2, n_jobs=-1)
result = method.quantify(peptides, ...)

# --- Check available methods ---
from mokume.quantification import list_quantification_methods
print(list_quantification_methods())
# {'top3': True, 'topn': True, 'maxlfq': True, 'directlfq': False, 'sum': True}

# --- iBAQ with Full Pipeline ---
peptides_to_protein(
    fasta="proteome.fasta",
    peptides="peptides.csv",
    enzyme="Trypsin",
    normalize=True,
    tpa=True,
    ruler=True,
    ploidy=2,
    cpc=200,
    organism="human",
    output="proteins-ibaq.tsv",
    min_aa=7,
    max_aa=30,
    verbose=True,
    qc_report="QC.pdf",
)
```

### Normalization

```python
from mokume.normalization.peptide import peptide_normalization

# Full peptide normalization pipeline
peptide_normalization(
    parquet="features.parquet",
    sdrf="experiment.sdrf.tsv",
    min_aa=7,
    min_unique=2,
    remove_ids=None,
    remove_decoy_contaminants=True,
    remove_low_frequency_peptides=True,
    output="peptides-norm.csv",
    skip_normalization=False,
    nmethod="median",      # Feature normalization: mean, median, iqr, none
    pnmethod="globalMedian",  # Peptide normalization: globalMedian, conditionMedian, none
    log2=True,
    save_parquet=False,
)
```

### Batch Correction

```python
from mokume.io.parquet import combine_ibaq_tsv_files
from mokume.postprocessing.reshape import pivot_wider, pivot_longer
from mokume.postprocessing.batch_correction import apply_batch_correction

# Load and combine multiple TSV files
df = combine_ibaq_tsv_files("data/", pattern="*ibaq.tsv", sep="\t")

# Reshape to wide format (proteins x samples)
df_wide = pivot_wider(
    df,
    row_name="ProteinName",
    col_name="SampleID",
    values="Ibaq",
    fillna=True
)

# Extract batch IDs from sample names
import pandas as pd
batch_ids = [name.split("-")[0] for name in df_wide.columns]
batch_ids = pd.factorize(batch_ids)[0]

# Apply ComBat batch correction
df_corrected = apply_batch_correction(df_wide, list(batch_ids), kwargs={})

# Reshape back to long format
df_long = pivot_longer(
    df_corrected,
    row_name="ProteinName",
    col_name="SampleID",
    values="IbaqCorrected"
)
```

### Data Reshaping

```python
from mokume.postprocessing.reshape import (
    pivot_wider,
    pivot_longer,
    remove_samples_low_protein_number,
    remove_missing_values,
    describe_expression_metrics,
)

# Long to wide format
df_wide = pivot_wider(df, row_name="ProteinName", col_name="SampleID", values="Ibaq")

# Wide to long format
df_long = pivot_longer(df_wide, row_name="ProteinName", col_name="SampleID", values="Ibaq")

# Quality filtering
df_filtered = remove_samples_low_protein_number(df, min_protein_num=100)
df_filtered = remove_missing_values(df, missingness_percentage=20, expression_column="Ibaq")

# Get expression statistics
metrics = describe_expression_metrics(df)
```

### Creating AnnData Objects

```python
from mokume.io.parquet import create_anndata

# Create AnnData from long-format DataFrame
adata = create_anndata(
    df,
    obs_col="SampleID",           # Observation (sample) column
    var_col="ProteinName",        # Variable (protein) column
    value_col="Ibaq",             # Main data values
    layer_cols=["IbaqNorm", "IbaqLog", "IbaqBec"],  # Additional layers
    obs_metadata_cols=["Condition"],  # Sample metadata
    var_metadata_cols=["GeneName"],   # Protein metadata
)

# Save to h5ad
adata.write("proteins.h5ad")
```

### FASTA Handling

```python
from mokume.io.fasta import (
    load_fasta,
    digest_protein,
    extract_fasta,
    get_protein_molecular_weights,
)

# Load FASTA file
proteins = load_fasta("proteome.fasta")

# Digest a single protein sequence
peptides = digest_protein(
    sequence="MKWVTFISLLFLFSSAYS...",
    enzyme="Trypsin",
    min_aa=7,
    max_aa=30,
)

# Extract info for specific proteins
unique_peptide_counts, mw_dict, found_proteins = extract_fasta(
    fasta="proteome.fasta",
    enzyme="Trypsin",
    proteins=["P12345", "P67890"],
    min_aa=7,
    max_aa=30,
    tpa=True,
)

# Get molecular weights
mw_dict = get_protein_molecular_weights("proteome.fasta", ["P12345", "P67890"])
```

### Organism Metadata

```python
from mokume.model.organism import OrganismDescription

# Get available organisms
organisms = OrganismDescription.registered_organisms()
# ['human', 'mouse', 'yeast', 'drome', 'caeel', 'schpo']

# Get organism description
human = OrganismDescription.get("human")
print(human.genome_size)      # Genome size in base pairs
print(human.histone_entries)  # List of histone protein accessions
```

## Computed Values Reference

| Column | Description | Method |
|--------|-------------|--------|
| `Ibaq` | Total intensity / theoretical peptides | iBAQ |
| `IbaqNorm` | `ibaq / sum(ibaq)` per sample | iBAQ |
| `IbaqLog` | `10 + log10(IbaqNorm)` | iBAQ |
| `IbaqPpb` | `IbaqNorm * 100,000,000` | iBAQ |
| `IbaqBec` | Batch effect corrected | iBAQ + ComBat |
| `TPA` | `NormIntensity / MolecularWeight` | iBAQ |
| `CopyNumber` | Protein copies per cell | ProteomicRuler |
| `Concentration[nM]` | Protein concentration | ProteomicRuler |
| `Top3Intensity` | Average of top 3 peptides | Top3 |
| `Top{N}Intensity` | Average of top N peptides | TopN |
| `MaxLFQIntensity` | MaxLFQ algorithm result | MaxLFQ |
| `DirectLFQIntensity` | DirectLFQ intensity traces | DirectLFQ |
| `SumIntensity` | Sum of all peptides | Sum |

## Normalization Methods

### Feature-Level Normalization (`--nmethod`)

| Method | Description |
|--------|-------------|
| `median` | Normalize by median across MS runs |
| `mean` | Normalize by mean across MS runs |
| `iqr` | Normalize by interquartile range |
| `none` | Skip feature normalization |

### Peptide-Level Normalization (`--pnmethod`)

| Method | Description |
|--------|-------------|
| `globalMedian` | Adjust all samples to global median |
| `conditionMedian` | Adjust samples within each condition to median |
| `none` | Skip peptide normalization |

## Data Processing Pipeline

### Feature Processing (`features2peptides`)

1. Parse protein identifiers and retain unique peptides
2. Remove entries with empty intensity or condition
3. Filter peptides by minimum amino acids
4. Remove low-confidence proteins (< min_unique peptides)
5. Optionally remove decoys, contaminants, and specified proteins
6. Normalize at feature level between MS runs
7. Merge peptidoforms across fractions and technical replicates
8. Normalize at sample level
9. Remove low-frequency peptides
10. Assemble peptidoforms to peptides
11. Optional log2 transformation

### iBAQ Calculation

1. Load peptide intensity data
2. Extract protein info from FASTA (theoretical peptide counts, MW)
3. Group peptide intensities by protein, sample, and condition
4. Sum protein intensities within each group
5. Normalize by detected peptide count
6. Divide by theoretical peptide count
7. Optional: Calculate TPA, copy number, concentration

## Citation

> Zheng P, Audain E, Webel H, Dai C, Klein J, Hitz MP, Sachsenberg T, Bai M, Perez-Riverol Y. Ibaqpy: A scalable Python package for baseline quantification in proteomics leveraging SDRF metadata. J Proteomics. 2025;317:105440. doi: [10.1016/j.jprot.2025.105440](https://doi.org/10.1016/j.jprot.2025.105440).

> Wang H, Dai C, Pfeuffer J, Sachsenberg T, Sanchez A, Bai M, Perez-Riverol Y. Tissue-based absolute quantification using large-scale TMT and LFQ experiments. Proteomics. 2023;23(20):e2300188. doi: [10.1002/pmic.202300188](https://doi.org/10.1002/pmic.202300188).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

- [Julianus Pfeuffer](https://github.com/jpfeuffer)
- [Yasset Perez-Riverol](https://github.com/ypriverol)
- [Hong Wang](https://github.com/WangHong007)
- [Ping Zheng](https://github.com/zprobot)
- [Joshua Klein](https://github.com/mobiusklein)
- [Enrique Audain](https://github.com/enriquea)
