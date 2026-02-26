# Cell2Map

Cell2Map is an unsupervised deep learning method that integrates single-cell RNA sequencing (scRNA-seq) and spatially resolved transcriptomics (SRT) to map individual cells to spatial locations within tissue sections. While scRNA-seq provides genome-wide expression profiles at single-cell resolution but loses spatial context, SRT preserves spatial information but typically lacks single-cell resolution and complete transcriptome coverage. Cell2Map bridges this gap by assigning individual cells to SRT spots using a graph attention autoencoder (GATE) equipped with a specially designed multi-term objective function that jointly optimizes expression-based, density-based, and embedding-level similarity and distance constraints.

## Scientific highlights

- Superior mapping accuracy: outperforms Celloc, CytoSPACE and Tangram on simulated benchmarks (mouse cerebellum and hippocampus) across varying noise levels and spot cell densities.
- Multi-term optimization: jointly optimizes expression similarity, density constraints and spatial distance penalties through a differentiable mapping matrix.
- Cancer heterogeneity resolution: accurately localizes tumor subclones and separates normal epithelial cells from ductal carcinoma in situ regions in real applications.
- Enhanced microenvironment reconstruction: more faithfully reconstructs tumor microenvironments and immune-cell localization compared to competing approaches.
- High sensitivity: consistently achieves higher sensitivity with fewer false positives across breast cancer and myocardial infarction (MI) datasets, in close agreement with histological annotations.

## Toolkit overview

Cell2Map provides a complete research-oriented toolkit for:

- Data integration: unified preprocessing pipeline for scRNA-seq and SRT datasets.
- Joint embedding: graph attention autoencoder (GATE) for learning shared representations.
- Probabilistic mapping: differentiable optimization of cell-to-spot assignments.
- Validation: comprehensive evaluation metrics.

## Repository layout

Top-level files and directories you will use most:

- `Cell2Map/` — core Python package:
  - `autoencoder.py` — GATE implementation, spatial graph builders, AnnData ↔ PyG converters
  - `common.py` — I/O helpers (`read_data`), evaluation utilities, clustering metrics
  - `map_utils.py` — high-level workflow (`process_adatas`, `map_cell_to_space`)
  - `map_optimizer.py` — `Mapper` class optimizing the mapping matrix M
  - `utils.py` — numeric helpers (distances, KL divergence, conversions)
  - `loss.py` — loss modules (activations, multi-term objectives)

- `data/` — example datasets (Cerebellum, Hippocampus, MI, breast cancer)
- `test/` — experimental notebooks validating the pipeline on real and simulated data
- `requirements*.txt` — dependency specifications

## Installation

Recommended Python: 3.8 or 3.9. Full GPU acceleration requires PyTorch and PyTorch-Geometric (PyG).

PowerShell (CPU example):

```powershell
# create environment
conda create -n cell2map python=3.9 -y; conda activate cell2map
# install core packages
pip install -r requirements.txt
# install torch CPU wheels (example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# install PyG following the official guide (platform & torch-version specific)
```

If you use GPU, install the matching torch wheel for your CUDA version and then follow PyG's install instructions.

## Quick start

Basic flow (Python):

```python
from Cell2Map import common, map_utils, autoencoder

# 1) Read data
sc_adata, spatial_adata, coords = common.read_data(
    scRNA_path, cell_type_path, st_path, coordinates_path, celltype_col
)

# 2) Preprocess
sc_adata, spatial_adata = map_utils.process_adatas(sc_adata, spatial_adata)

# 3) Compute embeddings (GATE) -- requires PyG
sc_adata, spatial_adata = autoencoder.embedding_feature(
    sc_adata, spatial_adata, k_cutoff=7
)

# 4) Map cells to spots
adata_map = map_utils.map_cell_to_space(
    sc_adata, spatial_adata, learning_rate=0.005, num_epochs=1500, device='cpu'
)
```

## Experimental notebooks

The `test/` folder contains notebooks used to run and validate the pipeline. These notebooks are the project's test files and show dataset-specific usage:

- `test/Run_DCIS1.ipynb`
- `test/Run_DCIS1_with_STsample1.ipynb`
- `test/Run_DCIS1_with_STsample2.ipynb`
- `test/Run_DCIS2.ipynb`
- `test/Run_HER2+_with_STsample1.ipynb`
- `test/Run_HER2+_with_STsample2.ipynb`
- `test/Run_MI.ipynb`
- `test/Run_simulated_mouse_cerebellum.ipynb`
- `test/Run_simulated_mouse_hippocampus.ipynb`

Open any of these notebooks for step-by-step demos and experimental settings used by the original authors.

## API (concise reference)

- `Cell2Map.common`:
  - `read_data(scRNA_path, cell_type_path, st_path, coordinates_path, celltype_col)` — read and construct AnnData objects (supports CSV, .h5ad and some .h5).
  - `estimate_cell_number_RNA_reads(st_data, mean_cell_numbers)` — estimate cells per spot from RNA reads.

- `Cell2Map.autoencoder`:
  - `embedding_feature(sc_adata, spatial_adata, k_cutoff=7, self_loop=True)` — train GATE and store embeddings in `.obsm['embedding']` and reconstructions in `.obsm['STAGATE_ReX']`.
  - `Cal_Spatial_Net`, `Cal_Spatial_Net_Radius` — build kNN or radius graphs and save adjacency in `adata.uns['Spatial_Net']`.
  - `Transfer_pytorch_Data(adata)` — convert AnnData + Spatial_Net into PyG `Data`.

- `Cell2Map.map_utils`:
  - `process_adatas(sc_adata, spatial_adata, genes=None, gene_to_lowercase=True)` — filtering, normalization, PCA and gene intersection.
  - `map_cell_to_space(sc_adata, spatial_adata, ...)` — high-level mapping routine that constructs a `Mapper` and returns `adata_map` with mapping in `adata_map.X`.

- `Cell2Map.map_optimizer`:
  - `Mapper` — optimizer for mapping matrix M. Use `Mapper.train(...)` to optimize. Hyperparameters tune expression vs spatial losses, KL weight and distance penalties.

- `Cell2Map.utils` and `Cell2Map.loss`:
  - Numeric helpers (distances, KL divergence, conversions) and loss modules (MeanAct, DispAct, exp_loss).

## Data formats

- Single-cell: AnnData (.h5ad) or CSV (genes x cells). For CSV, rows are genes and columns are cell IDs.
- Spatial: AnnData (.h5ad), CSV, or simple matrix; coordinates passed separately as CSV (X, Y) unless included in `.obsm`.
- Cell-type labels: CSV with a column specified by `celltype_col`.

## Tips & notes

- Gene consistency: ensure nomenclature alignment between modalities.
- GPU acceleration: required for GATE training; CPU fallback uses PCA.
- Reproducibility: utilize `seed_everything()` and `set_seed()` utilities.
- Validation: always validate against histological ground truth when available.

## Development & contributing

Contributions welcome for:

- Enhanced spatial graph construction methods
- Additional loss terms for the multi-term objective
- Support for emerging SRT technologies
- Visualization and interactive analysis tools

Please add unit tests and update `requirements-pinned.txt` for reproducibility.

## Contact

Open an issue in the repository for questions or feature requests.
