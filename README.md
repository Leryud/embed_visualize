A Python package for text embedding and visualization using Sentence Transformers, UMAP, HDBSCAN, and Plotly.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Embedding Texts](#embedding-texts)
  - [Clustering](#clustering)
  - [Visualization](#visualization)
- [Configuration](#configuration)
- [Features](#features)
- [Limitations](#limitations)

## Installation

1. Clone this repository:

```bash
git clone git@github.com:Leryud/embed_vis.git
cd embed_vis
```

2. Install dependencies using Poetry:

```bash
poetry install
```

## Usage

### Embedding Texts

The `Embedder` class in `src/embed.py` is responsible for embedding text corpora using SentenceTransformer models. It supports batch processing and filtering based on various criteria.

Example usage:

```python
from src.embed import Embedder
from src.utils import filter_by_date

embedder = Embedder(config)
filename = "mastodon_toots.jsonl"
batch_size = 32

texts, embeddings = embedder.embed_corpus(filename, filter_by_date, batch_size)
```

### Clustering

The `ClusteringPipeline` class in `src/clustering.py` reduces the dimensionality of text embeddings using UMAP and performs clustering with HDBSCAN.

Example usage:

```python
from src.clustering import ClusteringPipeline

clustering_pipeline = ClusteringPipeline(clustering_conf)
clusters, labels = clustering_pipeline.process(texts, embeddings.cpu().tolist(), view="3d")
```

### Visualization

The `ClusteringPipeline` class provides visualization methods for 2D and 3D plots using Plotly.

## Configuration

The configuration is handled in `src/config.py`. It includes settings for the embedding model, device, half precision, base path, text key, UMAP parameters, and HDBSCAN parameters.

## Features

- Text embedding using Sentence Transformers
- Batch processing with adjustable batch size
- Filtering of text corpora based on various criteria
- Dimensionality reduction using UMAP
- Clustering using HDBSCAN
- Visualization of clusters in 2D and 3D using Plotly

## Limitations

- The current implementation assumes that the input JSONL files contain a "Text" key for the text data.
- The filtering functionality is limited to date-based filtering at present. Additional filters can be implemented as required.
- The clustering and visualization methods are specific to UMAP and HDBSCAN, and may not work optimally with other dimensionality reduction or clustering algorithms.
