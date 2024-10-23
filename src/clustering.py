from logging import Logger
from typing import Literal, Optional

import numpy as np
import plotly.express as px
import umap
from sklearn.cluster import HDBSCAN


class ClusteringPipeline:
    def __init__(self, config):
        self.config = config
        self.logger = Logger("ClusteringPipeline")

    def reduce_dimensions(self, text_embeddings, n_components=3):
        dim_reducer = umap.UMAP(n_components=n_components, **self.config["umap"])
        return dim_reducer.fit_transform(text_embeddings)

    def cluster_data(self, umap_embeddings):
        clusterer = HDBSCAN(**self.config["hdbscan"])
        return clusterer.fit_predict(umap_embeddings)

    def process(
        self,
        texts: list[str],
        text_embeddings: list[list[float]] | np.ndarray,
        view: Optional[Literal["2d", "3d"]] = None,
    ):
        umap_embeddings = self.reduce_dimensions(text_embeddings)

        cluster_labels = self.cluster_data(umap_embeddings)
        final_clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in final_clusters:
                final_clusters[label] = []
            final_clusters[label].append(texts[i])

        if view == "2d":
            umap_embeddings_2d = self.reduce_dimensions(
                text_embeddings, n_components=2
            )
            self.visualize_2d(texts, umap_embeddings_2d, cluster_labels)

        elif view == "3d":
            umap_embeddings_3d = self.reduce_dimensions(
                text_embeddings, n_components=3
            )
            self.visualize_3d(texts, umap_embeddings_3d, cluster_labels)

        self.logger.info(
            f"Found {len(final_clusters.keys())} clusters from {len(texts)} texts."
        )
        return final_clusters, cluster_labels

    def visualize_2d(self, texts, umap_embeddings, cluster_labels):
        hover_name = [art[:128] for art in texts]

        fig = px.scatter(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            color=cluster_labels,
            labels={"color": "Cluster"},
            hover_name=hover_name,
        )
        fig.show()

    def visualize_3d(self, texts, umap_embeddings, cluster_labels):
        hover_name = [art[:128] for art in texts]

        fig = px.scatter_3d(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            z=umap_embeddings[:, 2],
            color=cluster_labels,
            labels={"color": "Cluster"},
            hover_name=hover_name,
        )
        fig.show()
