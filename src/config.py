import torch

# Embedder
embedder_conf = {
    "embedding_model": "dunzhang/stella_en_1.5B_v5",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "half_precision": True,
    "base_path": "./data",
    "text_key": "Text"
}

# ClusteringPipeline
clustering_conf = {
    "umap":
        {
            # https://umap-learn.readthedocs.io/en/latest/parameters.html
            "n_neighbors": 3, # Balance between local and global structure in the data. A low value of 3 means UMAP will focus more on very local structure, potentially at the expense of the overall picture
            "min_dist": 0.05,  # How tightly UMAP is allowed to pack points together. A small value of 0.05 will result in denser, more clumped embeddings
            "metric": "cosine", # How distance is computed in the input space
            # n_components: 7, # Output dimension of the UMAP algorithm. Defaults to 3 or 2 in the visualization methods.
        },
    "hdbscan":
        {
            #
            "algorithm": "auto",
            "leaf_size": 20,
            "cluster_selection_method": "leaf", # Tends to produce many small, homogeneous clusters. Suitable when you're interested in fine-grained, highly similar groupings.
            "metric": "euclidean",
            "min_cluster_size": 3, # Sets the minimum number of samples required to form a cluster
            "min_samples": 1, # Controls how conservative the clustering is. A value of 1 makes the clustering less conservative, potentially resulting in more noise points being assigned to clusters.
            "cluster_selection_epsilon": 0.6, # Helps merge nearby clusters. Ensures that clusters below the given threshold are not split further than the threshold value, which prevents the creation of too many micro-clusters in dense regions.
    }
}
