import torch

from src.clustering import ClusteringPipeline
from src.config import clustering_conf, embedder_conf
from src.embed import Embedder
from src.utils import filter_by_date

embedder = Embedder(embedder_conf)
filename = "mastodon_toots.jsonl"
batch_size = 32

texts, embeddings = embedder.embed_corpus(filename, filter_by_date, batch_size)

nan_mask = torch.isnan(embeddings)
nan_positions = torch.nonzero(nan_mask, as_tuple=True)
if nan_mask.sum() > 0:
    print(f"NaN at embedding {nan_positions[0][0]}")
else:
    print("No NaN values found in the embeddings.")

non_nan_mask = ~torch.isnan(embeddings).any(dim=1)
filtered_embeddings = embeddings[non_nan_mask]
filtered_texts = [text for text, mask in zip(texts, non_nan_mask) if mask]


print(f"Embedded {len(texts)} texts")
print(f"Embedding shape: {embeddings.shape}")

clustering_pipeline = ClusteringPipeline(clustering_conf)
clusters, labels = clustering_pipeline.process(filtered_texts, filtered_embeddings.cpu().tolist(), view="3d")
