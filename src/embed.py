import json
import os
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Embedder:
    """A class for embedding text corpora using SentenceTransformer models."""

    def __init__(self, config: Dict[str, Any]):

        self.config = config
        self.embedding_model = config["embedding_model"]
        self.device = config["device"]
        self.model = self.load_embedder()
        if config["half_precision"]:
            self.model.half()

    def load_embedder(self) -> SentenceTransformer:
        """Load the SentenceTransformer model."""
        return SentenceTransformer(
            self.embedding_model,
            trust_remote_code=True,
            device=self.device
        )

    def load_corpus(self, filename: str, filter: Optional[Callable] = None) -> List[Dict[str, str]]:
        """
        Load the corpus from a JSONL file.

        Args:
            filename (str): Name of the JSONL file containing the corpus.

        Returns:
            List[Dict[str, str]]: The loaded corpus.
        """
        corpus = []
        with open(os.path.join(self.config["base_path"], filename)) as f:
            json_lines = f.readlines()

        for line in json_lines:
            corpus.append(json.loads(line))

        if filter is not None:
            corpus = filter(corpus)

        return corpus

    def preprocess_corpus(self, corpus: List[Dict[str, str]]) -> List[str]:
        """
        Preprocess the corpus.

        Args:
            corpus (List[Dict[str, str]]): The corpus to preprocess.

        Returns:
            List[str]: The preprocessed corpus.
        """
        clean_text = []
        for elem in corpus:
            clean_text.append(elem[self.config["text_key"]])
        return clean_text

    def batch_corpus(self, corpus: List[str], batch_size: int) -> Generator[List[str], None, None]:
        """
        Yield batches of the corpus.

        Args:
            corpus (List[str]): The corpus to batch.
            batch_size (int): The size of each batch.

        Yields:
            List[str]: A batch of the corpus.
        """
        for i in range(0, len(corpus), batch_size):
            yield corpus[i:i + batch_size]

    def embed_corpus(self, filename: str, filter: Optional[Callable] = None, batch_size: int = 32) -> Tuple[List[str], torch.Tensor]:
        """
        Embed the corpus in batches.

        Args:
            filename (str): Name of the JSON file containing the corpus.
            batch_size (int): The size of each batch.

        Yields:
            Tuple[List[str], torch.Tensor]: A batch of the corpus and its corresponding embeddings.
        """
        corpus = self.load_corpus(filename, filter)
        clean_corpus = self.preprocess_corpus(corpus)

        def get_batch() -> Generator[Tuple[List[str], torch.Tensor], None, None]:
            for batch in self.batch_corpus(clean_corpus, batch_size):
                batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
                yield batch, batch_embeddings

        all_embeddings = []
        all_texts = []

        with tqdm(total=len(corpus), desc="Embedding corpus") as pbar:
            for batch_texts, batch_embeddings in get_batch():
                all_texts.extend(batch_texts)
                all_embeddings.append(batch_embeddings)
                pbar.update(len(batch_texts))

        all_embeddings = torch.cat(all_embeddings, dim=0)

        return all_texts, all_embeddings
