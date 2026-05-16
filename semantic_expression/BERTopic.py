import logging

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

logger = logging.getLogger(__name__)


class BERTopic_analysis:
    def __init__(self, unsupervised, reduction, embedding, docs, *args, **kwargs):
        if unsupervised is None:
            unsupervised = self.set_model_hdbscan()
        if reduction is None:
            reduction = self.set_model_umap()
        if embedding is None:
            embedding = "paraphrase-multilingual-MiniLM-L12-v2"
        self.unsupervised = unsupervised
        self.reduction = reduction
        self.embedding = embedding
        self.docs = docs

    def set_model_umap(
        self,
        n_neighbors: int = 30,
        n_components: int = 8,
        min_dist: float = 0.0,
        metric: str = "cosine",
        **kwargs,
    ) -> UMAP:
        return UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
        )

    def set_model_hdbscan(
        self,
        min_cluster_size: int = 15,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        prediction_data: bool = True,
        **kwargs,
    ) -> HDBSCAN:
        return HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            prediction_data=prediction_data,
        )

    def embedding_extraction(
        self,
        docs: list[str] | None,
        embedding: str | None,
        device: str = "cuda",
    ) -> np.ndarray:
        if embedding is None:
            embedding = self.embedding
        if self.docs is None and docs is None:
            raise ValueError("No hay documentos")
        if docs is None:
            docs = self.docs
        self.docs = docs
        model = SentenceTransformer(embedding, device=device)
        emb = model.encode(docs, batch_size=64, show_progress_bar=True)
        self.embedded = emb
        return emb

    def fit(self) -> tuple:
        if not hasattr(self, "embedded"):
            self.embedding_extraction(self.docs, self.embedding)
        embeddings = self.embedded
        topic_model = BERTopic(
            embedding_model=None,
            umap_model=self.reduction,
            hdbscan_model=self.unsupervised,
            language="multilingual",
            calculate_probabilities=True,
        )
        topics, probs = topic_model.fit_transform(self.docs, embeddings)
        self.model = topic_model
        self.topics = topics
        self.probs = probs
        return topics, probs

    def get_topics(self):
        return self.model.get_topic_info()

    def get_topic(self, topic_id: int):
        return self.model.get_topic(topic_id)

    def transform(self, new_docs: list[str]):
        embeddings = self.embedding_extraction(new_docs, self.embedding)
        return self.model.transform(new_docs, embeddings)
