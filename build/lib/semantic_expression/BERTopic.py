import logging

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

from config import Params, resolve_device

logger = logging.getLogger(__name__)


class BERTopic_analysis:
    def __init__(self, unsupervised, reduction, embedding, docs, *args, **kwargs):
        if unsupervised is None:
            unsupervised = self.set_model_hdbscan()
        if reduction is None:
            reduction = self.set_model_umap()
        if embedding is None:
            embedding = Params.EMBEDDING_MODEL
        self.unsupervised = unsupervised
        self.reduction = reduction
        self.embedding = embedding
        self.docs = docs

    def set_model_umap(
        self,
        n_neighbors: int | None = None,
        n_components: int | None = None,
        min_dist: float | None = None,
        metric: str | None = None,
        **kwargs,
    ) -> UMAP:
        return UMAP(
            n_neighbors=n_neighbors  if n_neighbors  is not None else Params.BERTOPIC_UMAP_N_NEIGHBORS,
            n_components=n_components if n_components is not None else Params.BERTOPIC_UMAP_N_COMPONENTS,
            min_dist=min_dist        if min_dist      is not None else Params.BERTOPIC_UMAP_MIN_DIST,
            metric=metric            if metric        is not None else Params.BERTOPIC_UMAP_METRIC,
            random_state=Params.RANDOM_STATE,
        )

    def set_model_hdbscan(
        self,
        min_cluster_size: int | None = None,
        metric: str | None = None,
        cluster_selection_method: str | None = None,
        prediction_data: bool = True,
        **kwargs,
    ) -> HDBSCAN:
        return HDBSCAN(
            min_cluster_size=min_cluster_size if min_cluster_size is not None else Params.BERTOPIC_HDBSCAN_MIN_CLUSTER_SIZE,
            metric=metric                     if metric            is not None else Params.BERTOPIC_HDBSCAN_METRIC,
            cluster_selection_method=cluster_selection_method if cluster_selection_method is not None else Params.BERTOPIC_HDBSCAN_CLUSTER_SELECTION,
            prediction_data=prediction_data,
        )

    def embedding_extraction(
        self,
        docs: list[str] | None,
        embedding: str | None,
        device: str | None = None,
    ) -> np.ndarray:
        if device is None:
            device = resolve_device(Params.DEVICE)
        if embedding is None:
            embedding = self.embedding
        if self.docs is None and docs is None:
            raise ValueError("No hay documentos")
        if docs is None:
            docs = self.docs
        self.docs = docs
        logger.info("SentenceTransformer device: %s", device)
        model = SentenceTransformer(embedding, device=device)
        emb = model.encode(docs, batch_size=Params.BERTOPIC_BATCH_SIZE, show_progress_bar=True)
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
            calculate_probabilities=Params.BERTOPIC_CALCULATE_PROBS,
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
