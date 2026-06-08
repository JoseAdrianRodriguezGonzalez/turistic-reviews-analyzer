"""
step_06_clustering.py
---------------------
Step 6: Clustering multi-fuente.

Ejecuta KMeans, Jerárquico y HDBSCAN sobre cuatro representaciones
vectoriales distintas (embeddings, features, tfidf, yake).

Input  : data/features/docs_with_topics.npy
         data/features/features_nlp.csv
         data/models/tfidf.pkl
         data/models/yake_vectorizer.pkl
         data/translations/normalized_spanish.csv
Output : data/clustering/comparacion_fuentes.csv
         data/clustering/embeddings/ranking_completo.csv
         data/clustering/embeddings/mejores_modelos.csv
         data/clustering/embeddings/etiquetas_mejores.json
         data/clustering/embeddings/proyeccion_2d.npy
"""

import logging
from pathlib import Path

from config import Params, Paths
from pipeline.base_step import BaseStep

logger = logging.getLogger(__name__)


class StepClustering(BaseStep):

    name = "clustering"

    @property
    def ejecutar_hdbscan(self)    -> bool: return Params.CLUSTERING_RUN_HDBSCAN
    @property
    def ejecutar_jerarquico(self) -> bool: return Params.CLUSTERING_RUN_JERARQUICO
    @property
    def ejecutar_embeddings(self) -> bool: return Params.CLUSTERING_RUN_EMBEDDINGS
    @property
    def ejecutar_features(self)   -> bool: return Params.CLUSTERING_RUN_FEATURES
    @property
    def ejecutar_tfidf(self)      -> bool: return Params.CLUSTERING_RUN_TFIDF
    @property
    def ejecutar_yake(self)       -> bool: return Params.CLUSTERING_RUN_YAKE

    @property
    def input_paths(self) -> list[Path]:
        lang = Params.LANGUAGE
        if lang == "all":
            text_path = Paths.NORMALIZED_SPANISH_CSV
        elif lang == "es":
            text_path = Paths.SPANISH_CLEAN_CSV
        elif lang == "en":
            text_path = Paths.ENGLISH_CLEAN_CSV
        elif lang == "fr":
            text_path = Paths.FRENCH_CLEAN_CSV
        else:
            raise ValueError(f"Idioma no soportado: {lang}")

        return [
            Paths.DOCS_WITH_TOPICS_NPY,
            Paths.FEATURES_NLP_CSV,
            Paths.TFIDF_PKL,
            Paths.YAKE_PKL,
            text_path,
        ]

    @property
    def output_paths(self) -> list[Path]:
        return [
            Paths.CLUSTERING_COMPARACION_CSV,
            Paths.CLUSTERING_EMBEDDINGS_DIR / Paths.CLUSTERING_RANKING_FILE,
            Paths.CLUSTERING_EMBEDDINGS_DIR / Paths.CLUSTERING_MEJORES_FILE,
            Paths.CLUSTERING_EMBEDDINGS_DIR / Paths.CLUSTERING_ETIQUETAS_FILE,
            Paths.CLUSTERING_EMBEDDINGS_DIR / Paths.CLUSTERING_PROYECCION_FILE,
            Paths.ENRICHMENT_DIR / "active_model.json"
        ]

    def _run(self) -> None:
        from clustering.clustering_pipeline import run_clustering_pipeline

        logger.info(
            "[%s] Ejecutando clustering — "
            "hdbscan=%s | embeddings=%s | features=%s | tfidf=%s | yake=%s",
            self.name,
            self.ejecutar_hdbscan,
            self.ejecutar_embeddings,
            self.ejecutar_features,
            self.ejecutar_tfidf,
            self.ejecutar_yake,
        )

        resultados = run_clustering_pipeline(
            ejecutar_hdbscan    = self.ejecutar_hdbscan,
            ejecutar_jerarquico = self.ejecutar_jerarquico,
            ejecutar_embeddings = self.ejecutar_embeddings,
            ejecutar_features   = self.ejecutar_features,
            ejecutar_tfidf      = self.ejecutar_tfidf,
            ejecutar_yake       = self.ejecutar_yake,
        )

        fuentes_procesadas = list(resultados.keys())
        logger.info(
            "[%s] Clustering completado — fuentes procesadas: %s",
            self.name, fuentes_procesadas,
        )
