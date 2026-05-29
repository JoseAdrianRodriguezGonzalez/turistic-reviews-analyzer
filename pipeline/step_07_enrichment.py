"""
step_07_enrichment.py
---------------------
Step 7: Enriquecimiento de tópicos.

Para cada cluster extrae keywords (TF-IDF interno), documentos
representativos (similitud coseno al centroide) y jerarquía.

Input  : data/clustering/embeddings/etiquetas_mejores.json
         data/clustering/embeddings/proyeccion_2d.npy
         data/translations/normalized_spanish.csv
Output : data/topic_enrichment/resumen_enrichment.csv
         data/topic_enrichment/embeddings/kmeans_k8/keywords_por_cluster.csv
         data/topic_enrichment/embeddings/kmeans_k8/documentos_representativos.csv
"""

import logging
from pathlib import Path

from config import Paths
from pipeline.base_step import BaseStep

logger = logging.getLogger(__name__)


class StepEnrichment(BaseStep):

    name = "enrichment"

    # Fuentes a enriquecer (deben coincidir con las generadas en step_06)
    fuentes: list[str] | None = None  # None = todas las disponibles

    @property
    def input_paths(self) -> list[Path]:
        return [
            Paths.CLUSTERING_EMBEDDINGS_DIR / Paths.CLUSTERING_ETIQUETAS_FILE,
            Paths.CLUSTERING_EMBEDDINGS_DIR / Paths.CLUSTERING_PROYECCION_FILE,
            Paths.NORMALIZED_SPANISH_CSV,
        ]

    @property
    def output_paths(self) -> list[Path]:
        return [
            Paths.ENRICHMENT_RESUMEN_CSV,
            Paths.ENRICHMENT_KEYWORDS_CSV,
        ]

    def _run(self) -> None:
        from topic_enrichment.enrichment_pipeline import run_enrichment_pipeline

        logger.info(
            "[%s] Ejecutando enriquecimiento de tópicos — fuentes: %s",
            self.name,
            self.fuentes if self.fuentes is not None else "todas",
        )

        run_enrichment_pipeline(fuentes=self.fuentes)

        logger.info(
            "[%s] Enriquecimiento completado — output en %s",
            self.name, Paths.ENRICHMENT_DIR,
        )
