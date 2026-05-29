"""
step_09_visualization.py
------------------------
Step 9: Generación de visualizaciones.

Ejecuta todos los scripts de visualización en orden.
Genera tanto archivos estáticos (.png) como interactivos (.html).

Input  : data/clustering/embeddings/proyeccion_2d.npy
         data/topic_enrichment/embeddings/kmeans_k8/keywords_por_cluster.csv
         data/analysis/sentiment/sentimiento_por_destino.csv
         data/analysis/sentiment/sentimiento_por_topico.csv
         data/analysis/trends/tendencias_topicos_destino.csv
         data/analysis/entities/entidades_por_destino.csv
         data/analysis/cooccurrence/coocurrencia_entidades.csv
         data/analysis/trends/perfil_destino.csv
Output : visualization/topic_graph.png
         visualization/topic_graph_2.html
         visualization/keywords_entities.png
         visualization/keywords_entities_2.html
         visualization/polarities.png
         visualization/polarities_2.html
         visualization/metadata_overview.html
"""

import logging
from pathlib import Path

from config import Paths
from pipeline.base_step import BaseStep

logger = logging.getLogger(__name__)

# Mapeo nombre -> función de cada visualización
# Separado de _run() para poder activar/desactivar individualmente
_VISUALIZACIONES = [
    ("topic_graph",           "visualization.topic_graph",            "run_topic_graph"),
    ("topic_graph_interactivo","visualization.topic_graph_2",         "run_topic_graph_interactivo"),
    ("keywords_entities",     "visualization.keywords_entities",      "run_keywords_entities"),
    ("keywords_entities_cluster","visualization.keywords_entities_cluster","generar_keywords_entities"),
    ("polarities",            "visualization.polarities",             "run_polarities"),
    ("polarities_heatmap",    "visualization.polarities_heatmap",     "run_polarities_heatmap"),
    ("overview",              "visualization.overview",               "generar_metadata_overview"),
]


class StepVisualization(BaseStep):

    name = "visualization"

    # Lista de nombres de visualizaciones a ejecutar
    # None = todas las disponibles
    visualizaciones: list[str] | None = None

    @property
    def input_paths(self) -> list[Path]:
        return [
            Paths.CLUSTERING_EMBEDDINGS_DIR / Paths.CLUSTERING_PROYECCION_FILE,
            Paths.ENRICHMENT_KEYWORDS_CSV,
            Paths.SENTIMIENTO_POR_DESTINO_CSV,
            Paths.SENTIMIENTO_POR_TOPICO_CSV,
            Paths.TENDENCIAS_TOPICOS_DESTINO_CSV,
            Paths.ENTIDADES_POR_DESTINO_CSV,
            Paths.COOCURRENCIA_ENTIDADES_CSV,
            Paths.PERFIL_DESTINO_CSV,
        ]

    @property
    def output_paths(self) -> list[Path]:
        return [
            Paths.VISUALIZATION_DIR / "topic_graph.png",
            Paths.VISUALIZATION_DIR / "topic_graph_2.html",
            Paths.VISUALIZATION_DIR / "keywords_entities.png",
            Paths.VISUALIZATION_DIR / "keywords_entities_2.html",
            Paths.VISUALIZATION_DIR / "polarities.png",
            Paths.VISUALIZATION_DIR / "polarities_2.html",
            Paths.VISUALIZATION_DIR / "metadata_overview.html",
        ]

    def _run(self) -> None:
        Paths.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

        for viz_name, modulo, funcion in _VISUALIZACIONES:

            # Filtrar si se especificaron visualizaciones concretas
            if self.visualizaciones is not None:
                if viz_name not in self.visualizaciones:
                    logger.info(
                        "[%s] Visualización '%s' omitida por configuración",
                        self.name, viz_name,
                    )
                    continue

            self._ejecutar_visualizacion(viz_name, modulo, funcion)

    def _ejecutar_visualizacion(
        self,
        viz_name: str,
        modulo: str,
        funcion: str,
    ) -> None:
        """
        Importa y ejecuta una visualización de forma aislada.
        Si falla, loguea el error y continúa con la siguiente.
        """
        logger.info("[%s] Generando: %s", self.name, viz_name)

        try:
            import importlib
            mod = importlib.import_module(modulo)
            fn  = getattr(mod, funcion)
            fn()
            logger.info("[%s] '%s' generado correctamente", self.name, viz_name)

        except Exception as error:
            logger.error(
                "[%s] Error en '%s' — %s: %s",
                self.name, viz_name, type(error).__name__, error,
            )
