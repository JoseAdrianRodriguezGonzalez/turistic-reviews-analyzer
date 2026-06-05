"""
step_08_analysis.py
-------------------
Step 8: Análisis semántico y estadístico.

Ejecuta en orden:
    1. Análisis de sentimiento (basado en estrellas + POS)
    2. Análisis de entidades NER
    3. Grafo de co-ocurrencia
    4. Detección de tendencias por destino y tópico

Input  : data/results/docs_with_topics.csv
         data/unified/analysis_unified.csv
         data/features/features_nlp.csv
         data/features/ner_groups.json
Output : data/analysis/sentiment/sentimiento_por_destino.csv
         data/analysis/sentiment/sentimiento_por_topico.csv
         data/analysis/entities/entidades_por_destino.csv
         data/analysis/cooccurrence/coocurrencia_entidades.csv
         data/analysis/trends/perfil_destino.csv
         data/analysis/resumen_analysis.csv
"""

import logging
from pathlib import Path

from config import Params, Paths
from pipeline.base_step import BaseStep

logger = logging.getLogger(__name__)


class StepAnalysis(BaseStep):

    name = "analysis"

    @property
    def ejecutar_sentiment(self)        -> bool: return Params.ANALYSIS_RUN_SENTIMENT
    @property
    def ejecutar_entities(self)         -> bool: return Params.ANALYSIS_RUN_ENTITIES
    @property
    def ejecutar_cooccurrence(self)     -> bool: return Params.ANALYSIS_RUN_COOCCURRENCE
    @property
    def ejecutar_trends(self)           -> bool: return Params.ANALYSIS_RUN_TRENDS
    @property
    def ejecutar_outliers(self)         -> bool: return Params.ANALYSIS_RUN_OUTLIERS
    @property
    def ejecutar_sentiment_topics(self) -> bool: return Params.ANALYSIS_RUN_SENTIMENT_TOPICS

    @property
    def input_paths(self) -> list[Path]:
        return [
            Paths.DOCS_WITH_TOPICS_CSV,
            Paths.UNIFIED_ANALYSIS_CSV,
            Paths.FEATURES_NLP_CSV,
            Paths.NER_GROUPS_JSON,
        ]

    @property
    def output_paths(self) -> list[Path]:
        paths = [
            Paths.SENTIMIENTO_POR_DESTINO_CSV,
            Paths.SENTIMIENTO_POR_TOPICO_CSV,
            Paths.ENTIDADES_POR_DESTINO_CSV,
            Paths.COOCURRENCIA_ENTIDADES_CSV,
            Paths.PERFIL_DESTINO_CSV,
            Paths.ANALYSIS_RESUMEN_CSV,
        ]
        if self.ejecutar_outliers:
            paths += [
                Paths.OUTLIERS_CSV,
                Paths.NORMALES_CSV,
                Paths.OUTLIERS_RESUMEN_CSV,
            ]
        if self.ejecutar_sentiment_topics:
            paths += [
                Paths.SENTIMENT_TOPICS_POS_JSON,
                Paths.SENTIMENT_TOPICS_NEG_JSON,
                Paths.PRECIO_VALOR_COSTO_JSON,
                Paths.SENTIMENT_TOPICS_RESUMEN_CSV,
            ]
        return paths

    def _run(self) -> None:
        from analysis.analysis_pipeline import run_analysis_pipeline

        logger.info(
            "[%s] Ejecutando análisis — "
            "sentiment=%s | entities=%s | cooccurrence=%s | trends=%s | outliers=%s | sentiment_topics=%s",
            self.name,
            self.ejecutar_sentiment,
            self.ejecutar_entities,
            self.ejecutar_cooccurrence,
            self.ejecutar_trends,
            self.ejecutar_outliers,
            self.ejecutar_sentiment_topics,
        )

        run_analysis_pipeline(
            ejecutar_sentiment        = self.ejecutar_sentiment,
            ejecutar_entities         = self.ejecutar_entities,
            ejecutar_cooccurrence     = self.ejecutar_cooccurrence,
            ejecutar_trends           = self.ejecutar_trends,
            ejecutar_outliers         = self.ejecutar_outliers,
            ejecutar_sentiment_topics = self.ejecutar_sentiment_topics,
        )

        logger.info(
            "[%s] Análisis completado — output en %s",
            self.name, Paths.ANALYSIS_DIR,
        )
