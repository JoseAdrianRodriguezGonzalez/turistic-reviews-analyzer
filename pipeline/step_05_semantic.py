"""
step_05_semantic.py
-------------------
Step 5: Extracción semántica con BERTopic.

Genera embeddings, modela tópicos y opcionalmente genera microtópicos
por región y tópico padre.

Input  : data/translations/normalized_spanish.csv
         data/data_spanish/analysis.json
         data/data_english/analysis.json
         data/data_mixed/analysis.json
Output : data/features/docs_with_topics.npy
         data/features/ner_groups.json
         data/models/tfidf.pkl
         data/models/yake_vectorizer.pkl
         data/results/docs_with_topics.csv
         data/results/topics.csv
"""

import logging
from pathlib import Path

from config import Params, Paths
from pipeline.base_step import BaseStep

logger = logging.getLogger(__name__)


class StepSemantic(BaseStep):

    name = "semantic"

    @property
    def run_microtopics(self) -> bool:
        return Params.SEMANTIC_RUN_MICROTOPICS

    @property
    def input_paths(self) -> list[Path]:
        lang=Params.LANGUAGE
        if lang=="all":
            return [
                Paths.NORMALIZED_SPANISH_CSV,
                Paths.SPANISH_ANALYSIS_JSON,
                Paths.ENGLISH_ANALYSIS_JSON,
                Paths.MIXED_ANALYSIS_JSON,
            ]
        if lang=="es":
            return [
                Paths.SPANISH_CLEAN_CSV,
                Paths.SPANISH_ANALYSIS_JSON,
            ]
        if lang == "en":
            return [
                Paths.ENGLISH_CLEAN_CSV,
                Paths.ENGLISH_ANALYSIS_JSON,
            ]

        if lang == "fr":
            return [
                Paths.FRENCH_CLEAN_CSV,
                Paths.FRENCH_ANALYSIS_JSON,
            ]
        return []

    @property
    def output_paths(self) -> list[Path]:
        return [
            Paths.DOCS_WITH_TOPICS_NPY,
            Paths.NER_GROUPS_JSON,
            Paths.TFIDF_PKL,
            Paths.YAKE_PKL,
            Paths.ENRICHMENT_TEXTS
            #Paths.DOCS_WITH_TOPICS_CSV,
            #Paths.TOPICS_CSV,

        ]

    def _run(self) -> None:
        from semantic_expression.pipeline import pipe, pipe_microtopics

        logger.info("[%s] Ejecutando pipeline semántico (BERTopic + NER)", self.name)
        pipe()

        if self.run_microtopics:
            logger.info("[%s] Generando microtópicos por región", self.name)
            pipe_microtopics()
        else:
            logger.info(
                "[%s] Microtópicos omitidos (run_microtopics=False)",
                self.name,
            )
