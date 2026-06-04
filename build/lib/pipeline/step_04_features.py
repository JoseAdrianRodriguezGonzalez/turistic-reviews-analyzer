"""
step_04_features.py
-------------------
Step 4: Feature engineering NLP.

Calcula features de longitud, keywords, POS y entidades NER
para cada documento del corpus.

Input  : data/translations/normalized_spanish.csv
         data/processed/rankings_unigrams.csv
         data/data_spanish/analysis.json  (entidades NER de Adrian)
Output : data/features/features_nlp.csv
"""

import logging
from pathlib import Path

from config import Paths
from pipeline.base_step import BaseStep

logger = logging.getLogger(__name__)


class StepFeatures(BaseStep):

    name = "features"

    @property
    def input_paths(self) -> list[Path]:
        return [
            Paths.NORMALIZED_SPANISH_CSV,
            Paths.RANKINGS_UNIGRAMS_CSV,
        ]

    @property
    def output_paths(self) -> list[Path]:
        return [Paths.FEATURES_NLP_CSV]

    def _run(self) -> None:
        from feature_engineering.features import run_feature_pipeline

        logger.info("[%s] Ejecutando pipeline de feature engineering", self.name)

        features_df = run_feature_pipeline(
            data_clean_path     = Paths.NORMALIZED_SPANISH_CSV,
            data_analysis_path  = Paths.UNIFIED_ANALYSIS_CSV,
            vocab_unigrams_path = Paths.RANKINGS_UNIGRAMS_CSV,
            analysis_json_path  = Paths.UNIFIED_ANALYSIS_JSON,
            output_path         = Paths.FEATURES_NLP_CSV,
        )

        logger.info(
            "[%s] features_nlp.csv generado: %d documentos x %d columnas",
            self.name, len(features_df), len(features_df.columns),
        )
