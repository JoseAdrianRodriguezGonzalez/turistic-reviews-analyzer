"""
step_03_vocabulary.py
---------------------
Step 3: Construcción de vocabulario y rankings de n-gramas.

Input  : data/translations/normalized_spanish.csv  (columna comentario_clean)
Output : data/processed/rankings_unigrams.csv
         data/processed/rankings_bigrams.csv
         data/processed/rankings_trigrams.csv
"""

import logging
from pathlib import Path

from config import Paths
from pipeline.base_step import BaseStep

logger = logging.getLogger(__name__)


class StepVocabulary(BaseStep):

    name = "vocabulary"

    @property
    def input_paths(self) -> list[Path]:
        return [Paths.NORMALIZED_SPANISH_CSV]

    @property
    def output_paths(self) -> list[Path]:
        return [
            Paths.RANKINGS_UNIGRAMS_CSV,
            Paths.RANKINGS_BIGRAMS_CSV,
            Paths.RANKINGS_TRIGRAMS_CSV,
        ]

    def _run(self) -> None:
        from feature_engineering.vocabulary import build_vocabulary_from_clean

        logger.info(
            "[%s] Generando vocabulario desde %s",
            self.name, Paths.NORMALIZED_SPANISH_CSV,
        )

        df_uni, df_bi, df_tri = build_vocabulary_from_clean(
            input_path = Paths.NORMALIZED_SPANISH_CSV,
            output_uni = Paths.RANKINGS_UNIGRAMS_CSV,
            output_bi  = Paths.RANKINGS_BIGRAMS_CSV,
            output_tri = Paths.RANKINGS_TRIGRAMS_CSV,
        )

        logger.info(
            "[%s] Vocabulario generado — unigramas: %d | bigramas: %d | trigramas: %d",
            self.name, len(df_uni), len(df_bi), len(df_tri),
        )
