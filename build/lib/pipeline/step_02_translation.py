"""
step_02_translation.py
----------------------
Step 2: Unión de idiomas y traducción a español.

Input  : data/data_spanish/clean.csv
         data/data_english/clean.csv
         data/data_mixed/clean.csv
Output : data/translations/joined.csv
         data/translations/normalized_spanish.csv
"""

import logging
from pathlib import Path

import pandas as pd

from config import Paths
from pipeline.base_step import BaseStep

logger = logging.getLogger(__name__)


class StepTranslation(BaseStep):

    name = "translation"

    @property
    def input_paths(self) -> list[Path]:
        return [
            Paths.SPANISH_CLEAN_CSV,
            Paths.ENGLISH_CLEAN_CSV,
            Paths.MIXED_CLEAN_CSV,
        ]

    @property
    def output_paths(self) -> list[Path]:
        return [
            Paths.JOINED_CSV,
            Paths.NORMALIZED_SPANISH_CSV,
        ]

    def _run(self) -> None:
        from translation.pipeline import normalize_language

        logger.info("[%s] Uniendo archivos por idioma", self.name)

        try:
            df_spanish = pd.read_csv(Paths.SPANISH_CLEAN_CSV)
            df_english = pd.read_csv(Paths.ENGLISH_CLEAN_CSV)
            df_mixed   = pd.read_csv(Paths.MIXED_CLEAN_CSV)
        except Exception as error:
            raise RuntimeError(
                f"Error al leer archivos de idioma: {error}"
            ) from error

        df_joined = pd.concat(
            [df_spanish, df_english, df_mixed],
            ignore_index=True,
        )

        Paths.TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
        df_joined.to_csv(Paths.JOINED_CSV, index=False)

        logger.info(
            "[%s] joined.csv generado: %d documentos totales",
            self.name, len(df_joined),
        )

        logger.info(
            "[%s] Traduciendo documentos no-españoles a español", self.name
        )

        df_normalized = normalize_language(df_joined)

        df_normalized.to_csv(Paths.NORMALIZED_SPANISH_CSV, index=False)

        logger.info(
            "[%s] normalized_spanish.csv generado: %d documentos",
            self.name, len(df_normalized),
        )
