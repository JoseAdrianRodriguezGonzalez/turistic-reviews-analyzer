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

from config import Params,Paths
from pipeline.base_step import BaseStep

logger = logging.getLogger(__name__)


class StepTranslation(BaseStep):

    name = "translation"

    @property
    def input_paths(self) -> list[Path]:
        lang = Params.LANGUAGE

        if lang == "all":
            return [
                Paths.SPANISH_CLEAN_CSV,
                Paths.ENGLISH_CLEAN_CSV,
                Paths.MIXED_CLEAN_CSV,
            ]

        if lang == "es":
            return [Paths.SPANISH_CLEAN_CSV]
        if lang == "en":
            return [Paths.ENGLISH_CLEAN_CSV]
        if lang == "fr":
            return [Paths.FRENCH_CLEAN_CSV]
        if lang == "mixed":
            return [Paths.MIXED_CLEAN_CSV]

        return []

    @property
    def output_paths(self) -> list[Path]:
        lang = Params.LANGUAGE
        if lang == "all":
            return [
                Paths.JOINED_CSV,
                Paths.NORMALIZED_SPANISH_CSV,
            ]
        #  clave: output = input
        return [Paths.NORMALIZED_SPANISH_CSV]

    def _run(self) -> None:
        from translation.pipeline import normalize_language
        Paths.TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
        lang=Params.LANGUAGE
        if lang=="all":

            logger.info("[%s] Uniendo archivos por idioma", self.name)
            dfs = []

            for path in [
                Paths.SPANISH_CLEAN_CSV,
                Paths.ENGLISH_CLEAN_CSV,
                Paths.MIXED_CLEAN_CSV,
            ]:
                if path.exists():
                    dfs.append(pd.read_csv(path))

            if not dfs:
                raise RuntimeError("No hay archivos para unir")

            df_joined = pd.concat(dfs, ignore_index=True)
            df_joined.to_csv(Paths.JOINED_CSV, index=False)

            logger.info("[%s] Traduciendo a español", self.name)

            df_normalized = normalize_language(df_joined)
            df_normalized.to_csv(Paths.NORMALIZED_SPANISH_CSV, index=False)

            return
        logger.info("[%s] Modo single-language: %s", self.name, lang)
        if lang == "es":
            input_path = Paths.SPANISH_CLEAN_CSV
        elif lang == "en":
            input_path = Paths.ENGLISH_CLEAN_CSV
        elif lang == "fr":
            input_path = Paths.FRENCH_CLEAN_CSV
        else:
            raise ValueError(f"Idioma no soportado: {lang}")
        if not input_path.exists():
            raise RuntimeError(f"No existe archivo de entrada: {input_path}")

        df = pd.read_csv(input_path)
        df.to_csv(Paths.NORMALIZED_SPANISH_CSV, index=False)

        logger.info(
            "[%s] Dataset pasado sin traducción (%d filas)",
            self.name, len(df)
        )