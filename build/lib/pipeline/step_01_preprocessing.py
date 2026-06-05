"""
step_01_preprocessing.py
------------------------
Step 1: Preprocesamiento del corpus crudo.

Input  : data/raw/complete.csv
Output : data/data_spanish/clean.csv
         data/data_english/clean.csv
         data/data_french/clean.csv
         data/data_mixed/clean.csv
         data/data_spanish/analysis.json
         data/data_english/analysis.json
         data/data_french/analysis.json
         data/data_mixed/analysis.json
"""

import logging
from pathlib import Path

from config import Paths
from pipeline.base_step import BaseStep

logger = logging.getLogger(__name__)


class StepPreprocessing(BaseStep):

    name = "preprocessing"

    @property
    def input_paths(self) -> list[Path]:
        return [Paths.RAW_COMPLETE_CSV]

    @property
    def output_paths(self) -> list[Path]:
        return [
            Paths.SPANISH_CLEAN_CSV,
            Paths.SPANISH_ANALYSIS_JSON,
            Paths.ENGLISH_CLEAN_CSV,
            Paths.ENGLISH_ANALYSIS_JSON,
            Paths.FRENCH_CLEAN_CSV,
            Paths.FRENCH_ANALYSIS_JSON,
            Paths.MIXED_CLEAN_CSV,
            Paths.MIXED_ANALYSIS_JSON,
        ]

    def _run(self) -> None:
        from preprocessing.individual_functions import (
            create_data_folders,
            save_results,
        )
        from preprocessing.processing_pipe import process_pipeline

        logger.info("[%s] Creando carpetas de datos por idioma", self.name)
        create_data_folders()

        logger.info("[%s] Ejecutando pipeline de preprocesamiento", self.name)
        spanish, english, french, mixed = process_pipeline(str(Paths.RAW_COMPLETE_CSV))

        logger.info(
            "[%s] Documentos procesados — es: %d | en: %d | fr: %d | mix: %d",
            self.name, len(spanish), len(english), len(french), len(mixed),
        )

        save_results(spanish, Paths.DATA_SPANISH_DIR)
        save_results(english, Paths.DATA_ENGLISH_DIR)
        save_results(french,  Paths.DATA_FRENCH_DIR)
        save_results(mixed,   Paths.DATA_MIXED_DIR)

        logger.info("[%s] Resultados guardados por idioma", self.name)
