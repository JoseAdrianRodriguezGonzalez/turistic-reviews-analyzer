"""
step_01_preprocessing.py
------------------------
Step 1: Preprocesamiento del corpus crudo.

Input  : data/raw/complete.csv
Output : data/data_spanish/clean.csv
         data/data_english/clean.csv
         data/data_mixed/clean.csv
         data/data_spanish/analysis.json
         data/data_english/analysis.json
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
        spanish, english, mixed = process_pipeline(str(Paths.RAW_COMPLETE_CSV))

        logger.info(
            "[%s] Documentos procesados — es: %d | en: %d | mix: %d",
            self.name, len(spanish), len(english), len(mixed),
        )

        save_results(spanish, "data_spanish")
        save_results(english, "data_english")
        save_results(mixed,   "data_mixed")

        logger.info("[%s] Resultados guardados por idioma", self.name)
