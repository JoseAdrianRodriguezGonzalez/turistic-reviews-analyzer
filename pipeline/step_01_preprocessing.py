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

from config import Params,Paths
from pipeline.base_step import BaseStep

logger = logging.getLogger(__name__)


class StepPreprocessing(BaseStep):

    name = "preprocessing"

    @property
    def input_paths(self) -> list[Path]:
        return [Paths.RAW_COMPLETE_CSV]

    @property
    def output_paths(self) -> list[Path]:
        lang = Params.LANGUAGE
        outputs = []
        if lang in ("es", "all"):
            outputs += [Paths.SPANISH_CLEAN_CSV, Paths.SPANISH_ANALYSIS_JSON]
        if lang in ("en", "all"):
            outputs += [Paths.ENGLISH_CLEAN_CSV, Paths.ENGLISH_ANALYSIS_JSON]
        if lang in ("fr", "all"):
            outputs += [Paths.FRENCH_CLEAN_CSV, Paths.FRENCH_ANALYSIS_JSON]
        if lang in ("all"):
            outputs += [Paths.MIXED_CLEAN_CSV, Paths.MIXED_ANALYSIS_JSON]
        return outputs

    def _run(self) -> None:
        from preprocessing.individual_functions import (
            create_data_folders,
            save_results,
        )
        from preprocessing.processing_pipe import process_pipeline

        logger.info("[%s] Creando carpetas de datos por idioma", self.name)
        create_data_folders()
        columna=Params.TEXT_COLUMN
        logger.info("[%s] Ejecutando pipeline de preprocesamiento", self.name)
        results = process_pipeline(str(Paths.RAW_COMPLETE_CSV),columna=columna)
        spanish,english,french,mixed=results
        logger.info(
            "[%s] Documentos procesados — es: %d | en: %d | fr: %d | mix: %d",
            self.name, len(spanish), len(english), len(french), len(mixed),
        )
        lang=Params.LANGUAGE
        
        if lang in ("es", "all"):
            save_results(spanish, Paths.DATA_SPANISH_DIR)

        if lang in ("en", "all"):
            save_results(english, Paths.DATA_ENGLISH_DIR)

        if lang in ("fr", "all"):
            save_results(french, Paths.DATA_FRENCH_DIR)

        if lang in ( "all"):
            save_results(mixed, Paths.DATA_MIXED_DIR)
        logger.info("[%s] Resultados guardados por idioma", self.name)
