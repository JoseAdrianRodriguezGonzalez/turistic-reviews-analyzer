"""
preprocessing/
--------------
Módulo de preprocesamiento del corpus crudo.

Funciones principales:
    process_pipeline     -- pipeline completo: lee CSV, detecta idioma,
                            aplica NLP y separa por idioma
    create_data_folders  -- crea carpetas data_spanish/, data_english/, data_mixed/
    save_results         -- guarda analysis.csv, clean.csv y analysis.json
    create_csv_master    -- construye el CSV maestro desde carpetas raw/
"""

from preprocessing.individual_functions import (
    create_csv_master,
    create_data_folders,
    csv_to_dictionary,
    detect_language_type,
    get_nlp_model,
    heavy_processing,
    linguistic_analysis,
    normalize_columns,
    normalize_df,
    normalize_ner,
    normalize_text,
    read_csv_safe,
    remove_light_noise,
    remove_noise,
    save_results,
    tokenize,
)
from preprocessing.processing_pipe import process_pipeline

__all__ = [
    "process_pipeline",
    "create_data_folders",
    "save_results",
    "create_csv_master",
    "csv_to_dictionary",
    "normalize_columns",
    "normalize_df",
    "normalize_text",
    "normalize_ner",
    "remove_noise",
    "remove_light_noise",
    "detect_language_type",
    "get_nlp_model",
    "tokenize",
    "linguistic_analysis",
    "heavy_processing",
    "read_csv_safe",
]