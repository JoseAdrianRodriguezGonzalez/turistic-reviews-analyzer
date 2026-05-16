"""
processing_pipe.py
------------------
Pipeline de preprocesamiento del corpus crudo.

Lee el CSV maestro, detecta el idioma de cada comentario,
aplica análisis lingüístico y separa los resultados en tres
grupos: español, inglés y mixto.

Uso:
    from preprocessing.processing_pipe import process_pipeline

    spanish, english, mixed = process_pipeline("data/raw/complete.csv")
"""

import logging

import pandas as pd

from preprocessing.individual_functions import (
    detect_language_type,
    get_nlp_model,
    heavy_processing,
    linguistic_analysis,
    normalize_columns,
    normalize_text,
    remove_light_noise,
    remove_noise,
)

logger = logging.getLogger(__name__)


def process_pipeline(
    input_path: str,
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """
    Ejecuta el pipeline completo de preprocesamiento sobre el corpus crudo.

    Para cada comentario:
        1. Normaliza y limpia el texto
        2. Detecta el idioma
        3. Aplica análisis lingüístico (POS, NER, noun phrases)
        4. Genera texto limpio con lematización
        5. Clasifica en español, inglés o mixto

    Parámetros:
        input_path : ruta al CSV maestro con columnas 'text', 'stars', 'location'

    Retorna:
        Tupla de tres listas (spanish_results, english_results, mixed_results).
        Cada elemento de la lista es una tupla:
            (dict_analisis, dict_limpio)

        dict_analisis contiene: indice, estrellas, comentario, pos_tags,
            noun_phrases, entities, entity_density
        dict_limpio contiene: indice, comentario_clean, lang, location
    """
    try:
        df = pd.read_csv(input_path, encoding="utf-8")
    except Exception as error:
        raise RuntimeError(
            f"No se pudo leer el archivo de entrada '{input_path}': {error}"
        ) from error

    df = normalize_columns(df)

    total = len(df)
    logger.info("Iniciando preprocesamiento — %d comentarios en '%s'", total, input_path)

    spanish_results: list[tuple] = []
    english_results: list[tuple] = []
    mixed_results:   list[tuple] = []

    for idx, row in df.iterrows():

        try:
            comentario = row["text"]
            estrellas  = row.get("stars", None)

            # --- Normalización y limpieza ligera ---
            text = normalize_text(comentario)
            text = remove_light_noise(text)

            # --- Detección de idioma ---
            lang = detect_language_type(text)
            nlp  = get_nlp_model(lang)

            # Si el idioma no tiene modelo, usar español como fallback
            if nlp is None:
                logger.debug(
                    "Comentario %d: idioma '%s' sin modelo, usando español como fallback",
                    idx, lang,
                )
                nlp  = get_nlp_model("es")
                lang = "mix"

            # --- Análisis lingüístico ---
            pos_tags, noun_phrases, entities, entity_density = linguistic_analysis(
                text, nlp
            )

            # --- Limpieza pesada y lematización ---
            text       = remove_noise(text)
            clean_text = heavy_processing(text, nlp)

            # --- Construir resultados ---
            dict_analisis = {
                "indice"        : idx,
                "estrellas"     : estrellas,
                "comentario"    : text,
                "pos_tags"      : pos_tags,
                "noun_phrases"  : noun_phrases,
                "entities"      : entities,
                "entity_density": entity_density,
            }

            dict_limpio = {
                "indice"          : idx,
                "comentario_clean": clean_text,
                "lang"            : lang,
                "location"        : row["location"],
            }

            # --- Clasificar por idioma ---
            if lang == "es":
                spanish_results.append((dict_analisis, dict_limpio))
            elif lang == "en":
                english_results.append((dict_analisis, dict_limpio))
            else:
                mixed_results.append((dict_analisis, dict_limpio))

        except Exception as error:
            logger.warning(
                "Error procesando comentario %d — %s: %s",
                idx, type(error).__name__, error,
            )
            continue

        if (idx + 1) % 100 == 0:
            logger.info("Progreso: %d / %d comentarios procesados", idx + 1, total)

    logger.info(
        "Preprocesamiento completado — es: %d | en: %d | mix: %d",
        len(spanish_results),
        len(english_results),
        len(mixed_results),
    )

    return spanish_results, english_results, mixed_results