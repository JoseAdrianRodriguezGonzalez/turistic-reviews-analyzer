"""
individual_functions.py
-----------------------
Funciones de preprocesamiento individual de texto.

Cubre:
    - Normalización de columnas y texto
    - Detección de idioma
    - Análisis lingüístico (POS, NER, noun phrases)
    - Procesamiento pesado (lematización, stopwords)
    - Lectura y guardado de resultados
    - Construcción del CSV maestro desde carpetas raw/
"""

import json
import logging
import os
import re
import subprocess

import pandas as pd
import spacy
from langdetect import DetectorFactory, LangDetectException, detect_langs
from unidecode import unidecode

from config import Params, Paths, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

DetectorFactory.seed = Params.RANDOM_STATE

# CARGA DE MODELOS spaCy
# Se carga bajo demanda, no al importar el módulo

_nlp_es: spacy.language.Language | None = None
_nlp_en: spacy.language.Language | None = None
_nlp_fr: spacy.language.Language | None = None


def _load_spacy_model(model_name: str) -> spacy.language.Language:
    """
    Carga un modelo spaCy. Si no está instalado, lo descarga
    automáticamente y reintenta.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        logger.warning("Modelo %s no encontrado, descargando...", model_name)
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=False)
        try:
            return spacy.load(model_name)
        except OSError as error:
            raise RuntimeError(
                f"No se pudo cargar el modelo spaCy '{model_name}': {error}"
            ) from error


def get_nlp_model(lang: str) -> spacy.language.Language | None:
    """
    Retorna el modelo spaCy correspondiente al idioma.
    Carga el modelo la primera vez que se solicita (lazy loading).

    Parámetros:
        lang : 'es' para español, 'en' para inglés, 'fr' para francés

    Retorna:
        Objeto nlp de spaCy o None si el idioma no está soportado
    """
    global _nlp_es, _nlp_en, _nlp_fr

    if lang == "es":
        if _nlp_es is None:
            logger.info("Cargando modelo spaCy: %s", Params.SPACY_MODEL_ES)
            _nlp_es = _load_spacy_model(Params.SPACY_MODEL_ES)
        return _nlp_es

    if lang == "en":
        if _nlp_en is None:
            logger.info("Cargando modelo spaCy: %s", Params.SPACY_MODEL_EN)
            _nlp_en = _load_spacy_model(Params.SPACY_MODEL_EN)
        return _nlp_en

    if lang == "fr":
        if _nlp_fr is None:
            logger.info("Cargando modelo spaCy: %s", Params.SPACY_MODEL_FR)
            _nlp_fr = _load_spacy_model(Params.SPACY_MODEL_FR)
        return _nlp_fr

    return None


# NORMALIZACIÓN DE COLUMNAS

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los nombres de columnas de un DataFrame:
    strip, minúsculas y espacios reemplazados por guión bajo.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


# DETECCIÓN DE IDIOMA

# Palabras funcionales para validación secundaria de idioma
_ES_STOPWORDS = frozenset({"el", "la", "los", "las", "y", "es", "en", "a", "de", "fue", "por", "con", "pero", "son", "muy", "porque", "para", "como", "estaba", "tiene"})
_EN_STOPWORDS = frozenset({"the", "and", "is", "in", "it", "to", "of", "was", "for", "on", "this", "that", "with", "but", "are", "very"})
_FR_STOPWORDS = frozenset({"le", "la", "les", "et", "est", "dans", "il", "pour", "sur", "ce", "cette", "avec", "mais", "sont", "très", "être", "avoir"})


def _secondary_validation(text: str, primary_prediction: str) -> str:
    '''Validación secundaria por stopwords cuando langdetect da resultado ambiguo.'''
    words = re.findall(r"\b\w+\b", text.lower())
    scores = {
        "es": sum(word in _ES_STOPWORDS for word in words),
        "en": sum(word in _EN_STOPWORDS for word in words),
        "fr": sum(word in _FR_STOPWORDS for word in words),
    }
    best_score = max(scores.values())
    if best_score == 0:
        return primary_prediction
    best_langs = [lang for lang, score in scores.items() if score == best_score]
    return best_langs[0] if len(best_langs) == 1 else "mixed"


def detect_language_type(text: str) -> str:
    """
    Detecta el idioma de un texto.

    Retorna:
        'es'      si es español con alta confianza
        'en'      si es inglés con alta confianza
        'fr'      si es francés con alta confianza
        'mixed'   si hay mezcla de idiomas soportados
        'unknown' si la detección falla
    """
    try:
        langs = detect_langs(text)

        if not langs:
            return "unknown"

        probs = {lang.lang: lang.prob for lang in langs}

        tokens = text.split()
        is_short = len(tokens) < Params.LANGDETECT_SHORT_THRESHOLD
        threshold = Params.LANGDETECT_CONF_HIGH if is_short else Params.LANGDETECT_CONF_NORMAL

        valid_langs = {
            lang: prob for lang, prob in probs.items()
            if lang in SUPPORTED_LANGUAGES and prob > threshold
        }

        if len(valid_langs) == 1:
            primary_lang = next(iter(valid_langs))
        else:
            primary_lang = "mixed"

        if primary_lang in SUPPORTED_LANGUAGES:
            if is_short or probs.get(primary_lang, 0.0) < Params.LANGDETECT_CONF_SECONDARY:
                return _secondary_validation(text, primary_lang)

        return primary_lang

    except LangDetectException:
        return "unknown"


# NORMALIZACIÓN DE TEXTO


def normalize_text(text: str) -> str:
    """
    Normalización ligera: colapsa espacios múltiples y hace strip.
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_ner(text: str) -> str:
    """
    Normalización para entidades NER:
    minúsculas, sin acentos, sin espacios extra.
    """
    text = text.lower().strip()
    text = unidecode(text)
    text = re.sub(r"\s+", " ", text)
    return text


def remove_noise(text: str) -> str:
    """
    Elimina etiquetas HTML y caracteres especiales no deseados.
    Conserva letras, números, espacios, puntos y comas.
    """
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s.,]", "", text)
    return text


def remove_light_noise(text: str) -> str:
    """
    Eliminación ligera de ruido: etiquetas y artefactos menores.
    """
    text = re.sub(r"<*:?>", "", text)
    return text


# ANÁLISIS LINGÜÍSTICO

def tokenize(text: str, nlp: spacy.language.Language) -> list[str]:
    """
    Tokeniza un texto usando el modelo spaCy indicado.
    """
    document = nlp(text)
    return [token.text for token in document]


def linguistic_analysis(
    text: str,
    nlp: spacy.language.Language,
) -> tuple[list, list, list, float]:
    """
    Realiza análisis lingüístico completo sobre un texto.

    Retorna:
        pos_tags       : lista de etiquetas POS por token
        noun_phrases   : lista de frases nominales
        entities       : lista de dicts {text, label} de entidades NER
        entity_density : entidades / tokens del documento
    """
    document = nlp(text)

    pos_tags     = [token.pos_ for token in document]
    noun_phrases = [chunk.text for chunk in document.noun_chunks]
    entities     = [
        {"text": ent.text, "label": ent.label_}
        for ent in document.ents
    ]
    entity_density = (
        len(document.ents) / len(document) if len(document) > 0 else 0.0
    )

    return pos_tags, noun_phrases, entities, entity_density


# PROCESAMIENTO PESADO

def heavy_processing(text: str, nlp: spacy.language.Language) -> str:
    """
    Aplica lematización y eliminación de stopwords y puntuación.
    Retorna el texto limpio como string con tokens separados por espacio.
    """
    document = nlp(text.lower())

    tokens = []
    for token in document:
        if not token.is_stop and not token.is_punct:
            tokens.append(unidecode(token.text))

    return " ".join(tokens)


# CONSTRUCCIÓN DE CSV MAESTRO DESDE raw/

def normalize_df(
    df: pd.DataFrame,
    source: str,
    location: str,
) -> pd.DataFrame:
    """
    Normaliza un DataFrame crudo de cualquier fuente (Google, TripAdvisor,
    Instagram) a un esquema común con columnas: text, stars, source, location.
    """
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.lower()

    df["text"] = None
    for col in Params.CSV_TEXT_COLUMN_ALIASES:
        if col in df.columns:
            df["text"] = df[col]
            break

    df["stars"] = None
    for col in Params.CSV_RATING_COLUMN_ALIASES:
        if col in df.columns:
            df["stars"] = df[col]
            break

    df["stars"]    = pd.to_numeric(df["stars"], errors="coerce")
    df["source"]   = source
    df["location"] = location

    df = df[["text", "stars", "source", "location"]]
    df = df[df["text"].notna()]

    return df


def read_csv_safe(path: str) -> pd.DataFrame:
    '''Lee un CSV probando los encodings definidos en Params.CSV_ENCODINGS.'''
    for encoding in Params.CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer el archivo {path} con ningún encoding conocido")


def csv_to_dictionary(src: str) -> pd.DataFrame:
    """
    Lee todas las carpetas y archivos CSV dentro de src/
    y los unifica en un solo DataFrame con columnas estándar.

    Estructura esperada de src/:
        src/
            google/
                huatulco.csv
                la_paz.csv
            trip/
                huatulco.csv
            insta/
                huatulco.csv
    """
    all_dfs = []

    try:
        folders = os.listdir(src)
    except OSError as error:
        raise RuntimeError(f"No se pudo leer la carpeta {src}: {error}") from error

    for folder in folders:
        path_folder = os.path.join(src, folder)

        if not os.path.isdir(path_folder):
            continue

        for file in os.listdir(path_folder):
            if not file.endswith(".csv"):
                continue

            path_file = os.path.join(path_folder, file)
            location  = os.path.splitext(file)[0]
            source    = folder

            try:
                df = read_csv_safe(path_file)
                df = normalize_df(df, source, location)
                all_dfs.append(df)
                logger.debug("Leído: %s/%s (%d filas)", folder, file, len(df))
            except Exception as error:
                logger.warning("No se pudo procesar %s: %s", path_file, error)

    if not all_dfs:
        raise RuntimeError(f"No se encontraron archivos CSV válidos en {src}")

    master_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        "CSV maestro construido: %d documentos desde %s",
        len(master_df), src,
    )
    return master_df


def create_csv_master(src: str, out: str) -> pd.DataFrame:
    """
    Crea el CSV maestro unificado si no existe.
    Si ya existe, lo carga directamente.

    Parámetros:
        src : carpeta raíz con subcarpetas por fuente (raw/)
        out : ruta de salida del CSV maestro
    """
    if os.path.exists(out):
        logger.info("CSV maestro ya existe, cargando desde %s", out)
        return pd.read_csv(out)

    logger.info("Creando CSV maestro desde %s", src)
    df = csv_to_dictionary(src)
    df.to_csv(out, index=False)
    logger.info("CSV maestro guardado en %s", out)
    return df


# GUARDADO DE RESULTADOS POR IDIOMA

def create_data_folders() -> None:
    """
    Crea las carpetas de datos por idioma si no existen.
    Usa las rutas definidas en config.Paths.
    """
    dirs = [
        Paths.DATA_SPANISH_DIR,
        Paths.DATA_ENGLISH_DIR,
        Paths.DATA_FRENCH_DIR,
        Paths.DATA_MIXED_DIR,
    ]
    for directory in dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Carpeta lista: %s", directory)
        except OSError as error:
            logger.warning("No se pudo crear %s: %s", directory, error)


def save_results(results: list[tuple], folder) -> None:
    """
    Guarda los resultados de preprocesamiento de un idioma en su carpeta.

    Genera tres archivos:
        analysis.csv  — datos lingüísticos (POS, NER, noun phrases)
        clean.csv     — texto limpio + idioma + location
        analysis.json — mismo contenido que analysis.csv en formato JSON

    Parámetros:
        results : lista de tuplas (dict_analisis, dict_limpio)
        folder  : Path a la carpeta de destino (e.g. Paths.DATA_SPANISH_DIR)
    """
    if not results:
        logger.warning("save_results: lista vacía para '%s', nada que guardar", folder)
        return

    stage_1_2 = [r[0] for r in results]
    stage_3   = [r[1] for r in results]

    base_path = folder

    try:
        base_path.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(stage_1_2).to_csv(
            base_path / "analysis.csv", index=False
        )
        pd.DataFrame(stage_3).to_csv(
            base_path / "clean.csv", index=False
        )
        with open(base_path / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(stage_1_2, f, indent=4, ensure_ascii=False)

        logger.info(
            "Resultados guardados en %s — %d documentos",
            base_path, len(results),
        )

    except OSError as error:
        logger.error("Error al guardar resultados en %s: %s", base_path, error)