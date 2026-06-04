'''
entity_features.py
------------------
Calcula features relacionadas con entidades nombradas (NER) por documento.

Modo A (preferido): lee el JSON de análisis ya generado por el pipeline
de preprocesamiento que contiene las entidades extraídas y la densidad.

Modo B (fallback): recalcula las entidades desde el texto usando spaCy
cuando el JSON no está disponible.

Tipos de entidades considerados: definidos en Params.NER_ENTITY_TYPES
'''

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import Params

logger = logging.getLogger(__name__)


def load_entities_from_json(
    analysis_json_path: str | Path,
    n_documents: int,
) -> pd.DataFrame | None:
    '''
    Modo A: lee el archivo analysis.json y extrae entity_density y conteos.
    Retorna None si el archivo no existe o tiene errores (activa Modo B).
    '''
    path = Path(analysis_json_path)
    if not path.exists():
        logger.warning('JSON de entidades no encontrado en %s, usando Modo B', path)
        return None

    try:
        with open(path, encoding='utf-8') as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        logger.warning('Error al leer JSON de entidades: %s, usando Modo B', error)
        return None

    entity_density = np.zeros(n_documents, dtype=float)
    type_counts    = {etype: np.zeros(n_documents, dtype=int) for etype in Params.NER_ENTITY_TYPES}

    for record in data:
        idx = record.get('indice')
        if idx is None or idx >= n_documents:
            continue

        entity_density[idx] = record.get('entity_density', 0.0)

        for entity in record.get('entities', []):
            label = entity.get('label', '')
            if label in type_counts:
                type_counts[label][idx] += 1

    result = pd.DataFrame({'entity_density': np.round(entity_density, 6)})
    for etype in Params.NER_ENTITY_TYPES:
        result[f'entity_count_{etype.lower()}'] = type_counts[etype]

    result['entity_count_total'] = sum(type_counts[t] for t in Params.NER_ENTITY_TYPES)

    logger.info(
        'Entidades cargadas desde JSON: %d registros, densidad media=%.4f',
        len(data),
        result['entity_density'].mean(),
    )
    return result


def compute_entities_from_text(
    cleaned_series: pd.Series,
    nlp,
) -> pd.DataFrame:
    '''
    Modo B: recalcula entidades desde el texto usando nlp.pipe() para
    procesamiento por lotes.
    '''
    texts = cleaned_series.fillna('').tolist()
    rows = []

    for doc in nlp.pipe(texts, batch_size=Params.SPACY_BATCH_SIZE):
        n_tokens = len(doc)
        n_ents   = len(doc.ents)
        density  = round(n_ents / n_tokens, 6) if n_tokens > 0 else 0.0

        counts = {etype: 0 for etype in Params.NER_ENTITY_TYPES}
        for ent in doc.ents:
            if ent.label_ in counts:
                counts[ent.label_] += 1

        row = {'entity_density': density}
        row.update({f'entity_count_{t.lower()}': counts[t] for t in Params.NER_ENTITY_TYPES})
        row['entity_count_total'] = n_ents
        rows.append(row)

    result = pd.DataFrame(rows, index=cleaned_series.index)

    logger.info(
        'Entidades recalculadas con spaCy: %d documentos, densidad media=%.4f',
        len(result),
        result['entity_density'].mean(),
    )
    return result


def compute_entity_features(
    cleaned_series: pd.Series,
    nlp,
    analysis_json_path: str | Path | None = None,
) -> pd.DataFrame:
    '''
    Orquesta Modo A y Modo B según disponibilidad del JSON.
    Intenta Modo A primero; si falla cae a Modo B (recalcula con spaCy).
    '''
    if analysis_json_path is not None:
        result = load_entities_from_json(analysis_json_path, n_documents=len(cleaned_series))
        if result is not None:
            result.index = cleaned_series.index
            return result

    logger.info('Ejecutando Modo B: recalculo de entidades con spaCy')
    return compute_entities_from_text(cleaned_series, nlp)
