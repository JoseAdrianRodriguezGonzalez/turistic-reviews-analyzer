'''
pos_features.py
---------------
Calcula la distribucion de partes del discurso (POS tags) por documento
usando el modelo de spaCy en espanol.

Las distribuciones se expresan como proporciones (valor entre 0 y 1)
del total de tokens con POS asignado en cada comentario, lo que hace
las features comparables entre comentarios de distinta longitud.

POS tags incluidas: definidas en Params.POS_RELEVANT_TAGS

Funciones:
    load_spacy_model    -- carga el modelo spaCy según Params.SPACY_MODEL_ES
    tag_document        -- aplica POS tagging a un texto y retorna conteos
    compute_pos_features -- aplica tagging a todo el corpus con nlp.pipe()
'''

import logging
import subprocess

import numpy as np
import pandas as pd

from config import Params

logger = logging.getLogger(__name__)


def load_spacy_model(model_name: str | None = None):
    '''
    Carga el modelo de spaCy indicado. Si no está instalado, lo descarga
    automáticamente y reintenta la carga.
    '''
    if model_name is None:
        model_name = Params.SPACY_MODEL_ES
    try:
        import spacy
        nlp = spacy.load(model_name)
        logger.info('Modelo spaCy cargado: %s', model_name)
        return nlp

    except OSError:
        logger.warning('Modelo %s no encontrado, descargando...', model_name)
        subprocess.run(['python', '-m', 'spacy', 'download', model_name], check=False)

        import spacy
        nlp = spacy.load(model_name)
        logger.info('Modelo spaCy descargado y cargado: %s', model_name)
        return nlp


def tag_document(text: str, nlp) -> dict[str, int]:
    '''
    Aplica POS tagging a un documento y retorna un diccionario con
    el conteo de cada tag relevante definido en Params.POS_RELEVANT_TAGS.
    '''
    counts = {tag: 0 for tag in Params.POS_RELEVANT_TAGS}

    if not text or not isinstance(text, str) or not text.strip():
        return counts

    doc = nlp(text)
    for token in doc:
        if token.pos_ in counts:
            counts[token.pos_] += 1

    return counts


def compute_pos_features(
    cleaned_series: pd.Series,
    nlp,
) -> pd.DataFrame:
    '''
    Aplica POS tagging a todos los documentos usando nlp.pipe() para
    procesamiento por lotes (más eficiente que llamar nlp() por documento).

    Columnas de salida (una por tag en Params.POS_RELEVANT_TAGS):
        pos_ratio_<tag>  -- proporción del tag respecto al total etiquetado
        pos_total_tagged -- total de tokens etiquetados con tags relevantes
    '''
    texts = cleaned_series.fillna('').tolist()
    rows = []

    for doc in nlp.pipe(texts, batch_size=Params.SPACY_BATCH_SIZE):
        counts = {tag: 0 for tag in Params.POS_RELEVANT_TAGS}
        for token in doc:
            if token.pos_ in counts:
                counts[token.pos_] += 1
        rows.append(counts)

    counts_df = pd.DataFrame(rows, index=cleaned_series.index)
    total_tagged = counts_df.sum(axis=1)

    ratio_df = pd.DataFrame(index=cleaned_series.index)
    for tag in Params.POS_RELEVANT_TAGS:
        col_name = f'pos_ratio_{tag.lower()}'
        ratio_df[col_name] = np.where(
            total_tagged > 0,
            counts_df[tag] / total_tagged,
            0.0,
        ).round(6)

    ratio_df['pos_total_tagged'] = total_tagged.to_numpy(dtype=int)

    logger.info(
        'POS features calculadas: %d documentos, promedio etiquetados por doc=%.2f',
        len(ratio_df),
        ratio_df['pos_total_tagged'].mean(),
    )
    return ratio_df
