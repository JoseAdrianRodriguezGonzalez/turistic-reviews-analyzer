'''
keyword_features.py
-------------------
Calcula features relacionadas con la presencia y frecuencia de
palabras clave (vocabulario TF-IDF) dentro de cada comentario.

Funciones:
    load_vocabulary          -- carga el vocabulario desde un CSV de rankings
    compute_keyword_presence -- cuenta keywords presentes por documento
    compute_keyword_density  -- keywords presentes / total tokens
    compute_keyword_features -- combina ambas en un DataFrame
'''

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _generar_ngrams(palabras: list[str], n: int) -> list[str]:
    '''
    Dado una lista de palabras y n, devuelve lista de n-gramas como strings.
    Ejemplo: ["a", "b", "c"], n=2 -> ["a b", "b c"]
    '''
    return [' '.join(palabras[i:i + n]) for i in range(len(palabras) - n + 1)]


def _contar_keywords_por_doc(corpus: list[str], vocabulario: list[str], n: int = 1) -> np.ndarray:
    '''
    Cuenta cuántos tokens del vocabulario (distintos) aparecen en cada documento.

    Usa intersección de conjuntos en vez de una matriz densa para evitar
    el OOM que ocurre con corpus grandes (54K docs × 45K vocab × 8 bytes ≈ 20 GB).
    Complejidad espacial: O(vocab_size) en vez de O(n × vocab_size).
    '''
    vocab_set = set(vocabulario)
    counts = np.zeros(len(corpus), dtype=np.int32)

    for i, doc in enumerate(corpus):
        if pd.isna(doc) or not isinstance(doc, str) or not doc.strip():
            continue
        palabras = doc.split()
        if len(palabras) < n:
            continue
        tokens_doc = set(_generar_ngrams(palabras, n))
        counts[i] = len(tokens_doc & vocab_set)

    return counts


def load_vocabulary(ranking_csv_path: str | Path) -> list[str]:
    '''
    Carga la lista de ngrams desde un archivo de rankings generado
    por nlp_analysis.py (columna "ngram").

    Parametros:
        ranking_csv_path -- ruta al CSV (rankings_unigrams.csv, etc.)
    '''
    df = pd.read_csv(ranking_csv_path)

    if 'ngram' not in df.columns:
        raise ValueError(
            f'El archivo {ranking_csv_path} no contiene la columna "ngram".'
        )

    vocabulary = df['ngram'].dropna().tolist()
    logger.info('Vocabulario cargado: %d terminos desde %s', len(vocabulary), ranking_csv_path)
    return vocabulary


def compute_keyword_presence(
    corpus: list[str],
    vocabulary: list[str],
    ngram_n: int,
) -> np.ndarray:
    '''
    Cuenta cuantas palabras del vocabulario aparecen (al menos una vez)
    en cada documento.

    El resultado esta alineado con el corpus original: los documentos
    vacios o nulos reciben valor 0.

    Parametros:
        corpus     -- lista de comentarios limpios
        vocabulary -- lista de ngrams del vocabulario
        ngram_n    -- 1 para unigramas, 2 para bigramas, 3 para trigramas
    '''
    if not corpus:
        logger.warning('Corpus vacio recibido en compute_keyword_presence')
        return np.array([], dtype=np.int32)

    keyword_presence = _contar_keywords_por_doc(corpus, vocabulary, n=ngram_n)

    logger.debug(
        'Presencia de keywords calculada: media=%.2f, max=%d',
        keyword_presence.mean(),
        keyword_presence.max(),
    )
    return keyword_presence


def compute_keyword_density(
    keyword_presence: np.ndarray,
    token_counts: np.ndarray,
) -> np.ndarray:
    '''
    Calcula la densidad de keywords como la proporcion de tokens del
    vocabulario respecto al total de tokens del documento.

        densidad = keywords_presentes / total_tokens

    Documentos con cero tokens reciben densidad 0.

    Parametros:
        keyword_presence -- array con conteo de keywords presentes
        token_counts     -- array con total de tokens por documento
    '''
    density = np.where(
        token_counts > 0,
        keyword_presence / token_counts,
        0.0,
    )
    return np.round(density, 6)


def compute_keyword_features(
    corpus: list[str],
    token_counts: np.ndarray,
    vocabulary: list[str],
    ngram_n: int,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    '''
    Combina presencia y densidad de keywords en un DataFrame listo
    para concatenar con el resto de features.

    Columnas de salida:
        keyword_count   -- numero de keywords del vocabulario presentes
        keyword_density -- keyword_count / token_count
    '''
    keyword_presence = compute_keyword_presence(corpus, vocabulary, ngram_n)
    keyword_density  = compute_keyword_density(keyword_presence, token_counts)

    result = pd.DataFrame(
        {
            'keyword_count'  : keyword_presence,
            'keyword_density': keyword_density,
        },
        index=index,
    )

    logger.info(
        'Features de keywords calculadas: densidad media=%.4f',
        result['keyword_density'].mean(),
    )
    return result