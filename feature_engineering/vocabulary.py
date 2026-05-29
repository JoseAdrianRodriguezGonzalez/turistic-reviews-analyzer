'''
vocabulary.py
--------------
Genera vocabulario de n-gramas (unigramas, bigramas, trigramas) desde
normalized_spanish.csv. Se ejecuta ANTES del feature_engineering para
preparar los rankings que necesitan las features de keywords.

El función principal genera 3 CSVs de rankings en data/processed/:
    -- rankings_unigrams.csv   (unigramas, 1-gramas)
    -- rankings_bigrams.csv    (bigramas, 2-gramas)
    -- rankings_trigrams.csv   (trigramas, 3-gramas)

Cada CSV contiene:
    -- ngram            : el n-grama como string
    -- total_frequency  : cantidad de veces que aparece en el corpus
    -- relative_frequency : proporcion respecto al total de n-gramas

Uso desde main.py:
    from feature_engineering.vocabulary import build_vocabulary_from_clean
    build_vocabulary_from_clean()

Uso directo:
    python vocabulary.py
'''

import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from nltk.util import ngrams

from config import Paths

logger = logging.getLogger(__name__)


def _build_ngrams_and_frequency(tokenized_texts: list[str], n: int) -> pd.DataFrame:
    '''
    Genera n-gramas a partir de una lista de textos ya tokenizados.

    Parametros:
        tokenized_texts -- lista de strings donde las palabras están separadas por espacios
        n               -- tamaño del n-grama (1 para unigramas, 2 para bigramas, etc.)

    Retorna:
        DataFrame con columnas: ngram, total_frequency, relative_frequency
        Ordenado por frecuencia descendente
    '''
    # Counter.update() en lugar de acumular lista + Counter(lista):
    # evita materializar todos los n-gramas en memoria (hasta 500 MB con trigramas).
    frequency_count: Counter = Counter()

    for text in tokenized_texts:
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            continue

        tokens = [t for t in text.split() if len(t) >= 2]

        if len(tokens) >= n:
            frequency_count.update(' '.join(gram) for gram in ngrams(tokens, n))

    if not frequency_count:
        logger.warning('No se generaron n-gramas para n=%d', n)
        return pd.DataFrame(columns=['ngram', 'total_frequency', 'relative_frequency'])

    total_ngrams = sum(frequency_count.values())

    results = []
    for ngram, count in frequency_count.most_common():
        relative_frequency = count / total_ngrams if total_ngrams > 0 else 0
        results.append((ngram, count, relative_frequency))

    df = pd.DataFrame(
        results,
        columns=['ngram', 'total_frequency', 'relative_frequency']
    )
    return df.round({'relative_frequency': 6})


def build_vocabulary_from_clean(
    input_path: str | Path = Paths.NORMALIZED_SPANISH_CSV,
    output_uni: str | Path = Paths.RANKINGS_UNIGRAMS_CSV,
    output_bi: str | Path = Paths.RANKINGS_BIGRAMS_CSV,
    output_tri: str | Path = Paths.RANKINGS_TRIGRAMS_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Genera vocabulario de n-gramas desde normalized_spanish.csv y exporta rankings a CSVs.

    Parametros:
        input_path  -- ruta al CSV de entrada (defecto: data/translations/normalized_spanish.csv)
        output_uni  -- ruta salida unigramas (defecto: data/processed/rankings_unigrams.csv)
        output_bi   -- ruta salida bigramas (defecto: data/processed/rankings_bigrams.csv)
        output_tri  -- ruta salida trigramas (defecto: data/processed/rankings_trigrams.csv)

    Retorna:
        Tupla (df_unigrams, df_bigrams, df_trigrams)
    '''
    input_path = Path(input_path)
    output_uni = Path(output_uni)
    output_bi = Path(output_bi)
    output_tri = Path(output_tri)

    output_uni.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error('Archivo no encontrado: %s', input_path)
        raise FileNotFoundError(f'No existe {input_path}')

    logger.info('Leyendo %s...', input_path)
    df = pd.read_csv(input_path)

    columna = 'comentario_clean'
    if columna not in df.columns:
        logger.error('Columna "%s" no encontrada en %s', columna, input_path)
        raise ValueError(f'{input_path} debe contener columna "{columna}"')

    texts = df[columna].tolist()
    logger.info('Corpus cargado: %d comentarios', len(texts))

    logger.info('Generando unigramas...')
    df_unigrams = _build_ngrams_and_frequency(texts, n=1)

    logger.info('Generando bigramas...')
    df_bigrams = _build_ngrams_and_frequency(texts, n=2)

    logger.info('Generando trigramas...')
    df_trigrams = _build_ngrams_and_frequency(texts, n=3)

    df_unigrams.to_csv(output_uni, index=False)
    df_bigrams.to_csv(output_bi, index=False)
    df_trigrams.to_csv(output_tri, index=False)

    logger.info(
        'Vocabulario generado: unigramas=%d | bigramas=%d | trigramas=%d',
        len(df_unigrams), len(df_bigrams), len(df_trigrams),
    )

    return df_unigrams, df_bigrams, df_trigrams


if __name__ == '__main__':
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S',
    )
    build_vocabulary_from_clean()
