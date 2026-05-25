'''
transformer_sentiment.py
------------------------
Bloque 8 — Clasificación de sentimiento mediante modelos transformer.

Expone run_transformer_sentiment() que es llamado desde sentiment_analysis.py.

Métodos:
    rapido   — etiqueta cada comentario directamente con el transformer (default).
    robusto  — usa las etiquetas del transformer como supervisión para entrenar
               un LogisticRegression sobre las features ya extraídas, lo que
               permite capturar comentarios ambiguos que el transformer duda.

Modelos configurables vía Params.SENTIMENT_IDIOMA:
    es    — pysentimiento/robertuito-sentiment-analysis  (español, default)
    multi — cardiffnlp/twitter-xlm-roberta-base-sentiment  (multilingüe)

Features combinables para el método robusto (cualquier subconjunto):
    embeddings  — vectores densos BERTopic (docs_with_topics.npy)
    tfidf       — matriz sparse del vectorizador TF-IDF guardado
    yake        — matriz sparse del vectorizador YAKE guardado
'''

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

# Texto limpio para transformar con los vectorizadores
PATH_NORMALIZED_CSV = DATA_DIR / 'translations' / 'normalized_spanish.csv'
COLUMNA_TEXTO_CLEAN = 'comentario_clean'

MODELOS_SENTIMIENTO = {
    'es'   : 'pysentimiento/robertuito-sentiment-analysis',
    'multi': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
}

# Labels heterogéneas de distintos modelos → etiqueta unificada
MAPA_ETIQUETAS = {
    'pos': 'positivo', 'positive': 'positivo', 'positivo': 'positivo', 'label_2': 'positivo',
    'neg': 'negativo', 'negative': 'negativo', 'negativo': 'negativo', 'label_0': 'negativo',
    'neu': 'neutro',   'neutral':  'neutro',   'neutro':   'neutro',   'label_1': 'neutro',
}


def _normalizar_etiqueta(label: str) -> str:
    return MAPA_ETIQUETAS.get(label.lower(), 'neutro')


def _cargar_pipeline_transformer(idioma: str):
    nombre = MODELOS_SENTIMIENTO.get(idioma, MODELOS_SENTIMIENTO['es'])
    logger.info('Cargando modelo de sentimiento: %s', nombre)
    return hf_pipeline('text-classification', model=nombre, truncation=True, max_length=512)


def _clasificar_por_lotes(textos: list[str], pipe, batch_size: int) -> list[dict]:
    '''
    Inferencia en batches para no saturar memoria en corpus grandes.
    Textos vacíos/nulos reciben etiqueta neutro con confianza 0.
    '''
    resultados = []
    total = len(textos)

    for i in range(0, total, batch_size):
        lote_raw = textos[i : i + batch_size]
        # El tokenizador falla con strings vacíos; se reemplaza con punto
        lote = [t if isinstance(t, str) and t.strip() else '.' for t in lote_raw]
        try:
            resultados.extend(pipe(lote))
        except Exception as error:
            logger.warning('Error en lote %d-%d: %s', i, i + batch_size, error)
            resultados.extend([{'label': 'neutral', 'score': 0.0}] * len(lote))

        procesados = min(i + batch_size, total)
        if procesados % (batch_size * 10) == 0 or procesados == total:
            logger.info('Clasificados %d / %d documentos', procesados, total)

    return resultados


def _cargar_textos_limpios() -> pd.Series:
    '''
    Carga el texto limpio del corpus normalizado, necesario para
    transformar con los vectorizadores TF-IDF y YAKE.
    '''
    df = pd.read_csv(PATH_NORMALIZED_CSV, usecols=['indice', COLUMNA_TEXTO_CLEAN])
    df = df.set_index('indice')
    return df[COLUMNA_TEXTO_CLEAN]


def _cargar_features(fuentes: list[str]) -> np.ndarray | scipy.sparse.spmatrix:
    '''
    Carga y concatena features según las fuentes indicadas.

    Embeddings son densos; tfidf/yake producen matrices sparse.
    Si se mezclan tipos, todo se convierte a sparse para poder
    concatenar con scipy.sparse.hstack sin perder memoria.
    '''
    partes: list[tuple[str, object]] = []

    if 'embeddings' in fuentes:
        ruta = DATA_DIR / 'features' / 'docs_with_topics.npy'
        emb = np.load(ruta)
        logger.info('Embeddings cargados: %s', emb.shape)
        partes.append(('dense', emb))

    if 'tfidf' in fuentes:
        vec = joblib.load(DATA_DIR / 'models' / 'tfidf.pkl')
        textos = _cargar_textos_limpios().fillna('')
        mat = vec.transform(textos)
        logger.info('TF-IDF transformado: %s', mat.shape)
        partes.append(('sparse', mat))

    if 'yake' in fuentes:
        vec = joblib.load(DATA_DIR / 'models' / 'yake_vectorizer.pkl')
        textos = _cargar_textos_limpios().fillna('')
        mat = vec.transform(textos)
        logger.info('YAKE transformado: %s', mat.shape)
        partes.append(('sparse', mat))

    if not partes:
        raise ValueError('No se especificó ninguna fuente de features válida.')

    if len(partes) == 1:
        return partes[0][1]

    tipos = {t for t, _ in partes}
    if tipos == {'dense'}:
        return np.hstack([m for _, m in partes])

    # Mezcla dense + sparse: convertir todo a sparse
    convertidas = [
        scipy.sparse.csr_matrix(mat) if tipo == 'dense' else mat
        for tipo, mat in partes
    ]
    return scipy.sparse.hstack(convertidas)


def _entrenar_clasificador_robusto(
    X,
    etiquetas: pd.Series,
) -> LogisticRegression:
    '''
    LogisticRegression entrenada con las etiquetas del transformer.
    Permite al modelo aprender fronteras de decisión más suaves
    sobre el espacio de features (destilación de conocimiento).
    '''
    mascara = etiquetas.isin(['positivo', 'negativo', 'neutro'])
    X_train = X[mascara.values]
    y_train = etiquetas[mascara].values

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train)
    logger.info('Clasificador robusto listo. Clases: %s', list(clf.classes_))
    return clf


def run_transformer_sentiment(
    textos_originales: pd.Series,
    textos_cleaned: pd.Series,
    idioma: str = 'es',
    metodo: str = 'rapido',
    usar_texto_original: bool = True,
    features: list[str] | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    '''
    Clasifica el sentimiento de cada documento con el transformer
    y opcionalmente entrena un clasificador supervisado sobre features.

    Parámetros:
        textos_originales  — texto sin procesar (mejor para el transformer)
        textos_cleaned     — texto limpio/lematizado (para vectorizadores)
        idioma             — 'es' o 'multi'
        metodo             — 'rapido' o 'robusto'
        usar_texto_original — True para pasar texto original al transformer
        features           — fuentes para el método robusto
        batch_size         — tamaño de lote para inferencia

    Retorna DataFrame con el mismo índice que textos_originales:
        etiqueta_transformer  — etiqueta directa del modelo
        confianza_transformer — score de confianza [0, 1]
        etiqueta_robusto      — predicción del clasificador (solo robusto)
        etiqueta_final        — clasificación definitiva usada downstream
    '''
    if features is None:
        features = ['embeddings']

    textos_entrada = textos_originales if usar_texto_original else textos_cleaned

    pipe = _cargar_pipeline_transformer(idioma)
    logger.info('Clasificando %d documentos con transformer...', len(textos_entrada))
    salidas = _clasificar_por_lotes(textos_entrada.tolist(), pipe, batch_size)

    etiquetas_norm = [_normalizar_etiqueta(s['label']) for s in salidas]
    confianzas     = [round(s['score'], 4) for s in salidas]

    df_resultado = pd.DataFrame(
        {
            'etiqueta_transformer' : etiquetas_norm,
            'confianza_transformer': confianzas,
        },
        index=textos_originales.index,
    )
    df_resultado['etiqueta_final'] = df_resultado['etiqueta_transformer']

    if metodo == 'robusto':
        logger.info('Entrenando clasificador robusto sobre features: %s', features)
        try:
            X   = _cargar_features(features)
            clf = _entrenar_clasificador_robusto(X, pd.Series(etiquetas_norm))
            preds = clf.predict(X)
            df_resultado['etiqueta_robusto'] = preds
            df_resultado['etiqueta_final']   = preds
        except Exception as error:
            logger.warning(
                'Falló el método robusto (%s). Se usan etiquetas del transformer.', error
            )

    logger.info(
        'Distribución final: %s',
        pd.Series(df_resultado['etiqueta_final']).value_counts().to_dict(),
    )
    return df_resultado
