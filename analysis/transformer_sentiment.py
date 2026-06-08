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

import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from transformers import pipeline as hf_pipeline

from config import Params, Paths, resolve_device

logger = logging.getLogger(__name__)


def _hf_device(device_str: str) -> int:
    '''Convierte "cuda"/"cpu" al índice que espera HuggingFace pipeline (-1=CPU, 0=CUDA).'''
    return 0 if device_str == 'cuda' else -1

MODELOS_SENTIMIENTO = {
    'es'   : 'pysentimiento/robertuito-sentiment-analysis',
    'multi': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
}

# Labels heterogéneas de distintos modelos -> etiqueta unificada
MAPA_ETIQUETAS = {
    'pos': 'positivo', 'positive': 'positivo', 'positivo': 'positivo', 'label_2': 'positivo',
    'neg': 'negativo', 'negative': 'negativo', 'negativo': 'negativo', 'label_0': 'negativo',
    'neu': 'neutro',   'neutral':  'neutro',   'neutro':   'neutro',   'label_1': 'neutro',
}


def _normalizar_etiqueta(label: str) -> str:
    return MAPA_ETIQUETAS.get(label.lower(), 'neutro')


def _cargar_pipeline_transformer(idioma: str, device: int | None = None):
    if device is None:
        device = _hf_device(resolve_device(Params.DEVICE))
    nombre =MODELOS_SENTIMIENTO["multi"] if idioma=="fr" or idioma=="en" else MODELOS_SENTIMIENTO["es"] 
    logger.info('Cargando modelo de sentimiento: %s (device=%s)', nombre, 'cuda:0' if device == 0 else 'cpu')
    return hf_pipeline('text-classification', model=nombre, truncation=True, max_length=Params.SENTIMENT_MAX_LENGTH, device=device)


def _sanitizar(texto: str) -> str:
    '''
    Elimina caracteres que producen token IDs fuera de rango en robertuito.

    La traducción Helsinki-NLP puede introducir bytes nulos, caracteres de
    control o unicode no estándar que el tokenizador BPE no puede manejar.
    '''
    import unicodedata
    if not isinstance(texto, str):
        return '.'
    limpio = ''.join(
        c for c in texto
        if unicodedata.category(c)[0] != 'C' or c in ' \t\n'
    )
    limpio = limpio.replace('\x00', '').replace('\n', ' ').replace('\t', ' ')
    return limpio.strip() or '.'


def _clasificar_por_lotes(textos: list[str], pipe, batch_size: int, idioma: str = 'es') -> list[dict]:
    '''
    Inferencia en batches para no saturar memoria en corpus grandes.

    Sanitiza los textos antes de tokenizar para evitar token IDs fuera de
    rango. Si un lote falla, reintenta documento a documento para no perder
    los clasificables del mismo lote.
    Si detecta un error CUDA, recarga el pipeline en CPU y continúa.
    '''
    import torch
    resultados = []
    total = len(textos)
    en_cpu_fallback = False
    errores_lote = 0

    for i in range(0, total, batch_size):
        lote_raw = textos[i : i + batch_size]
        lote = [_sanitizar(t) for t in lote_raw]
        try:
            resultados.extend(pipe(lote))
        except Exception as error:
            error_str = str(error)

            # CUDA device-side assert corrompe el contexto GPU — recargar en CPU
            if 'CUDA' in error_str and not en_cpu_fallback:
                logger.warning(
                    'CUDA context corrompido — recargando en CPU (%d docs restantes)',
                    total - i,
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                pipe = _cargar_pipeline_transformer(idioma, device=-1)
                en_cpu_fallback = True

            # Reintento doc-a-doc para rescatar los clasificables del lote
            errores_lote += 1
            recuperados = 0
            for texto_ind in lote:
                try:
                    resultados.extend(pipe([texto_ind]))
                    recuperados += 1
                except Exception:
                    resultados.append({'label': 'neutral', 'score': 0.0})
            if recuperados < len(lote):
                logger.warning(
                    'Lote %d-%d: %d/%d docs recuperados en reintento individual',
                    i, i + batch_size, recuperados, len(lote),
                )

        procesados = min(i + batch_size, total)
        if procesados % (batch_size * 10) == 0 or procesados == total:
            logger.info('Clasificados %d / %d documentos', procesados, total)

    if errores_lote:
        logger.warning('Total lotes con error (recuperados individualmente): %d', errores_lote)
    return resultados


def _cargar_textos_limpios() -> pd.Series:
    '''
    Carga el texto limpio del corpus normalizado, necesario para
    transformar con los vectorizadores TF-IDF y YAKE.
    '''
    df = pd.read_csv(Paths.NORMALIZED_SPANISH_CSV, usecols=['indice', Params.COLUMNA_TEXTO])
    df = df.set_index('indice')
    return df[Params.COLUMNA_TEXTO]


def _cargar_features(fuentes: list[str]) -> np.ndarray | scipy.sparse.spmatrix:
    '''
    Carga y concatena features según las fuentes indicadas.

    Embeddings son densos; tfidf/yake producen matrices sparse.
    Si se mezclan tipos, todo se convierte a sparse para poder
    concatenar con scipy.sparse.hstack sin perder memoria.
    '''
    partes: list[tuple[str, object]] = []

    if 'embeddings' in fuentes:
        emb = np.load(Paths.DOCS_WITH_TOPICS_NPY)
        logger.info('Embeddings cargados: %s', emb.shape)
        partes.append(('dense', emb))

    if 'tfidf' in fuentes:
        vec = joblib.load(Paths.TFIDF_PKL)
        textos = _cargar_textos_limpios().fillna('')
        mat = vec.transform(textos)
        logger.info('TF-IDF transformado: %s', mat.shape)
        partes.append(('sparse', mat))

    if 'yake' in fuentes:
        vec = joblib.load(Paths.YAKE_PKL)
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

    clf = LogisticRegression(max_iter=Params.SENTIMENT_LR_MAX_ITER, C=Params.SENTIMENT_LR_C, solver=Params.SENTIMENT_LR_SOLVER)
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
    batch_size: int | None = None,
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
    if batch_size is None:
        batch_size = Params.SENTIMENT_BATCH_SIZE

    textos_entrada = textos_originales if usar_texto_original else textos_cleaned

    pipe = _cargar_pipeline_transformer(idioma)
    logger.info('Clasificando %d documentos con transformer...', len(textos_entrada))
    salidas = _clasificar_por_lotes(textos_entrada.tolist(), pipe, batch_size, idioma=idioma)

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
