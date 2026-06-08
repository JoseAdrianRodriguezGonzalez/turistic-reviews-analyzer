'''
outlier_detection.py
--------------------
Bloque 8 — Detección de comentarios atípicos en el corpus turístico.

Identifica outliers usando tres técnicas de detección de anomalías sobre
los embeddings semánticos (384-dim) del corpus:

    - Isolation Forest  : ensemble de árboles de aislamiento probabilístico
    - One-Class SVM     : frontera de máximo margen en espacio kernel RBF
    - KNN / LOF         : densidad local relativa (Local Outlier Factor)

El veredicto final combina los tres modelos por mayoría de votos: un
documento se considera atípico si ≥ Params.OUTLIER_UMBRAL_ENSEMBLE modelos
lo etiquetan como outlier (default 2 de 3).

Además se registra la señal de HDBSCAN: documentos con topic == -1 en
docs_with_topics.csv ya son clasificados como ruido por el modelo de
clustering, y su flag se incluye como columna informativa adicional.

Sobre los documentos atípicos se genera análisis de n-gramas
(unigramas, bigramas, trigramas). Los documentos normales se exportan
en normales.csv para el siguiente paso del análisis.

Inputs:
    data/features/docs_with_topics.npy          -- embeddings 384-dim (step 05)
    data/translations/normalized_spanish.csv    -- texto limpio con comentario_clean
    data/results/docs_with_topics.csv           -- topic/location/lang por documento

Outputs en data/analysis/outliers/:
    outliers.csv            -- documentos atípicos + scores de cada modelo
    normales.csv            -- documentos normales para el siguiente paso
    outliers_unigrams.csv
    outliers_bigrams.csv
    outliers_trigrams.csv
    resumen_outliers.csv

Uso desde analysis_pipeline.py:
    from analysis.outlier_detection import run_outlier_detection
    resultados = run_outlier_detection()
'''

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.util import ngrams as nltk_ngrams
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from config import Params, Paths

logger = logging.getLogger(__name__)

_OUTLIERS_DIR = Paths.OUTLIERS_DIR


def _cargar_embeddings() -> np.ndarray:
    '''Carga los embeddings semánticos precomputados (docs_with_topics.npy).'''
    path = Paths.DOCS_WITH_TOPICS_NPY
    if not path.exists():
        raise FileNotFoundError(
            f'Embeddings no encontrados: {path}\n'
            'Asegúrate de haber ejecutado el step 05 (semantic) antes.'
        )
    embeddings = np.load(path)
    logger.info('Embeddings cargados: shape=%s dtype=%s', embeddings.shape, embeddings.dtype)
    return embeddings


def _cargar_corpus() -> pd.DataFrame:
    '''
    Carga corpus con texto limpio y metadata de tópicos.

    Combina normalized_spanish.csv (comentario_clean) con docs_with_topics.csv
    (topic, location, lang) por alineación de índice de fila — ambos CSVs
    comparten el mismo orden de documentos producido por el step 05.
    '''
    path_clean  = Paths.NORMALIZED_SPANISH_CSV
   # path_topics = Paths.DOCS_WITH_TOPICS_CSV

    if not path_clean.exists():
        raise FileNotFoundError(f'No encontrado: {path_clean}')

    df = pd.read_csv(path_clean).reset_index(drop=True)
    """
    if path_topics.exists():
        df_topics = pd.read_csv(path_topics).reset_index(drop=True)
        # Solo agregar columnas que no existan ya en df (evita conflictos de merge)
        cols = [
            c for c in ('topic', 'location', 'lang')
            if c in df_topics.columns and c not in df.columns
        ]
        if cols:
            df = df.join(df_topics[cols])
    else:
        logger.warning('docs_with_topics.csv no encontrado — sin columnas topic/location/lang')
    """
    logger.info('Corpus cargado: %d documentos, columnas: %s', len(df), df.columns.tolist())
    return df


def _detectar_isolation_forest(
    X: np.ndarray,
    contamination: float,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Isolation Forest: aísla puntos anómalos en pocos cortes de árbol.

    Retorna (máscara de outliers, scores de anomalía).
    Score más negativo indica mayor anomalía.
    '''
    clf = IsolationForest(contamination=contamination, random_state=Params.RANDOM_STATE, n_jobs=-1)
    clf.fit(X)
    scores = clf.decision_function(X)
    mask   = clf.predict(X) == -1
    logger.info(
        'Isolation Forest → %d outliers (%.1f%%, contamination=%.3f)',
        mask.sum(), 100 * mask.mean(), contamination,
    )
    return mask, scores


def _detectar_svm(
    X: np.ndarray,
    nu: float,
    max_train: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if max_train is None:
        max_train = Params.OUTLIER_SVM_MAX_TRAIN
    '''
    One-Class SVM con kernel RBF: frontera de decisión de alto margen.

    Para corpus grandes (n > max_train) el kernel RBF es O(n²) en memoria
    y tiempo. Se entrena sobre un submuestreo aleatorio de max_train docs
    y se predice sobre todos — el modelo aprendido generaliza bien.
    Retorna (máscara de outliers, scores de decisión).
    '''
    n = len(X)
    if n > max_train:
        rng = np.random.default_rng(Params.RANDOM_STATE)
        idx_train = rng.choice(n, size=max_train, replace=False)
        X_train = X[idx_train]
        logger.info(
            'One-Class SVM: corpus grande (%d docs) — entrenando sobre %d muestras',
            n, max_train,
        )
    else:
        X_train = X

    clf = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    clf.fit(X_train)
    scores = clf.decision_function(X)
    mask   = clf.predict(X) == -1
    logger.info(
        'One-Class SVM → %d outliers (%.1f%%, nu=%.3f)',
        mask.sum(), 100 * mask.mean(), nu,
    )
    return mask, scores


def _detectar_knn_lof(
    X: np.ndarray,
    n_neighbors: int,
    contamination: float,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Local Outlier Factor: compara la densidad local de cada punto
    con la de sus k vecinos más cercanos.

    Un punto es outlier si su densidad es significativamente menor que
    la de sus vecinos (LOF >> 1).
    Retorna (máscara de outliers, factores LOF negativos — más negativo = más anómalo).
    '''
    clf   = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
    preds = clf.fit_predict(X)
    mask  = preds == -1
    logger.info(
        'KNN/LOF → %d outliers (%.1f%%, n_neighbors=%d)',
        mask.sum(), 100 * mask.mean(), n_neighbors,
    )
    return mask, clf.negative_outlier_factor_


def _ensemble_outliers(mascaras: list[np.ndarray], umbral: int) -> np.ndarray:
    '''
    Combina máscaras de outliers por mayoría de votos.

    umbral controla cuántos modelos deben coincidir para declarar outlier.
    '''
    votos = np.stack(mascaras, axis=1).sum(axis=1)
    return votos >= umbral


def _frecuencias_ngrams(textos: list[str], n: int) -> pd.DataFrame:
    '''
    Genera tabla de frecuencias de n-gramas sobre los textos dados.

    Retorna DataFrame con columnas: ngram, total_frequency, relative_frequency.
    '''
    freq: Counter = Counter()

    for text in textos:
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            continue
        tokens = [t for t in text.split() if len(t) >= Params.VOCAB_MIN_TOKEN_LEN]
        if len(tokens) >= n:
            freq.update(' '.join(g) for g in nltk_ngrams(tokens, n))

    if not freq:
        logger.warning('Sin n-gramas para n=%d en los outliers', n)
        return pd.DataFrame(columns=['ngram', 'total_frequency', 'relative_frequency'])

    total = sum(freq.values())
    rows  = [(ng, c, round(c / total, 6)) for ng, c in freq.most_common()]
    return pd.DataFrame(rows, columns=['ngram', 'total_frequency', 'relative_frequency'])


def run_outlier_detection() -> dict[str, object]:
    '''
    Pipeline completo de detección de outliers.

    Aplica Isolation Forest, One-Class SVM y KNN/LOF sobre los embeddings.
    Combina sus resultados por mayoría de votos para producir una etiqueta
    final es_outlier. Sobre los outliers detectados genera análisis de
    uni-, bi- y trigramas.

    Retorna:
        {
            "outliers"  : DataFrame de documentos atípicos,
            "normales"  : DataFrame de documentos normales,
            "unigrams"  : DataFrame de frecuencias de unigramas,
            "bigrams"   : DataFrame de frecuencias de bigramas,
            "trigrams"  : DataFrame de frecuencias de trigramas,
            "resumen"   : dict con métricas globales,
        }
    '''
    _OUTLIERS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info('=== DETECCIÓN DE OUTLIERS — INICIO ===')

    embeddings = _cargar_embeddings()
    df         = _cargar_corpus()

    if len(df) != len(embeddings):
        raise ValueError(
            f'Desalineación corpus/embeddings: {len(df)} docs vs {len(embeddings)} vectores. '
            'Regenera los embeddings ejecutando el step 05 (semantic).'
        )

    # Estandarizar — mejora estabilidad de SVM y LOF en espacio de alta dimensión
    X = StandardScaler().fit_transform(embeddings)

    contamination = Params.OUTLIER_CONTAMINATION
    n_neighbors   = Params.OUTLIER_KNN_NEIGHBORS
    umbral        = Params.OUTLIER_UMBRAL_ENSEMBLE

    mask_if,  score_if  = _detectar_isolation_forest(X, contamination)
    mask_svm, score_svm = _detectar_svm(X, nu=contamination, max_train=Params.OUTLIER_SVM_MAX_TRAIN)
    mask_lof, score_lof = _detectar_knn_lof(X, n_neighbors, contamination)

    mask_final = _ensemble_outliers([mask_if, mask_svm, mask_lof], umbral)
    logger.info(
        'Ensemble (umbral=%d/3) → %d outliers totales (%.1f%%)',
        umbral, mask_final.sum(), 100 * mask_final.mean(),
    )

    df_res = df.copy()
    df_res['outlier_isolation_forest'] = mask_if.astype(int)
    df_res['outlier_svm']              = mask_svm.astype(int)
    df_res['outlier_lof']              = mask_lof.astype(int)
    df_res['score_isolation_forest']   = score_if
    df_res['score_svm']                = score_svm
    df_res['score_lof']                = score_lof

    # Señal HDBSCAN como indicador informativo adicional (no entra en el ensemble)
    if 'topic' in df.columns:
        hdbscan_noise = (df['topic'] == -1).astype(int)
        df_res['hdbscan_noise'] = hdbscan_noise
        logger.info(
            'HDBSCAN noise (topic=-1): %d documentos (%.1f%%) — informativo',
            hdbscan_noise.sum(), 100 * hdbscan_noise.mean(),
        )

    df_res['votos_outlier'] = (
        mask_if.astype(int) + mask_svm.astype(int) + mask_lof.astype(int)
    )
    df_res['es_outlier'] = mask_final.astype(int)

    df_outliers = df_res[df_res['es_outlier'] == 1].copy()
    df_normales = df_res[df_res['es_outlier'] == 0].copy()

    col_texto = Params.COLUMNA_TEXTO
    if col_texto in df_outliers.columns and not df_outliers.empty:
        textos = df_outliers[col_texto].dropna().tolist()
        logger.info('Generando n-gramas sobre %d comentarios atípicos...', len(textos))
        df_uni = _frecuencias_ngrams(textos, n=1)
        df_bi  = _frecuencias_ngrams(textos, n=2)
        df_tri = _frecuencias_ngrams(textos, n=3)
        logger.info(
            'N-gramas generados: uni=%d | bi=%d | tri=%d',
            len(df_uni), len(df_bi), len(df_tri),
        )
    else:
        logger.warning('Sin texto limpio disponible o sin outliers — n-gramas vacíos')
        df_uni = df_bi = df_tri = pd.DataFrame(
            columns=['ngram', 'total_frequency', 'relative_frequency']
        )

    resumen = {
        'total_documentos'       : len(df),
        'total_outliers'         : int(mask_final.sum()),
        'total_normales'         : int((~mask_final).sum()),
        'pct_outliers'           : float(round(100 * mask_final.mean(), 2)),
        'outliers_isolation_forest': int(mask_if.sum()),
        'outliers_svm'           : int(mask_svm.sum()),
        'outliers_lof'           : int(mask_lof.sum()),
        'ngrams_unigrams'        : len(df_uni),
        'ngrams_bigrams'         : len(df_bi),
        'ngrams_trigrams'        : len(df_tri),
    }

    df_outliers.to_csv(Paths.OUTLIERS_CSV,          index=False, encoding='utf-8-sig')
    df_normales.to_csv(Paths.NORMALES_CSV,           index=False, encoding='utf-8-sig')
    df_uni.to_csv(Paths.OUTLIERS_NGRAMS_UNI_CSV,    index=False, encoding='utf-8-sig')
    df_bi.to_csv(Paths.OUTLIERS_NGRAMS_BI_CSV,      index=False, encoding='utf-8-sig')
    df_tri.to_csv(Paths.OUTLIERS_NGRAMS_TRI_CSV,    index=False, encoding='utf-8-sig')
    pd.DataFrame([resumen]).to_csv(Paths.OUTLIERS_RESUMEN_CSV, index=False, encoding='utf-8-sig')

    logger.info('Outliers   → %s (%d docs)', Paths.OUTLIERS_CSV,       len(df_outliers))
    logger.info('Normales   → %s (%d docs)', Paths.NORMALES_CSV,        len(df_normales))
    logger.info('N-gramas   → %s', _OUTLIERS_DIR)
    logger.info('Resumen    → %s', resumen)
    logger.info('=== DETECCIÓN DE OUTLIERS — FIN ===')

    return {
        'outliers' : df_outliers,
        'normales' : df_normales,
        'unigrams' : df_uni,
        'bigrams'  : df_bi,
        'trigrams' : df_tri,
        'resumen'  : resumen,
    }


if __name__ == '__main__':
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S',
    )
    run_outlier_detection()
