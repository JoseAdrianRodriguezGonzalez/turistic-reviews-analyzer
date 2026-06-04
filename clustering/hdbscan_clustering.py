'''
hdbscan_clustering.py
---------------------
Bloque 6 — HDBSCAN: grid search sobre min_cluster_size y min_samples.

Score: silhouette * (1 - pct_ruido) * penalizacion_balance

Funciones públicas:
    evaluar_hdbscan(X, min_cluster_sizes, min_samples_list) -> list[dict]
    silhouette_score_safe(X, etiq) -> float | None
'''

import logging

import numpy as np

from config import Params

logger = logging.getLogger(__name__)


def silhouette_score_safe(X: np.ndarray, etiq: np.ndarray) -> float | None:
    '''
    Calcula silhouette_score con muestreo para n > Params.SILHOUETTE_SAMPLE_THRESHOLD.
    Retorna None si el cálculo falla (ej. todos los puntos en un cluster).
    '''
    from sklearn.metrics import silhouette_score
    try:
        if len(X) > Params.SILHOUETTE_SAMPLE_THRESHOLD:
            rng = np.random.default_rng(Params.RANDOM_STATE)
            idx = rng.choice(len(X), size=Params.SILHOUETTE_SAMPLE_SIZE, replace=False)
            return float(silhouette_score(X[idx], etiq[idx]))
        return float(silhouette_score(X, etiq))
    except ValueError as e:
        logger.warning('silhouette_score falló: %s', e)
        return None


def evaluar_hdbscan(
    X: np.ndarray,
    min_cluster_sizes: list[int] | None = None,
    min_samples_list : list[int] | None = None,
) -> list[dict]:
    '''
    Grid search sobre (min_cluster_size, min_samples) para HDBSCAN.

    Parámetros:
        X                  -- matriz reducida (n_docs x n_dims)
        min_cluster_sizes  -- default: Params.MIN_CLUSTER_SIZES_HDBSCAN
        min_samples_list   -- default: Params.MIN_SAMPLES_HDBSCAN
    '''
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        logger.error('hdbscan no está instalado. Ejecuta: pip install hdbscan')
        return []

    if min_cluster_sizes is None:
        min_cluster_sizes = Params.MIN_CLUSTER_SIZES_HDBSCAN
    if min_samples_list is None:
        min_samples_list = Params.MIN_SAMPLES_HDBSCAN

    logger.info('HDBSCAN: grid search min_cluster_size=%s, min_samples=%s',
                min_cluster_sizes, min_samples_list)

    n_total = X.shape[0]
    umbral_cluster_minimo = max(
        Params.CLUSTERING_MIN_CLUSTER_ABS,
        int(n_total * Params.CLUSTERING_MIN_CLUSTER_PCT),
    )
    filas = []

    for min_cs in min_cluster_sizes:
        for min_s in min_samples_list:
            modelo = HDBSCAN(min_cluster_size=min_cs, min_samples=min_s)
            etiq   = modelo.fit_predict(X)

            mascara    = etiq != -1
            n_ruido    = int((~mascara).sum())
            n_validos  = int(mascara.sum())
            n_clusters = len(set(etiq[mascara])) if n_validos > 0 else 0

            if n_clusters < 2 or n_validos < 4:
                logger.debug('HDBSCAN min_cs=%d min_s=%d: clusters=%d, saltando',
                             min_cs, min_s, n_clusters)
                continue

            counts = np.bincount(etiq[mascara])

            if counts.max() <= umbral_cluster_minimo:
                continue

            sil = silhouette_score_safe(X[mascara], etiq[mascara])
            if sil is None:
                continue

            pct_ruido    = n_ruido / n_total
            pct_max      = counts.max() / n_validos
            penalizacion = 0.0 if pct_max > Params.MAX_CLUSTER_PCT else 1.0
            score        = sil * (1.0 - pct_ruido) * penalizacion

            if score <= 0:
                continue

            logger.debug('HDBSCAN min_cs=%d min_s=%d | clusters=%d ruido=%d sil=%.4f score=%.4f',
                         min_cs, min_s, n_clusters, n_ruido, sil, score)

            filas.append({
                'modelo'         : 'hdbscan',
                'score_ranking'  : round(score, 6),
                'silhouette'     : round(sil, 6),
                'inercia'        : None,
                'n_clusters'     : n_clusters,
                'n_ruido'        : n_ruido,
                'hiperparametros': f'min_cluster_size={min_cs},min_samples={min_s}',
                'codo_k'         : None,
                '_etiquetas'     : etiq.tolist(),
            })

    if filas:
        mejor = max(filas, key=lambda r: r['score_ranking'])
        logger.info('HDBSCAN: mejor %s | score=%.4f | silhouette=%.4f | clusters=%d | ruido=%d',
                    mejor['hiperparametros'], mejor['score_ranking'],
                    mejor['silhouette'], mejor['n_clusters'], mejor['n_ruido'])
    else:
        logger.warning('HDBSCAN: ninguna combinación produjo resultados válidos')

    return filas
