'''
kmeans_clustering.py
--------------------
Bloque 6 — KMeans: grid search sobre el número de clusters k.

Score combinado: alpha * silhouette_norm + (1 - alpha) * elbow_score
El codo se detecta como la k de máxima curvatura (segunda derivada) de la
curva de inercias.

Funciones públicas:
    evaluar_kmeans(X, k_rango, alpha) -> list[dict]
'''

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import Params

logger = logging.getLogger(__name__)


def _silhouette_aproximado(X: np.ndarray, etiq: np.ndarray, rng: np.random.Generator) -> float:
    '''Silhouette sobre muestra aleatoria cuando n > Params.SILHOUETTE_SAMPLE_THRESHOLD.'''
    if len(X) > Params.SILHOUETTE_SAMPLE_THRESHOLD:
        idx = rng.choice(len(X), size=Params.SILHOUETTE_SAMPLE_SIZE, replace=False)
        return float(silhouette_score(X[idx], etiq[idx]))
    return float(silhouette_score(X, etiq))


def _detectar_codo(inercias: dict, ks: list[int]) -> int:
    '''
    Detecta la k óptima como el punto de máxima curvatura de la curva
    de inercias usando la segunda derivada discreta.
    '''
    iv  = np.array([inercias[k] for k in ks])
    d2  = np.diff(np.diff(iv))
    return ks[int(np.argmax(np.abs(d2))) + 1]


def evaluar_kmeans(
    X: np.ndarray,
    k_rango: range | None = None,
    alpha: float | None   = None,
) -> list[dict]:
    '''
    Grid search sobre k para KMeans.

    Parámetros:
        X       -- matriz reducida (n_docs x n_dims)
        k_rango -- rango de k (default: Params.K_RANGO)
        alpha   -- peso de silhouette en el score combinado (default: Params.ALPHA_KMEANS)
    '''
    if k_rango is None:
        k_rango = Params.K_RANGO
    if alpha is None:
        alpha = Params.ALPHA_KMEANS

    logger.info('KMeans: grid search k=%s, alpha=%.2f', list(k_rango), alpha)
    if len(X) > Params.SILHOUETTE_SAMPLE_THRESHOLD:
        logger.info(
            'n=%d > %d — silhouette aproximado con muestra de %d docs',
            len(X), Params.SILHOUETTE_SAMPLE_THRESHOLD, Params.SILHOUETTE_SAMPLE_SIZE,
        )

    rng         = np.random.default_rng(Params.RANDOM_STATE)
    inercias    : dict[int, float]      = {}
    silhouettes : dict[int, float]      = {}
    etiquetas   : dict[int, np.ndarray] = {}

    for k in k_rango:
        modelo        = KMeans(n_clusters=k, random_state=Params.RANDOM_STATE, n_init='auto')
        etiq          = modelo.fit_predict(X)
        inercias[k]   = modelo.inertia_
        silhouettes[k] = _silhouette_aproximado(X, etiq, rng)
        etiquetas[k]  = etiq
        logger.debug('KMeans k=%d | silhouette=%.4f | inercia=%.2f', k, silhouettes[k], inercias[k])

    ks     = list(k_rango)
    codo_k = _detectar_codo(inercias, ks)
    logger.info('KMeans: codo detectado en k=%d', codo_k)

    elbow_sc = {k: max(0.0, 1.0 - abs(k - codo_k) / len(ks)) for k in ks}

    s_arr  = np.array([silhouettes[k] for k in ks])
    s_norm = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + 1e-9)

    filas = []
    for i, k in enumerate(ks):
        score = alpha * s_norm[i] + (1.0 - alpha) * elbow_sc[k]
        filas.append({
            'modelo'         : 'kmeans',
            'score_ranking'  : round(score, 6),
            'silhouette'     : round(silhouettes[k], 6),
            'inercia'        : round(inercias[k], 4),
            'n_clusters'     : k,
            'n_ruido'        : 0,
            'hiperparametros': f'k={k}',
            'codo_k'         : codo_k,
            '_etiquetas'     : etiquetas[k].tolist(),
        })

    mejor = max(filas, key=lambda r: r['score_ranking'])
    logger.info('KMeans: mejor k=%d | score=%.4f | silhouette=%.4f',
                mejor['n_clusters'], mejor['score_ranking'], mejor['silhouette'])
    return filas
