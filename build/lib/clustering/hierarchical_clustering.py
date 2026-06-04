'''
hierarchical_clustering.py
--------------------------
Bloque 6 — Clustering Jerárquico (AgglomerativeClustering):
grid search sobre k y método de enlace (linkage).

Score: silhouette directo (único criterio disponible sin inercia).

Funciones públicas:
    evaluar_jerarquico(X, k_rango, metodos) -> list[dict]
'''

import logging

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

from config import Params

logger = logging.getLogger(__name__)

# Métodos que soportan connectivity en sklearn (restricción de la librería)
_METODOS_SPARSE = ['ward', 'average']


def _construir_connectivity(X: np.ndarray) -> object:
    '''
    Grafo kNN sparse como connectivity para AgglomerativeClustering.
    Evita calcular la matriz de distancias densa O(n²).
    '''
    k = Params.JERARQUICO_KNN_VECINOS
    logger.info(
        'Corpus grande (%d docs) — usando grafo kNN k=%d como connectivity '
        '(~%.1f MB vs ~%.1f GB con matriz densa)',
        len(X), k,
        len(X) * k * 8 / 1e6,
        len(X) ** 2 * 8 / 2 / 1e9,
    )
    return kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)


def _silhouette_jerarquico(X: np.ndarray, etiq: np.ndarray) -> float:
    '''Silhouette con muestreo para n > Params.SILHOUETTE_SAMPLE_THRESHOLD.'''
    if len(X) > Params.SILHOUETTE_SAMPLE_THRESHOLD:
        rng = np.random.default_rng(Params.RANDOM_STATE)
        idx = rng.choice(len(X), size=Params.SILHOUETTE_SAMPLE_SIZE, replace=False)
        return float(silhouette_score(X[idx], etiq[idx]))
    return float(silhouette_score(X, etiq))


def evaluar_jerarquico(
    X: np.ndarray,
    k_rango: range | None  = None,
    metodos: list | None   = None,
) -> list[dict]:
    '''
    Grid search sobre (método de enlace, k) para Clustering Jerárquico.

    Para corpus con n > Params.CLUSTERING_SPARSE_THRESHOLD usa un grafo kNN
    sparse como connectivity, evitando la matriz de distancias O(n²). En ese
    modo solo se evalúan los métodos que soportan connectivity ('ward', 'average').

    Parámetros:
        X       -- matriz reducida (n_docs x n_dims)
        k_rango -- default: Params.K_RANGO
        metodos -- default: Params.METODOS_JERARQUICO
    '''
    if k_rango is None:
        k_rango = Params.K_RANGO
    if metodos is None:
        metodos = Params.METODOS_JERARQUICO

    usar_sparse  = len(X) > Params.CLUSTERING_SPARSE_THRESHOLD
    connectivity = None
    metodos_activos = metodos

    if usar_sparse:
        connectivity    = _construir_connectivity(X)
        # Pre-conectar el grafo una sola vez para evitar el warning de sklearn
        # "n_connected_components > 1" que se repite en cada llamada al loop.
        from sklearn.cluster._agglomerative import _fix_connectivity
        connectivity, n_comp = _fix_connectivity(X, connectivity, Params.HIERARCHY_LINKAGE_METRIC)
        if n_comp > 1:
            logger.info(
                'Grafo kNN tenía %d componentes desconectados — conectado automáticamente. '
                'Considera aumentar jerarquico_knn_vecinos en el YAML para evitarlo.',
                n_comp,
            )
        metodos_activos = [m for m in metodos if m in _METODOS_SPARSE]
        omitidos        = [m for m in metodos if m not in _METODOS_SPARSE]
        if omitidos:
            logger.info(
                'Métodos omitidos en modo sparse (no admiten connectivity): %s', omitidos
            )

    logger.info('Jerárquico: grid search k=%s, métodos=%s', list(k_rango), metodos_activos)

    filas = []

    for metodo in metodos_activos:
        for k in k_rango:
            modelo = AgglomerativeClustering(
                n_clusters=k, linkage=metodo, connectivity=connectivity
            )
            etiq = modelo.fit_predict(X)

            if len(set(etiq)) < 2:
                logger.debug('Jerárquico k=%d metodo=%s: menos de 2 clusters, saltando', k, metodo)
                continue

            sil = _silhouette_jerarquico(X, etiq)
            logger.debug('Jerárquico k=%d metodo=%s | silhouette=%.4f', k, metodo, sil)

            filas.append({
                'modelo'         : 'jerarquico',
                'score_ranking'  : round(sil, 6),
                'silhouette'     : round(sil, 6),
                'inercia'        : None,
                'n_clusters'     : k,
                'n_ruido'        : 0,
                'hiperparametros': f'k={k},metodo={metodo},sparse={usar_sparse}',
                'codo_k'         : None,
                '_etiquetas'     : etiq.tolist(),
            })

    if filas:
        mejor = max(filas, key=lambda r: r['score_ranking'])
        logger.info('Jerárquico: mejor %s | score=%.4f | silhouette=%.4f',
                    mejor['hiperparametros'], mejor['score_ranking'], mejor['silhouette'])
    else:
        logger.warning('Jerárquico: no se generaron resultados válidos')

    return filas
