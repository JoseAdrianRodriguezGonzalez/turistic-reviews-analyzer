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

# Hiperparámetros del grid search
K_RANGO_DEFAULT  = range(2, 11)
METODOS_DEFAULT  = ['ward', 'complete', 'average', 'single']

# Para n > este umbral se usa grafo kNN sparse en lugar de matriz densa.
# Sin connectivity: O(n²) memoria.  Con kNN k=Params.JERARQUICO_KNN_VECINOS: O(n·k).
_UMBRAL_SPARSE   = 10_000
# complete y single no admiten connectivity en sklearn — solo ward y average
_METODOS_SPARSE  = ['ward', 'average']


def _construir_connectivity(X: np.ndarray) -> object:
    '''
    Grafo kNN sparse (n × n con n·k entradas no nulas).
    Usar como connectivity en AgglomerativeClustering evita calcular
    la matriz de distancias densa O(n²) — algoritmo interno: Borůvka.
    '''
    k = Params.JERARQUICO_KNN_VECINOS
    logger.info(
        'Corpus grande (%d docs) — usando grafo kNN k=%d como connectivity '
        '(~%.1f MB vs ~%.1f GB con matriz densa)',
        len(X),
        k,
        len(X) * k * 8 / 1e6,
        len(X) ** 2 * 8 / 2 / 1e9,
    )
    return kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)


def evaluar_jerarquico(
    X: np.ndarray,
    k_rango: range  = K_RANGO_DEFAULT,
    metodos: list   = METODOS_DEFAULT,
) -> list[dict]:
    '''
    Grid search sobre (método de enlace, k) para Clustering Jerárquico.
    Devuelve lista de dicts con métricas por combinación, lista para
    consolidar en el orquestador.

    Para corpus con n > _UMBRAL_SPARSE usa un grafo kNN sparse como
    connectivity, evitando la matriz de distancias O(n²). En ese modo
    solo se evalúan los métodos que soportan connectivity ('ward', 'average').

    Cada dict contiene:
        modelo, score_ranking, silhouette, n_clusters,
        n_ruido, hiperparametros, codo_k, _etiquetas (list[int])

    Parámetros:
        X       -- matriz reducida (n_docs x n_dims)
        k_rango -- rango de valores de k a evaluar
        metodos -- lista de métodos de enlace a probar
    '''
    usar_sparse  = len(X) > _UMBRAL_SPARSE
    connectivity = None
    metodos_activos = metodos

    if usar_sparse:
        connectivity    = _construir_connectivity(X)
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

            sil = silhouette_score(X, etiq)
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