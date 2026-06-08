'''
clustering_pipeline.py
----------------------
Bloque 6 — Orquestador: ejecuta el grid search de KMeans, Jerárquico y HDBSCAN
sobre TRES fuentes de representación vectorial distintas y produce una tabla
comparativa consolidada al final.

Fuentes procesadas:
    1. embeddings  -- vectores densos de BERTopic (docs_with_topics.npy)
                      Reducción: UMAP
    2. features    -- features numéricas NLP calculadas por Jorge (features_nlp.csv)
                      Reducción: StandardScaler + UMAP
    3. tfidf_yake  -- matrices dispersas TF-IDF y YAKE de Adrian (*.pkl)
                      Reducción: TruncatedSVD + UMAP

Estructura de salida en data/clustering/:
    embeddings/
        ranking_completo.csv
        mejores_modelos.csv
        etiquetas_mejores.json
    features/
        ranking_completo.csv
        mejores_modelos.csv
        etiquetas_mejores.json
    tfidf/
        ranking_completo.csv
        mejores_modelos.csv
        etiquetas_mejores.json
    yake/
        ranking_completo.csv
        mejores_modelos.csv
        etiquetas_mejores.json
    comparacion_fuentes.csv   <-- tabla resumen cruzada al final

Uso desde main.py:
    from clustering.clustering_pipeline import run_clustering_pipeline
    run_clustering_pipeline()
'''

import gc
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from clustering.hdbscan_clustering      import evaluar_hdbscan
from clustering.hierarchical_clustering import evaluar_jerarquico
from clustering.kmeans_clustering       import evaluar_kmeans
from config import Params, Paths


def _liberar_memoria() -> None:
    '''Libera workers de joblib y fuerza GC para evitar acumulación entre fuentes.'''
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=False)
    except Exception:
        pass
    gc.collect()

logger = logging.getLogger(__name__)


def _reducir_umap(X: np.ndarray, nombre_fuente: str) -> np.ndarray:
    '''
    Aplica UMAP para reducir cualquier matriz densa a UMAP_COMPONENTS dimensiones.
    Se usa como paso final para embeddings y features NLP ya normalizadas.
    '''
    umap_dims = Params.UMAP_COMPONENTS
    logger.info('%s — UMAP: %s -> %d dims', nombre_fuente, X.shape, umap_dims)
    reducer = UMAP(n_components=umap_dims, random_state=Params.RANDOM_STATE)
    X_red   = reducer.fit_transform(X)
    logger.info('%s — UMAP completado. Shape resultante: %s', nombre_fuente, X_red.shape)
    return X_red


def _preparar_embeddings(path: Path) -> np.ndarray:
    '''
    Carga los embeddings densos de BERTopic y los reduce con UMAP.
    Los embeddings ya son vectores continuos, solo necesitan reducción dimensional.
    '''
    logger.info('Fuente embeddings: cargando desde %s', path)
    embeddings = np.load(path)
    logger.info('Embeddings cargados. Shape original: %s', embeddings.shape)
    return _reducir_umap(embeddings, 'embeddings')


def _preparar_features_nlp(path: Path) -> np.ndarray:
    '''
    Carga features_nlp.csv, elimina columnas de metadatos, normaliza con
    StandardScaler y reduce con UMAP.

    El escalado es necesario porque las columnas tienen escalas muy distintas
    (ej: token_count en decenas vs pos_ratio_noun entre 0 y 1).
    '''
    logger.info('Fuente features: cargando desde %s', path)
    df = pd.read_csv(path)

    # Eliminar columnas de metadatos que no son features numéricas
    columnas_a_eliminar = [c for c in Params.CLUSTERING_METADATA_COLS if c in df.columns]
    df = df.drop(columns=columnas_a_eliminar)

    # Eliminar filas con NaN (pueden existir si algún documento no tuvo POS tags)
    df = df.dropna()

    # Convertir a matriz numpy de float64
    X = df.values.astype(np.float64)
    logger.info('Features NLP cargadas. Shape: %s | Columnas: %s', X.shape, list(df.columns))

    # Normalizar: cada feature queda con media 0 y desviación estándar 1
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    logger.info('Features NLP normalizadas con StandardScaler')

    return _reducir_umap(X_norm, 'features')


def _cargar_corpus_limpio(path: Path) -> list[str]:
    '''
    Lee clean.csv y devuelve la lista de textos limpios como strings.
    Se usa para reconstruir las matrices dispersas desde los pkl de Adrian.
    '''
    logger.info('Cargando corpus limpio desde %s', path)
    df    = pd.read_csv(path)
    textos = df[Params.COLUMNA_TEXTO].fillna('').astype(str).tolist()
    logger.info('Corpus cargado: %d documentos', len(textos))
    return textos


def _preparar_matriz_dispersa(path_pkl: Path, nombre_fuente: str) -> np.ndarray:
    '''
    Carga un vectorizador sklearn desde pkl, transforma el corpus limpio
    para obtener la matriz dispersa, aplica TruncatedSVD para densificarla
    y finalmente UMAP para reducir a 2 dimensiones.

    TruncatedSVD es el equivalente a PCA para matrices dispersas: no requiere
    centrar los datos y es eficiente con vocabularios grandes.

    Parámetros:
        path_pkl      -- ruta al .pkl con el vectorizador (TfidfVectorizer o CountVectorizer)
        nombre_fuente -- etiqueta para los logs ('tfidf' o 'yake')
    '''
    logger.info('%s: cargando vectorizador desde %s', nombre_fuente, path_pkl)
    vectorizador = joblib.load(path_pkl)

    textos = _cargar_corpus_limpio(Paths.NORMALIZED_SPANISH_CSV)

    logger.info('%s: transformando corpus con el vectorizador...', nombre_fuente)
    X_dispersa = vectorizador.transform(textos)
    logger.info('%s: matriz dispersa generada. Shape: %s', nombre_fuente, X_dispersa.shape)

    # Determinar cuántos componentes SVD son posibles dado el vocabulario
    n_componentes_svd = min(Params.SVD_COMPONENTS, X_dispersa.shape[1] - 1, X_dispersa.shape[0] - 1)
    logger.info('%s: aplicando TruncatedSVD con %d componentes', nombre_fuente, n_componentes_svd)

    svd   = TruncatedSVD(n_components=n_componentes_svd, random_state=Params.RANDOM_STATE)
    X_svd = svd.fit_transform(X_dispersa)

    varianza_explicada = svd.explained_variance_ratio_.sum()
    logger.info('%s: SVD completado. Varianza explicada: %.2f%%', nombre_fuente, varianza_explicada * 100)

    return _reducir_umap(X_svd, nombre_fuente)


def _guardar_proyeccion_2d(X_red: np.ndarray, dir_fuente: Path) -> None:
    '''
    Guarda la matriz 2D reducida por UMAP como proyeccion_2d.npy en la
    subcarpeta de la fuente, para que clustering_visualizacion.py la
    reutilice en las gráficas de scatter sin recalcular UMAP.

    Esto garantiza que los puntos del scatter corresponden exactamente
    al espacio donde se midió el silhouette y se asignaron los clusters.
    '''
    dir_fuente.mkdir(parents=True, exist_ok=True)
    path_out = dir_fuente / Paths.CLUSTERING_PROYECCION_FILE
    np.save(path_out, X_red)
    logger.info('Proyección 2D guardada: %s  shape=%s', path_out, X_red.shape)


def _exportar_resultados_fuente(todos: list[dict], dir_fuente: Path) -> pd.DataFrame:
    '''
    Exporta ranking_completo.csv, mejores_modelos.csv y etiquetas_mejores.json
    para una fuente específica dentro de su subcarpeta.

    Es una versión de la función original adaptada para recibir la ruta
    de salida como parámetro en lugar de usar una ruta global fija.

    Retorna el DataFrame de mejores modelos (rank == 1 por algoritmo).
    '''
    dir_fuente.mkdir(parents=True, exist_ok=True)

    # Separar etiquetas del resto de métricas antes de armar el DataFrame
    etiquetas_dict: dict[str, list[int]] = {}
    filas_csv: list[dict] = []

    for row in todos:
        etiq = row.pop('_etiquetas')
        key  = f"{row['modelo']}|{row['hiperparametros']}"
        etiquetas_dict[key] = etiq
        filas_csv.append(row)

    df = pd.DataFrame(filas_csv)
    df = df.sort_values(
        ['modelo', 'score_ranking'],
        ascending=[True, False]
    ).reset_index(drop=True)

    df['rank'] = (
        df.groupby('modelo')['score_ranking']
          .rank(ascending=False, method='first')
          .astype(int)
    )

    columnas_orden = [
        'rank', 'modelo', 'score_ranking', 'silhouette',
        'n_clusters', 'n_ruido', 'inercia', 'codo_k', 'hiperparametros'
    ]
    df = df[[c for c in columnas_orden if c in df.columns]]

    path_ranking = dir_fuente / Paths.CLUSTERING_RANKING_FILE
    df.to_csv(path_ranking, index=False, encoding='utf-8-sig')
    logger.info('ranking_completo.csv guardado en %s (%d combinaciones)', dir_fuente, len(df))

    df_mejores = (
        df[df['rank'] == 1]
          .sort_values('modelo')
          .reset_index(drop=True)
    )
    path_mejores = dir_fuente / Paths.CLUSTERING_MEJORES_FILE
    df_mejores.to_csv(path_mejores, index=False, encoding='utf-8-sig')
    logger.info('mejores_modelos.csv guardado en %s (%d modelos)', dir_fuente, len(df_mejores))

    # Guardar solo las etiquetas de los mejores modelos para graficado posterior
    etiquetas_mejores: dict[str, list[int]] = {}
    for _, row in df_mejores.iterrows():
        key = f"{row['modelo']}|{row['hiperparametros']}"
        if key in etiquetas_dict:
            etiquetas_mejores[key] = etiquetas_dict[key]

    path_etiq = dir_fuente / Paths.CLUSTERING_ETIQUETAS_FILE
    with open(path_etiq, 'w', encoding='utf-8') as f:
        json.dump(etiquetas_mejores, f)
    logger.info('etiquetas_mejores.json guardado en %s', dir_fuente)

    return df_mejores


def _exportar_comparacion_fuentes(resultados_por_fuente: dict[str, pd.DataFrame]) -> None:
    '''
    Genera comparacion_fuentes.csv: una fila por (fuente, algoritmo) con las
    métricas del mejor modelo de cada combinación.

    Esto permite comparar de un vistazo qué representación vectorial produce
    los clusters más coherentes según silhouette y score_ranking.
    '''
    filas = []
    for nombre_fuente, df_mejores in resultados_por_fuente.items():
        for _, row in df_mejores.iterrows():
            filas.append({
                'fuente'         : nombre_fuente,
                'modelo'         : row['modelo'],
                'score_ranking'  : row['score_ranking'],
                'silhouette'     : row['silhouette'],
                'n_clusters'     : row['n_clusters'],
                'n_ruido'        : row.get('n_ruido', 0),
                'hiperparametros': row['hiperparametros'],
            })

    df_comp = pd.DataFrame(filas).sort_values(
        ['modelo', 'silhouette'],
        ascending=[True, False]
    ).reset_index(drop=True)

    path_comp = Paths.CLUSTERING_COMPARACION_CSV
    Paths.CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(path_comp, index=False, encoding='utf-8-sig')

    logger.info('\n--- COMPARACION ENTRE FUENTES ---\n%s', df_comp.to_string(index=False))
    logger.info('comparacion_fuentes.csv guardado en %s', Paths.CLUSTERING_DIR)


def _ejecutar_grid_search(
    X_red: np.ndarray,
    ejecutar_hdbscan: bool,
    ejecutar_jerarquico: bool = True,
) -> list[dict]:
    '''
    Ejecuta KMeans, Jerárquico y opcionalmente HDBSCAN sobre la matriz
    ya reducida X_red. Devuelve la lista consolidada de resultados.

    Se separa en función propia para evitar duplicar las 10 líneas de
    lógica de control en cada bloque de fuente.
    '''
    todos: list[dict] = []

    todos += evaluar_kmeans(X_red)

    if ejecutar_jerarquico:
        todos += evaluar_jerarquico(X_red)
    else:
        logger.info('Clustering jerárquico omitido por configuración')

    if ejecutar_hdbscan:
        resultados_hdbscan = evaluar_hdbscan(X_red)
        if resultados_hdbscan:
            todos += resultados_hdbscan
        else:
            logger.warning('HDBSCAN no produjo resultados válidos para esta fuente; se omite')
    else:
        logger.info('HDBSCAN omitido por configuración')

    return todos

def run_clustering_pipeline(
    ejecutar_hdbscan    : bool = True,
    ejecutar_jerarquico : bool = True,
    ejecutar_embeddings : bool = True,
    ejecutar_features   : bool = True,
    ejecutar_tfidf      : bool = True,
    ejecutar_yake       : bool = True,
) -> dict[str, pd.DataFrame]:
    '''
    Pipeline completo de clustering multi-fuente.

    Ejecuta el grid search (KMeans + Jerárquico + HDBSCAN) sobre cada una
    de las representaciones vectoriales disponibles y exporta resultados
    separados por fuente más una tabla comparativa consolidada.

    Parámetros (todos opcionales — permiten activar/desactivar fuentes):
        ejecutar_hdbscan    -- False para omitir HDBSCAN en todas las fuentes
        ejecutar_jerarquico -- False para omitir clustering jerárquico (recomendado
                               en corpus grandes, requiere O(n²) memoria)
        ejecutar_embeddings -- False para omitir la fuente de embeddings BERTopic
        ejecutar_features   -- False para omitir features_nlp.csv
        ejecutar_tfidf      -- False para omitir la matriz TF-IDF
        ejecutar_yake       -- False para omitir la matriz YAKE

    Retorna diccionario {nombre_fuente: DataFrame_mejores_modelos}.
    '''
    # Resultados acumulados por fuente para la comparación final
    resultados_por_fuente: dict[str, pd.DataFrame] = {}

    if ejecutar_embeddings:
        if not Paths.DOCS_WITH_TOPICS_NPY.exists():
            logger.warning('Embeddings no encontrados en %s — fuente omitida', Paths.DOCS_WITH_TOPICS_NPY)
        else:
            logger.info('=== FUENTE: embeddings BERTopic ===')
            X_red   = _preparar_embeddings(Paths.DOCS_WITH_TOPICS_NPY)
            _guardar_proyeccion_2d(X_red, Paths.CLUSTERING_EMBEDDINGS_DIR)
            todos   = _ejecutar_grid_search(X_red, ejecutar_hdbscan, ejecutar_jerarquico)
            mejores = _exportar_resultados_fuente(todos, Paths.CLUSTERING_EMBEDDINGS_DIR)
            resultados_por_fuente['embeddings'] = mejores
            del X_red, todos
            _liberar_memoria()
    else:
        logger.info('Fuente embeddings omitida por configuración')

    if ejecutar_features:
        if not Paths.FEATURES_NLP_CSV.exists():
            logger.warning('features_nlp.csv no encontrado en %s — fuente omitida', Paths.FEATURES_NLP_CSV)
        else:
            logger.info('=== FUENTE: features NLP ===')
            X_red   = _preparar_features_nlp(Paths.FEATURES_NLP_CSV)
            _guardar_proyeccion_2d(X_red, Paths.CLUSTERING_FEATURES_DIR)
            todos   = _ejecutar_grid_search(X_red, ejecutar_hdbscan, ejecutar_jerarquico)
            mejores = _exportar_resultados_fuente(todos, Paths.CLUSTERING_FEATURES_DIR)
            resultados_por_fuente['features'] = mejores
            del X_red, todos
            _liberar_memoria()
    else:
        logger.info('Fuente features omitida por configuración')

    if ejecutar_tfidf:
        if not Paths.TFIDF_PKL.exists():
            logger.warning('tfidf.pkl no encontrado en %s — fuente omitida', Paths.TFIDF_PKL)
        elif not Paths.NORMALIZED_SPANISH_CSV.exists():
            logger.warning('normalized_spanish.csv no encontrado — fuente tfidf omitida')
        else:
            logger.info('=== FUENTE: TF-IDF ===')
            X_red   = _preparar_matriz_dispersa(Paths.TFIDF_PKL, 'tfidf')
            _guardar_proyeccion_2d(X_red, Paths.CLUSTERING_TFIDF_DIR)
            todos   = _ejecutar_grid_search(X_red, ejecutar_hdbscan, ejecutar_jerarquico)
            mejores = _exportar_resultados_fuente(todos, Paths.CLUSTERING_TFIDF_DIR)
            resultados_por_fuente['tfidf'] = mejores
            del X_red, todos
            _liberar_memoria()
    else:
        logger.info('Fuente tfidf omitida por configuración')

    if ejecutar_yake:
        if not Paths.YAKE_PKL.exists():
            logger.warning('yake_vectorizer.pkl no encontrado en %s — fuente omitida', Paths.YAKE_PKL)
        elif not Paths.NORMALIZED_SPANISH_CSV.exists():
            logger.warning('normalized_spanish.csv no encontrado — fuente yake omitida')
        else:
            logger.info('=== FUENTE: YAKE ===')
            X_red   = _preparar_matriz_dispersa(Paths.YAKE_PKL, 'yake')
            _guardar_proyeccion_2d(X_red, Paths.CLUSTERING_YAKE_DIR)
            todos   = _ejecutar_grid_search(X_red, ejecutar_hdbscan, ejecutar_jerarquico)
            mejores = _exportar_resultados_fuente(todos, Paths.CLUSTERING_YAKE_DIR)
            resultados_por_fuente['yake'] = mejores
            del X_red, todos
            _liberar_memoria()
    else:
        logger.info('Fuente yake omitida por configuración')

    if resultados_por_fuente:
        _exportar_comparacion_fuentes(resultados_por_fuente)
    else:
        logger.warning('Ninguna fuente produjo resultados; comparacion_fuentes.csv no generado')
    # Sincronizar Params activos con el mejor modelo encontrado por el grid search
    fuente_activa = Params.CLUSTERING_ACTIVO_FUENTE
    modelo_activo = Params.CLUSTERING_ACTIVO_MODELO
    if fuente_activa in resultados_por_fuente:
        df_activa = resultados_por_fuente[fuente_activa]
        fila = df_activa[df_activa['modelo'] == modelo_activo]
        if not fila.empty:
            mejor = fila.sort_values('score_ranking', ascending=False).iloc[0]
            k_nuevo = int(mejor['hiperparametros'].split('k=')[1].split(',')[0])
            key_nuevo = f"{modelo_activo}|k={k_nuevo}"
            Params.CLUSTERING_ACTIVO_K   = k_nuevo
            Params.CLUSTERING_ACTIVO_KEY = key_nuevo
            logger.info(
                'Params activos actualizados: CLUSTERING_ACTIVO_K=%d, CLUSTERING_ACTIVO_KEY=%s',
                k_nuevo, key_nuevo,
            )
    config_path = Paths.ENRICHMENT_DIR / "active_model.json"

    best_model_info = {
        "fuente": fuente_activa,
        "modelo": key_nuevo,
        "k": k_nuevo
    }

    with open(config_path, "w") as f:
        json.dump(best_model_info, f)
    return resultados_por_fuente


if __name__ == '__main__':
    run_clustering_pipeline()
