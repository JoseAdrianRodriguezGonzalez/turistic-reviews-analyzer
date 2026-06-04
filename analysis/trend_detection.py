'''
trend_detection.py
------------------
Bloque 8 — Detección de tendencias y patrones de distribución.

Sin datos de series temporales explícitas en el corpus, la detección
de tendencias opera sobre dos ejes:
    1. Distribución temática por destino: qué tópicos son más frecuentes
       en cada location y qué diferencia a un destino de otro.
    2. Distribución de tópicos por sentimiento: qué temas concentran
       reseñas negativas vs positivas.
    3. Microtópicos: patrones internos dentro de cada tópico-destino
       usando los resultados de pipe_microtopics().

Salidas:
    tendencias_topicos_destino.csv     -- distribución de tópicos por destino
    tendencias_sentimiento_topico.csv  -- perfil de sentimiento por tópico
    microtopicos_resumen.csv           -- resumen de microtópicos por región
    perfil_destino.csv                 -- perfil comparativo entre destinos
'''

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import Params, Paths

logger = logging.getLogger(__name__)


def calcular_distribucion_topicos_destino(
    df: pd.DataFrame,
    topics_meta: pd.DataFrame | None,
) -> pd.DataFrame:
    '''
    Para cada combinación (topic, location) calcula:
        - n_docs: número de documentos
        - pct_en_destino: % del destino que pertenece al tópico
        - pct_del_topico: % del tópico que proviene del destino
        - indice_especificidad: ratio entre pct_del_topico y proporción
          esperada si fuera uniforme — detecta qué tópicos son más
          propios de un destino específico

    Excluye tópico -1 (ruido) del análisis.
    '''
    df_valido = df[df['topic'] != -1].copy()

    tabla = (
        df_valido
        .groupby(['location', 'topic'])
        .size()
        .reset_index(name='n_docs')
    )

    total_por_destino = df_valido.groupby('location').size().rename('total_destino')
    total_por_topico  = df_valido.groupby('topic').size().rename('total_topico')
    total_corpus      = len(df_valido)

    tabla = tabla.merge(total_por_destino.reset_index(), on='location')
    tabla = tabla.merge(total_por_topico.reset_index(), on='topic')

    tabla['pct_en_destino'] = (tabla['n_docs'] / tabla['total_destino'] * 100).round(3)
    tabla['pct_del_topico'] = (tabla['n_docs'] / tabla['total_topico'] * 100).round(3)

    # Proporción esperada si el tópico se distribuyera uniformemente
    proporcion_esperada = tabla['total_topico'] / total_corpus
    tabla['indice_especificidad'] = (
        tabla['pct_del_topico'] / (proporcion_esperada * 100)
    ).round(4)

    # Añadir nombre del tópico si está disponible
    if topics_meta is not None:
        tabla = tabla.merge(
            topics_meta[['Topic', 'Name']].rename(columns={'Topic': 'topic', 'Name': 'topic_name'}),
            on='topic',
            how='left',
        )

    tabla = tabla.sort_values(
        ['location', 'pct_en_destino'],
        ascending=[True, False],
    ).reset_index(drop=True)

    logger.info(
        'Distribución tópicos x destino: %d combinaciones',
        len(tabla),
    )
    return tabla


def calcular_perfil_sentimiento_topico(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Para cada tópico BERTopic calcula el perfil completo de sentimiento:
        - distribución de categorías (negativo/neutro/positivo/sin_etiqueta)
        - sentimiento medio y desviación estándar
        - estrella media
        - intensidad emocional promedio (ratio de adjetivos + adverbios)

    Útil para identificar qué temas generan más polarización o
    qué temas tienen sentimiento consistentemente positivo/negativo.
    '''
    df_valido = df[df['topic'] != -1].copy()

    # Columnas requeridas con valores por defecto si no existen
    if 'sentimiento_numerico' not in df_valido.columns:
        df_valido['sentimiento_numerico'] = np.nan
    if 'sentimiento_estrella' not in df_valido.columns:
        df_valido['sentimiento_estrella'] = 'sin_etiqueta'
    if 'estrellas' not in df_valido.columns:
        df_valido['estrellas'] = np.nan
    if 'intensidad_adjetivo' not in df_valido.columns:
        df_valido['intensidad_adjetivo'] = 0.0
    if 'intensidad_adverbio' not in df_valido.columns:
        df_valido['intensidad_adverbio'] = 0.0

    # Conteo por categoría de sentimiento
    conteo = (
        df_valido
        .groupby(['topic', 'sentimiento_estrella'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ['negativo', 'neutro', 'positivo', 'sin_etiqueta']:
        if col not in conteo.columns:
            conteo[col] = 0

    # Métricas numéricas por tópico
    metricas = (
        df_valido
        .groupby('topic')
        .agg(
            n_total=('indice', 'count'),
            sentimiento_medio=('sentimiento_numerico', 'mean'),
            sentimiento_std=('sentimiento_numerico', 'std'),
            estrella_media=('estrellas', 'mean'),
            intensidad_adj_media=('intensidad_adjetivo', 'mean'),
            intensidad_adv_media=('intensidad_adverbio', 'mean'),
        )
        .round(4)
        .reset_index()
    )

    tabla = conteo.merge(metricas, on='topic', how='left')

    # Ratio de polarización: qué tan dividido está el sentimiento
    # Alta polarización = muchos negativos Y muchos positivos
    n_con_rating = tabla['negativo'] + tabla['neutro'] + tabla['positivo']
    tabla['ratio_negativo']     = (tabla['negativo'] / n_con_rating.replace(0, np.nan) * 100).round(2)
    tabla['ratio_positivo']     = (tabla['positivo'] / n_con_rating.replace(0, np.nan) * 100).round(2)
    tabla['n_con_rating']       = n_con_rating
    tabla['indice_polarizacion'] = (
        (tabla['negativo'] * tabla['positivo']) /
        (n_con_rating ** 2).replace(0, np.nan)
    ).round(4)  # máximo cuando negativo = positivo = 50%

    tabla = tabla.sort_values('sentimiento_medio', ascending=True).reset_index(drop=True)
    logger.info('Perfil de sentimiento por tópico calculado: %d tópicos', len(tabla))
    return tabla


def calcular_resumen_microtopicos(
    path_microtopics: Path,
) -> pd.DataFrame:
    '''
    Resume los microtópicos producidos por pipe_microtopics() de Adrian.
    Calcula la distribución de microtópicos por región y tópico padre,
    identificando los grupos más grandes.

    Retorna DataFrame con columnas:
        region, parent_topic, microtopic, n_docs, pct_del_parent
    '''
    if not path_microtopics.exists():
        logger.warning('microtopics.csv no encontrado en %s', path_microtopics)
        return pd.DataFrame()

    df = pd.read_csv(path_microtopics)
    logger.info('Microtópicos cargados: %d documentos', len(df))

    tabla = (
        df
        .groupby(['region', 'parent_topic', 'microtopic'])
        .size()
        .reset_index(name='n_docs')
    )

    total_por_parent = (
        df
        .groupby(['region', 'parent_topic'])
        .size()
        .reset_index(name='total_parent')
    )

    tabla = tabla.merge(total_por_parent, on=['region', 'parent_topic'], how='left')
    tabla['pct_del_parent'] = (tabla['n_docs'] / tabla['total_parent'] * 100).round(2)

    tabla = tabla.sort_values(
        ['region', 'parent_topic', 'n_docs'],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    logger.info('Resumen de microtópicos: %d combinaciones', len(tabla))
    return tabla


def calcular_perfil_destino(
    df: pd.DataFrame,
    df_distribucion_topicos: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Construye un perfil comparativo de cada destino con indicadores clave:
        - total de documentos
        - % de documentos con rating
        - estrella media y sentimiento medio
        - tópico dominante (el más frecuente)
        - distribución de idioma (lang)
        - diversidad temática (número de tópicos con >= 1% del corpus del destino)
    '''
    # Métricas base
    metricas_base = (
        df
        .groupby('location')
        .agg(
            n_total=('indice', 'count'),
        )
        .reset_index()
    )

    # Métricas de sentimiento (pueden no estar disponibles)
    cols_sent = ['location', 'sentimiento_numerico', 'estrellas']
    cols_disponibles = [c for c in cols_sent if c in df.columns]

    if len(cols_disponibles) > 1:
        metricas_sent = (
            df[cols_disponibles]
            .groupby('location')
            .agg(
                sentimiento_medio=('sentimiento_numerico', 'mean') if 'sentimiento_numerico' in cols_disponibles else None,
                estrella_media=('estrellas', 'mean') if 'estrellas' in cols_disponibles else None,
                n_con_rating=('estrellas', lambda x: x.notna().sum()) if 'estrellas' in cols_disponibles else None,
            )
            .dropna(axis=1, how='all')
            .round(4)
            .reset_index()
        )
        metricas_base = metricas_base.merge(metricas_sent, on='location', how='left')

    # Distribución de idioma
    if 'lang' in df.columns:
        lang_dist = (
            df
            .groupby(['location', 'lang'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        metricas_base = metricas_base.merge(lang_dist, on='location', how='left')

    # Tópico dominante por destino
    if not df_distribucion_topicos.empty and 'pct_en_destino' in df_distribucion_topicos.columns:
        topico_dominante = (
            df_distribucion_topicos
            .sort_values('pct_en_destino', ascending=False)
            .groupby('location')
            .first()[['topic']]
            .rename(columns={'topic': 'topico_dominante'})
            .reset_index()
        )
        metricas_base = metricas_base.merge(topico_dominante, on='location', how='left')

        # Diversidad temática (tópicos con >= 1% de presencia en el destino)
        diversidad = (
            df_distribucion_topicos[df_distribucion_topicos['pct_en_destino'] >= 1.0]
            .groupby('location')
            .size()
            .rename('n_topicos_relevantes')
            .reset_index()
        )
        metricas_base = metricas_base.merge(diversidad, on='location', how='left')

    logger.info('Perfil de destinos construido: %d destinos', len(metricas_base))
    return metricas_base


def run_trend_detection() -> dict[str, pd.DataFrame]:
    '''
    Pipeline completo de detección de tendencias.

    Retorna diccionario con los DataFrames producidos:
        dist_topicos_destino, perfil_sentimiento_topico,
        microtopicos_resumen, perfil_destino
    '''
    Paths.TRENDS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info('=== Iniciando detección de tendencias ===')

    logger.info('Cargando corpus...')
    df_docs = pd.read_csv(Paths.DOCS_WITH_TOPICS_CSV)

    if Paths.CORPUS_CON_SENTIMIENTO_CSV.exists():
        df_sent = pd.read_csv(
            Paths.CORPUS_CON_SENTIMIENTO_CSV,
            usecols=['indice', 'sentimiento_numerico', 'sentimiento_estrella',
                     'estrellas', 'intensidad_adjetivo', 'intensidad_adverbio'],
        )
        df = df_docs.merge(df_sent, on='indice', how='left')
        logger.info('Corpus enriquecido con sentimiento')
    elif Paths.UNIFIED_ANALYSIS_CSV.exists():
        df_unified = pd.read_csv(Paths.UNIFIED_ANALYSIS_CSV, usecols=['indice', 'estrellas'])
        df = df_docs.merge(df_unified, on='indice', how='left')
        logger.info('Corpus enriquecido con estrellas (sin sentimiento calculado)')
    else:
        df = df_docs.copy()
        logger.warning('Sin datos de sentimiento disponibles')

    topics_meta = None
    if Paths.TOPICS_CSV.exists():
        topics_meta = pd.read_csv(Paths.TOPICS_CSV)

    df_dist_topicos = calcular_distribucion_topicos_destino(df, topics_meta)
    df_dist_topicos.to_csv(Paths.TENDENCIAS_TOPICOS_DESTINO_CSV, index=False, encoding='utf-8-sig')

    df_perfil_sent = calcular_perfil_sentimiento_topico(df)
    df_perfil_sent.to_csv(Paths.TENDENCIAS_SENTIMIENTO_TOPICO_CSV, index=False, encoding='utf-8-sig')

    df_micro = calcular_resumen_microtopicos(Paths.MICROTOPICS_CSV)
    if not df_micro.empty:
        df_micro.to_csv(Paths.MICROTOPICOS_RESUMEN_CSV, index=False, encoding='utf-8-sig')

    df_perfil_destino = calcular_perfil_destino(df, df_dist_topicos)
    df_perfil_destino.to_csv(Paths.PERFIL_DESTINO_CSV, index=False, encoding='utf-8-sig')

    logger.info('=== Detección de tendencias completada. Archivos en: %s ===', Paths.TRENDS_DIR)

    return {
        'dist_topicos_destino'       : df_dist_topicos,
        'perfil_sentimiento_topico'  : df_perfil_sent,
        'microtopicos_resumen'       : df_micro,
        'perfil_destino'             : df_perfil_destino,
    }


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    run_trend_detection()
