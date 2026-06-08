'''
sentiment_analysis.py

Bloque 8 — Análisis de sentimiento y emoción a partir de:
    1. Estrellas (rating) como proxy de sentimiento supervisado
    2. Distribución léxica POS como proxy de intensidad emocional
    3. Polaridad por tópico (BERTopic) y por destino

Lógica de sentimiento basada en estrellas:
    1-2  -> negativo
    3    -> neutro
    4-5  -> positivo
    null -> sin_etiqueta (para documentos sin rating, ej. Instagram)

Columnas de salida principales:
    sentimiento_estrella      -- negativo / neutro / positivo / sin_etiqueta  (basado en estrellas)
    sentimiento_numerico      -- -1 / 0 / 1 / NaN
    sentimiento_binario_estrellas -- clasificación binaria por umbral de estrellas (validación)
    etiqueta_transformer      -- positivo / negativo / neutro  (modelo transformer, todos los docs)
    confianza_transformer     -- score del transformer [0, 1]
    etiqueta_robusto          -- predicción del clasificador supervisado (solo método robusto)
    sentimiento_binario       -- clasificación definitiva del transformer (neutro -> negativo)
    intensidad_adjetivo       -- ratio de adjetivos en el documento (POS)
    intensidad_adverbio       -- ratio de adverbios en el documento (POS)

Agrupaciones exportadas:
    comentarios_positivos.csv        -- documentos con sentimiento_binario == positivo
    comentarios_negativos.csv        -- documentos con sentimiento_binario == negativo
    sentimiento_por_topico.csv       -- distribución por tópico BERTopic
    sentimiento_por_destino.csv      -- distribución por location
    sentimiento_por_topico_destino.csv -- cruce tópico x destino

Uso:
    from analysis.sentiment_analysis import run_sentiment_analysis
    run_sentiment_analysis()
'''

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import Params, Paths

logger = logging.getLogger(__name__)


# CATEGORIZACIÓN DE SENTIMIENTO

def _mapear_sentimiento_estrella(estrellas: pd.Series) -> pd.Series:
    '''
    Convierte la columna de estrellas a etiqueta de sentimiento categórico.

    Escala usada:
        1-2  -> negativo
        3    -> neutro
        4-5  -> positivo
        NaN  -> sin_etiqueta
    '''
    def _categorizar(valor):
        if pd.isna(valor):
            return 'sin_etiqueta'
        if valor <= 2:
            return 'negativo'
        if valor == 3:
            return 'neutro'
        return 'positivo'

    return estrellas.apply(_categorizar)


def _mapear_sentimiento_numerico(categoria: pd.Series) -> pd.Series:
    '''
    Convierte la categoría textual a valor numérico [-1, 0, 1, NaN].
    Útil para promedios ponderados.
    '''
    mapa = {
        'negativo'    : -1.0,
        'neutro'      :  0.0,
        'positivo'    :  1.0,
        'sin_etiqueta':  np.nan,
    }
    return categoria.map(mapa)


def _mapear_sentimiento_binario(sentimiento_estrella: pd.Series) -> pd.Series:
    '''
    Colapsa la clasificación de 3 clases a binaria usando umbral de polaridad.
    Neutro se incluye en negativo porque 3 estrellas no es una recomendación.
        positivo  -> positivo  (4-5 estrellas)
        neutro    -> negativo  (3 estrellas, umbral no alcanzado)
        negativo  -> negativo  (1-2 estrellas)
        sin_etiqueta -> sin_etiqueta
    '''
    def _binarizar(cat):
        if cat in ('negativo', 'neutro'):
            return 'negativo'
        if cat == 'positivo':
            return 'positivo'
        return 'sin_etiqueta'

    return sentimiento_estrella.apply(_binarizar)


# CARGA Y ENSAMBLADO

def _cargar_corpus_base() -> pd.DataFrame:
    '''
    Carga y ensambla el corpus base unificando:
        - docs_with_topics.csv  (topic, location, lang)
        - analysis_unified.csv  (estrellas, texto original)
        - normalized_spanish.csv (texto limpio para vectorizadores)
        - features_nlp.csv      (pos_ratio_adj, pos_ratio_adv)

    El join se hace por indice. Los campos faltantes se rellenan con NaN.
    '''
    logger.info('Cargando docs_with_topics.csv...')
    df_docs = pd.read_csv(Paths.DOCS_WITH_TOPICS_CSV)

    logger.info('Cargando analysis_unified.csv...')
    df_unified_all = pd.read_csv(Paths.UNIFIED_ANALYSIS_CSV)
    cols_unified = ['indice']
    col_comentario = Params.COLUMNA_COMENTARIO
    if col_comentario in df_unified_all.columns:
        cols_unified.append(col_comentario)
    df_unified = df_unified_all[cols_unified]

    logger.info('Cargando normalized_spanish.csv...')
    df = pd.read_csv(Paths.NORMALES_CSV, usecols=['indice', Params.COLUMNA_TEXTO])

    logger.info('Cargando features_nlp.csv...')
    df_features = pd.read_csv(
        Paths.FEATURES_NLP_CSV,
        usecols=['indice', 'pos_ratio_adj', 'pos_ratio_adv'],
    )

    df = df_docs.merge(df_unified, on='indice', how='left')
    #df = df.merge(df_normalized, on='indice', how='left')
    df = df.merge(df_features, on='indice', how='left')
#    df=df.merge(df_docs,on='indices',how='left')
    logger.info(
        'Corpus ensamblado: %d documentos',
        len(df),
    )
    return df


def _construir_sentimiento(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Añade columnas de sentimiento basadas en estrellas al corpus.
    sentimiento_binario_estrellas se guarda como referencia de validación;
    el sentimiento_binario definitivo lo asigna el transformer en run_sentiment_analysis().
    '''
    df = df.copy()
#    df['sentimiento_estrella']          = _mapear_sentimiento_estrella(df['estrellas'])
#    df['sentimiento_numerico']          = _mapear_sentimiento_numerico(df['sentimiento_estrella'])
#    df['sentimiento_binario_estrellas'] = _mapear_sentimiento_binario(df['sentimiento_estrella'])
    df['intensidad_adjetivo']           = df['pos_ratio_adj'].fillna(0.0)
    df['intensidad_adverbio']           = df['pos_ratio_adv'].fillna(0.0)
    return df


# AGRUPACIONES

def _sentimiento_por_topico(df: pd.DataFrame, topics_meta: pd.DataFrame) -> pd.DataFrame:
    '''
    Distribución de sentimiento por tópico BERTopic.

    Excluye documentos sin rating (sin_etiqueta) del cálculo de porcentajes
    para no distorsionar la distribución supervisada, pero reporta
    n_sin_etiqueta como columna informativa adicional.
    '''
    # Solo con rating
   # df_con_rating = df[df['sentimiento_estrella'] != 'sin_etiqueta'].copy()

    tabla = (
        df
        .groupby(['topic','etiqueta_final'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Asegurarse de que existan las tres columnas de sentimiento
    for col in ['negativo', 'neutro', 'positivo']:
        if col not in tabla.columns:
            tabla[col] = 0

    tabla['total'] = tabla[['negativo', 'neutro', 'positivo']].sum(axis=1)

    # Porcentajes
    for col in ['negativo', 'neutro', 'positivo']:
        tabla[f'pct_{col}'] = (
            tabla[col] / tabla['total_con_rating'].replace(0, np.nan) * 100
        ).round(2)

    # Añadir nombre del tópico desde topics.csv
    if topics_meta is not None:
        tabla = tabla.merge(
            topics_meta[['Topic', 'Name']].rename(columns={'Topic': 'topic', 'Name': 'topic_name'}),
            on='topic',
            how='left',
        )

    tabla = tabla.sort_values('total', ascending=True).reset_index(drop=True)
    return tabla


def _sentimiento_por_destino(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Distribución de sentimiento por destino (location).
    Incluye todos los documentos con rating y reporta la media.
    '''
    
    tabla = (
        df
        .groupby(['location', 'etiqueta_final'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ['negativo', 'neutro', 'positivo']:
        if col not in tabla.columns:
            tabla[col] = 0

    tabla['total'] = tabla[['negativo', 'neutro', 'positivo']].sum(axis=1)

    for col in ['negativo', 'neutro', 'positivo']:
        tabla[f'pct_{col}'] = (
            tabla[col] / tabla['total'].replace(0, np.nan) * 100
        ).round(2)

    return tabla.sort_values('total', ascending=False).reset_index(drop=True)

def _sentimiento_por_topico_destino(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Cruce tópico x destino: promedio de sentimiento numérico y
    estrellas medias. Solo documentos con rating.
    Solo se incluyen tópicos válidos (topic != -1).
    '''
    df_valido = df[df['topic'] != -1].copy()

    tabla = (
        df_valido
        .groupby(['topic', 'location', 'etiqueta_final'])
        .size()
        .rename('n_docs')
        .reset_index()
    )

    return tabla.sort_values(['topic', 'n_docs'], ascending=[True, False])

def _exportar_grupos_binarios(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Filtra el corpus en dos grupos según sentimiento_binario y los exporta.
    Solo incluye documentos clasificables (excluye sin_etiqueta).
    '''
    df_positivos = df[df['sentimiento_binario'] == 'positivo'].reset_index(drop=True)
    df_negativos = df[df['sentimiento_binario'] == 'negativo'].reset_index(drop=True)

    df_positivos.to_csv(output_dir / 'comentarios_positivos.csv', index=False, encoding='utf-8-sig')
    df_negativos.to_csv(output_dir / 'comentarios_negativos.csv', index=False, encoding='utf-8-sig')

    logger.info(
        'Grupos binarios exportados: %d positivos | %d negativos | %d sin_etiqueta',
        len(df_positivos),
        len(df_negativos),
        (df['sentimiento_binario'] == 'sin_etiqueta').sum(),
    )
    return df_positivos, df_negativos


# PIPELINE PRINCIPAL

def _binarizar_etiqueta_transformer(etiqueta: str) -> str:
    '''
    Colapsa neutro en negativo para producir la clasificación binaria final.
    Neutro significa ambigüedad, no recomendación positiva.
    '''
    if etiqueta == 'positivo':
        return 'positivo'
    if etiqueta in ('negativo', 'neutro'):
        return 'negativo'
    return 'sin_etiqueta'


def run_sentiment_analysis() -> dict[str, pd.DataFrame]:
    output_dir = Paths.SENTIMENT_DIR
    '''
    Pipeline completo de análisis de sentimiento.

    Clasifica cada documento con un transformer (independiente de estrellas)
    y opcionalmente refina con un clasificador supervisado sobre embeddings.
    La configuración del método se lee desde Params (SENTIMENT_METODO,
    SENTIMENT_IDIOMA, SENTIMENT_TEXTO, SENTIMENT_FEATURES).

    Retorna diccionario con:
        corpus, positivos, negativos, por_topico, por_destino, por_topico_destino
    '''
    from analysis.transformer_sentiment import run_transformer_sentiment

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('=== Iniciando análisis de sentimiento ===')
    
    topics_meta = None
    if Paths.TOPICS_CSV.exists():
        topics_meta = pd.read_csv(Paths.TOPICS_CSV)
        logger.info('Metadatos de tópicos cargados: %d tópicos', len(topics_meta))
    
    df = _cargar_corpus_base()
    print("si")
    df = _construir_sentimiento(df)

  #  logger.info(
  #      'Distribución por estrellas (referencia): %s',
  #      df['sentimiento_estrella'].value_counts().to_dict(),
  #  )

    # Clasificación con transformer — funciona para TODOS los documentos
    col_original = Params.COLUMNA_COMENTARIO
    col_cleaned  = Params.COLUMNA_TEXTO

    textos_orig    = df.get(col_original, pd.Series([''] * len(df), index=df.index))
    textos_cleaned = df.get(col_cleaned,  pd.Series([''] * len(df), index=df.index))

    df_transformer = run_transformer_sentiment(
        textos_originales  = textos_orig,
        textos_cleaned     = textos_cleaned,
        idioma             = Params.SENTIMENT_IDIOMA,
        metodo             = Params.SENTIMENT_METODO,
        usar_texto_original= Params.SENTIMENT_TEXTO == 'original',
        features           = Params.SENTIMENT_FEATURES,
    )

    df = df.join(df_transformer)

    # sentimiento_binario definitivo viene del transformer (cubre Instagram y similares)
    df['sentimiento_binario'] = df['etiqueta_final'].apply(_binarizar_etiqueta_transformer)

    logger.info(
        'Distribución transformer (binario final): %s',
        df['sentimiento_binario'].value_counts().to_dict(),
    )

    path_corpus = Paths.CORPUS_CON_SENTIMIENTO_CSV
    df.to_csv(path_corpus, index=False, encoding='utf-8-sig')
    logger.info('Corpus con sentimiento exportado: %s', path_corpus)

    df_positivos, df_negativos = _exportar_grupos_binarios(df, output_dir)

    df_por_topico  = _sentimiento_por_topico(df, topics_meta)
    df_por_destino = _sentimiento_por_destino(df)
    df_cruce       = _sentimiento_por_topico_destino(df)

    df_por_topico.to_csv(Paths.SENTIMIENTO_POR_TOPICO_CSV, index=False, encoding='utf-8-sig')
    df_por_destino.to_csv(Paths.SENTIMIENTO_POR_DESTINO_CSV, index=False, encoding='utf-8-sig')
    df_cruce.to_csv(Paths.SENTIMIENTO_POR_TOPICO_DESTINO_CSV, index=False, encoding='utf-8-sig')

    logger.info('=== Análisis de sentimiento completado. Archivos en: %s ===', output_dir)

    return {
        'corpus'             : df,
        'positivos'          : df_positivos,
        'negativos'          : df_negativos,
        'por_topico'         : df_por_topico,
        'por_destino'        : df_por_destino,
        'por_topico_destino' : df_cruce,
    }


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    run_sentiment_analysis()
