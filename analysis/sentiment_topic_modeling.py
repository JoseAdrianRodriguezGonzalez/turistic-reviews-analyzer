'''
sentiment_topic_modeling.py
---------------------------
Issue #5: Modelado de tópicos por sentimiento + análisis semántico
del concepto "precio / valor / costo".

Para cada grupo (positivo / negativo):
    - len >= Params.SENTIMENT_TOPIC_MIN_DOCS  →  BERTopic
    - len <  umbral                            →  lista de frecuencias de palabras

Por cada tópico BERTopic extrae:
    - Palabras clave (model.get_topic)
    - Comentario más cercano al centroide (similitud coseno)

Análisis precio/valor/costo:
    - Embedding sintético del concepto con el mismo SentenceTransformer
    - Similitud coseno contra todos los documentos del corpus
    - Top-N más similares
    - Datos de scatter (coordenadas UMAP + similitud) exportados a CSV

Salidas en data/analysis/sentiment_topics/:
    topicos_positivos.json
    topicos_negativos.json
    topicos_positivos_scatter.csv      (solo si se usó BERTopic)
    topicos_negativos_scatter.csv      (solo si se usó BERTopic)
    precio_valor_costo.json
    precio_valor_costo_scatter.csv
    resumen_sentiment_topics.csv
'''

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import Params, Paths

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Paths.ANALYSIS_DIR / 'sentiment_topics'


def _cargar_corpus() -> tuple[pd.DataFrame, np.ndarray]:
    '''
    Carga corpus_con_sentimiento.csv y los embeddings 384-dim.
    El índice del DataFrame se usa para indexar el array de embeddings.
    Si el corpus no incluye comentario_clean (columna omitida al exportar),
    la carga directamente desde normalized_spanish.csv por posición de fila.
    '''
    df = pd.read_csv(Paths.CORPUS_CON_SENTIMIENTO_CSV)

    if 'Unnamed: 0' in df.columns:
        df.index = df.pop('Unnamed: 0').astype(int)
    else:
        df.index = range(len(df))

    if 'comentario_clean' not in df.columns:
        logger.info('comentario_clean ausente en corpus — cargando desde normalized_spanish.csv')
        df_limpio = pd.read_csv(str(Paths.NORMALIZED_SPANISH_CSV), usecols=['comentario_clean'])
        df['comentario_clean'] = df_limpio['comentario_clean'].values[:len(df)]

    X = np.load(str(Paths.DOCS_WITH_TOPICS_NPY))

    if len(df) > len(X):
        logger.warning(
            'Corpus (%d filas) mayor que embeddings (%d) — truncando corpus',
            len(df), len(X),
        )
        df = df.iloc[:len(X)]

    return df, X


def _cargar_proyeccion_2d() -> np.ndarray | None:
    '''Carga proyección UMAP 2D del clustering activo si existe'''
    path = Paths.CLUSTERING_EMBEDDINGS_DIR / Paths.CLUSTERING_PROYECCION_FILE
    if path.exists():
        return np.load(str(path))
    logger.warning('Proyección 2D no encontrada en %s — scatter sin coordenadas', path)
    return None


def _frecuencia_fallback(textos: list[str], top_n: int = 30) -> list[dict]:
    '''Frecuencia de palabras para grupos por debajo del umbral de BERTopic'''
    counter: Counter = Counter()
    for texto in textos:
        for token in str(texto).split():
            if len(token) >= 3:
                counter[token] += 1
    return [{'palabra': p, 'frecuencia': int(f)} for p, f in counter.most_common(top_n)]


def _doc_mas_cercano_centroide(embs: np.ndarray, textos: list[str]) -> dict:
    '''Comentario con mayor similitud coseno al centroide del tópico'''
    centroide = embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(embs, centroide).ravel()
    best = int(np.argmax(sims))
    return {
        'texto': textos[best],
        'similitud_centroide': float(sims[best]),
    }


def _extraer_info_topicos(
    analisis,
    topics: list[int],
    textos_grupo: list[str],
) -> list[dict]:
    '''Extrae keywords y documento representativo por cada tópico BERTopic'''
    embs = analisis.embedded
    topics_arr = np.array(topics)
    ids_validos = sorted(t for t in set(topics) if t != -1)

    resultados = []
    for tid in ids_validos:
        mask = topics_arr == tid
        embs_t = embs[mask]
        txts_t = [textos_grupo[i] for i, m in enumerate(mask) if m]

        words = analisis.model.get_topic(tid) or []
        keywords = [w for w, _ in words[:10]]

        rep = _doc_mas_cercano_centroide(embs_t, txts_t) if txts_t else {}

        resultados.append({
            'id': int(tid),
            'keywords': keywords,
            'n_docs': int(mask.sum()),
            'doc_representativo': rep,
        })

    return resultados


def _exportar_scatter_topicos(
    nombre: str,
    df_grupo: pd.DataFrame,
    topics: list[int],
    proyeccion_2d: np.ndarray | None,
) -> None:
    '''Exporta CSV con coordenadas 2D y tópico asignado por BERTopic'''
    registros = []
    idx_lista = df_grupo.index.tolist()
    comentario_col = 'comentario' if 'comentario' in df_grupo.columns else 'comentario_clean'

    for local_i, (global_idx, topic_id) in enumerate(zip(idx_lista, topics)):
        fila = df_grupo.loc[global_idx]
        punto: dict = {
            'idx': int(global_idx),
            'topico_id': int(topic_id),
            'location': str(fila.get('location', '')),
            'sentimiento': nombre,
            'comentario': str(fila.get(comentario_col, ''))[:300],
        }
        if proyeccion_2d is not None and global_idx < len(proyeccion_2d):
            punto['x'] = float(proyeccion_2d[global_idx, 0])
            punto['y'] = float(proyeccion_2d[global_idx, 1])
        else:
            punto['x'] = None
            punto['y'] = None
        registros.append(punto)

    path = _OUTPUT_DIR / f'topicos_{nombre}_scatter.csv'
    pd.DataFrame(registros).to_csv(path, index=False, encoding='utf-8-sig')
    logger.info('Scatter tópicos %s exportado: %s', nombre, path)


def _procesar_grupo(
    nombre: str,
    df_grupo: pd.DataFrame,
    proyeccion_2d: np.ndarray | None,
) -> dict:
    '''
    Aplica BERTopic o fallback de frecuencias a un grupo de sentimiento.
    Exporta el scatter si se usó BERTopic.
    '''
    textos = df_grupo['comentario_clean'].fillna('').tolist()
    n = len(textos)

    logger.info('[%s] %d documentos', nombre, n)

    if n < Params.SENTIMENT_TOPIC_MIN_DOCS:
        logger.info(
            '[%s] Grupo pequeño (%d < %d) — usando frecuencia de palabras',
            nombre, n, Params.SENTIMENT_TOPIC_MIN_DOCS,
        )
        return {
            'sentimiento': nombre,
            'n_docs': n,
            'metodo': 'frecuencia',
            'frecuencias': _frecuencia_fallback(textos),
            'topicos': [],
        }

    logger.info('[%s] Ajustando BERTopic...', nombre)
    try:
        from semantic_expression.BERTopic import BERTopic_analysis

        analisis = BERTopic_analysis(
            unsupervised=None,
            reduction=None,
            embedding=Params.EMBEDDING_MODEL,
            docs=textos,
        )
        topics, _ = analisis.fit()

        topicos = _extraer_info_topicos(analisis, topics, textos)
        n_ruido = int(sum(1 for t in topics if t == -1))

        _exportar_scatter_topicos(nombre, df_grupo, topics, proyeccion_2d)

        return {
            'sentimiento': nombre,
            'n_docs': n,
            'n_ruido': n_ruido,
            'metodo': 'bertopic',
            'topicos': topicos,
        }

    except Exception as error:
        logger.error('[%s] BERTopic falló (%s) — fallback a frecuencias', nombre, error)
        return {
            'sentimiento': nombre,
            'n_docs': n,
            'metodo': 'frecuencia',
            'frecuencias': _frecuencia_fallback(textos),
            'topicos': [],
        }


def _analisis_precio_valor_costo(
    df: pd.DataFrame,
    X: np.ndarray,
    proyeccion_2d: np.ndarray | None,
    top_n: int,
) -> dict:
    '''
    Análisis semántico del concepto "precio / valor / costo".

    Genera un embedding del concepto con el mismo SentenceTransformer usado
    en el pipeline y calcula similitud coseno contra todos los documentos.
    Exporta scatter a CSV y devuelve el top-N en el JSON.
    '''
    from sentence_transformers import SentenceTransformer

    logger.info('[precio/valor/costo] Codificando concepto sintético...')
    modelo = SentenceTransformer(Params.EMBEDDING_MODEL)
    concepto_emb = modelo.encode(
        ['precio valor costo dinero pago tarifa caro barato económico'],
        show_progress_bar=False,
    )[0].reshape(1, -1)

    sims = cosine_similarity(X, concepto_emb).ravel()

    df_sim = df.copy()
    df_sim['similitud_precio'] = sims

    comentario_col = 'comentario' if 'comentario' in df_sim.columns else 'comentario_clean'

    # Top-N más relevantes
    top_indices = df_sim['similitud_precio'].nlargest(top_n).index.tolist()
    top_comentarios = []
    for idx in top_indices:
        fila = df_sim.loc[idx]
        top_comentarios.append({
            'idx': int(idx),
            'comentario': str(fila.get(comentario_col, '')),
            'comentario_clean': str(fila.get('comentario_clean', '')),
            'similitud': float(fila['similitud_precio']),
            'location': str(fila.get('location', '')),
            'sentimiento': str(fila.get('sentimiento_binario', '')),
        })

    # Scatter: todos los puntos — se exporta como CSV por volumen
    registros = []
    for i, (row_idx, fila) in enumerate(df_sim.iterrows()):
        punto: dict = {
            'idx': int(row_idx),
            'similitud_precio': float(fila['similitud_precio']),
            'location': str(fila.get('location', '')),
            'sentimiento': str(fila.get('sentimiento_binario', '')),
            'comentario': str(fila.get(comentario_col, ''))[:300],
        }
        if proyeccion_2d is not None and row_idx < len(proyeccion_2d):
            punto['x'] = float(proyeccion_2d[row_idx, 0])
            punto['y'] = float(proyeccion_2d[row_idx, 1])
        else:
            punto['x'] = None
            punto['y'] = None
        registros.append(punto)

    scatter_path = _OUTPUT_DIR / 'precio_valor_costo_scatter.csv'
    pd.DataFrame(registros).to_csv(scatter_path, index=False, encoding='utf-8-sig')
    logger.info('[precio/valor/costo] Scatter exportado: %s', scatter_path)

    sim_media_top = float(df_sim.loc[top_indices, 'similitud_precio'].mean())
    logger.info(
        '[precio/valor/costo] Top-%d similitud media: %.3f', top_n, sim_media_top
    )

    return {
        'concepto': 'precio / valor / costo',
        'top_n': top_n,
        'similitud_media_top': sim_media_top,
        'comentarios_mas_relevantes': top_comentarios,
    }


def _exportar_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info('Exportado: %s', path)


def run_sentiment_topic_modeling() -> dict:
    '''
    Función pública del módulo.

    Ejecuta modelado de tópicos sobre comentarios positivos y negativos,
    extrae palabras clave y documentos representativos por tópico, y
    realiza el análisis semántico de precio/valor/costo.

    Retorna diccionario con los resultados de ambos grupos y del análisis
    de precio.
    '''
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, X = _cargar_corpus()
    proyeccion_2d = _cargar_proyeccion_2d()

    sent_col = 'sentimiento_binario'
    if sent_col not in df.columns:
        logger.error('Columna %s no encontrada en corpus — abortando', sent_col)
        return {}

    mask_pos = df[sent_col] == 'positivo'
    mask_neg = df[sent_col] == 'negativo'

    logger.info(
        'Corpus: %d docs | positivos: %d | negativos: %d',
        len(df), int(mask_pos.sum()), int(mask_neg.sum()),
    )

    resultado_pos = _procesar_grupo('positivo', df[mask_pos], proyeccion_2d)
    resultado_neg = _procesar_grupo('negativo', df[mask_neg], proyeccion_2d)

    _exportar_json(resultado_pos, _OUTPUT_DIR / 'topicos_positivos.json')
    _exportar_json(resultado_neg, _OUTPUT_DIR / 'topicos_negativos.json')

    resultado_precio = _analisis_precio_valor_costo(
        df, X, proyeccion_2d, top_n=Params.PRECIO_VALOR_COSTO_TOP_N
    )
    _exportar_json(resultado_precio, _OUTPUT_DIR / 'precio_valor_costo.json')

    resumen = pd.DataFrame([
        {
            'grupo': 'positivo',
            'n_docs': resultado_pos['n_docs'],
            'metodo': resultado_pos['metodo'],
            'n_topicos': len(resultado_pos.get('topicos', [])),
        },
        {
            'grupo': 'negativo',
            'n_docs': resultado_neg['n_docs'],
            'metodo': resultado_neg['metodo'],
            'n_topicos': len(resultado_neg.get('topicos', [])),
        },
    ])
    resumen.to_csv(
        _OUTPUT_DIR / 'resumen_sentiment_topics.csv',
        index=False,
        encoding='utf-8-sig',
    )

    return {
        'positivo': resultado_pos,
        'negativo': resultado_neg,
        'precio_valor_costo': resultado_precio,
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    run_sentiment_topic_modeling()
