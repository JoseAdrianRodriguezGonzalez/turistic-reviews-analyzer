'''
features.py
-----------
Orquestador principal del bloque 3: Feature Engineering NLP.

Lee normalized_spanish.csv, ejecuta los cuatro modulos de features y exporta
un unico CSV consolidado (features_nlp.csv) listo para usar en
modelos predictivos o analisis estadistico.

Columnas del CSV de salida:
    -- Metadatos del documento (indice, edad, genero, lugar)
    -- Features de longitud  (text_features.py)
    -- Features de keywords  (keyword_features.py)
    -- Features POS          (pos_features.py)
    -- Features de entidades (entity_features.py)

Uso desde main.py:
    from feature_engineering.features import run_feature_pipeline
    run_feature_pipeline()

Uso directo:
    python features.py
'''

import json
import logging
from pathlib import Path

import pandas as pd

from config import Paths

logger = logging.getLogger(__name__)

DEMOGRAPHIC_COLUMNS = ['edad', 'genero', 'lugar']


def _load_all_analysis_csv(data_dir: Path) -> pd.DataFrame:
    paths = [
        data_dir / 'data_spanish' / 'analysis.csv',
        data_dir / 'data_english' / 'analysis.csv',
        data_dir / 'data_mixed'   / 'analysis.csv',
    ]
    dfs = [pd.read_csv(p) for p in paths if p.exists()]
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all.drop_duplicates(subset=['indice'])


def _load_all_analysis_json(data_dir: Path) -> list[dict]:
    paths = [
        data_dir / 'data_spanish' / 'analysis.json',
        data_dir / 'data_english' / 'analysis.json',
        data_dir / 'data_mixed'   / 'analysis.json',
    ]
    all_data: list[dict] = []
    for p in paths:
        if p.exists():
            with open(p, encoding='utf-8') as f:
                all_data.extend(json.load(f))

    seen: set = set()
    unique: list[dict] = []
    for row in all_data:
        idx = row['indice']
        if idx not in seen:
            unique.append(row)
            seen.add(idx)
    return unique


def run_feature_pipeline(
    data_clean_path: str | Path         = Paths.NORMALIZED_SPANISH_CSV,
    data_analysis_path: str | Path      = Paths.UNIFIED_ANALYSIS_CSV,
    vocab_unigrams_path: str | Path     = Paths.RANKINGS_UNIGRAMS_CSV,
    analysis_json_path: str | Path      = Paths.UNIFIED_ANALYSIS_JSON,
    output_path: str | Path             = Paths.FEATURES_NLP_CSV,
) -> pd.DataFrame:
    '''
    Ejecuta el pipeline completo de feature engineering y exporta el resultado.

    Parametros (todos opcionales, usan las rutas del proyecto por defecto):
        data_clean_path     -- ruta a normalized_spanish.csv desde preprocessing
        data_analysis_path  -- ruta a analysis_unified.csv para metadatos
        vocab_unigrams_path -- ruta a rankings_unigrams.csv (opcional)
        analysis_json_path  -- ruta al analysis_json_unified.json para entidades
        output_path         -- ruta de salida para features_nlp.csv

    Retorna el DataFrame con todas las features para uso inmediato.
    '''
    from feature_engineering.text_features    import compute_text_length_features
    from feature_engineering.keyword_features import load_vocabulary, compute_keyword_features
    from feature_engineering.pos_features     import load_spacy_model, compute_pos_features
    from feature_engineering.entity_features  import compute_entity_features

    logger.info('Iniciando pipeline de feature engineering')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Consolidar análisis de los tres idiomas y guardar unified
    Paths.UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    df_analysis = _load_all_analysis_csv(Paths.DATA_DIR)
    df_analysis.to_csv(Paths.UNIFIED_ANALYSIS_CSV, index=False)

    json_data = _load_all_analysis_json(Paths.DATA_DIR)
    with open(Paths.UNIFIED_ANALYSIS_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False)

    # Cargar corpus limpio y unir con metadatos
    df_clean = pd.read_csv(data_clean_path)
    df = df_clean.merge(df_analysis, on='indice', how='left')
    cleaned_series = df['comentario']
    corpus_list = cleaned_series.tolist()

    # Cargar modelo spaCy (compartido entre pos y entity features)
    nlp = load_spacy_model()

    logger.info('Calculando features de longitud de texto')
    text_feats = compute_text_length_features(cleaned_series)

    logger.info('Calculando features de keywords')
    vocab_path = Path(vocab_unigrams_path)
    if vocab_path.exists():
        vocabulary = load_vocabulary(vocab_unigrams_path)
        kw_feats = compute_keyword_features(
            corpus       = corpus_list,
            token_counts = text_feats['token_count'].to_numpy(),
            vocabulary   = vocabulary,
            ngram_n      = 1,
            index        = cleaned_series.index,
        )
    else:
        logger.warning('Vocabulario no encontrado: %s. Saltando keyword features.', vocab_unigrams_path)
        kw_feats = pd.DataFrame(index=cleaned_series.index)

    logger.info('Calculando features de distribucion POS')
    pos_feats = compute_pos_features(cleaned_series, nlp)

    logger.info('Calculando features de entidades nombradas')
    entity_feats = compute_entity_features(
        cleaned_series     = cleaned_series,
        nlp                = nlp,
        analysis_json_path = Paths.UNIFIED_ANALYSIS_JSON,
    )

    demographic_cols = [col for col in DEMOGRAPHIC_COLUMNS if col in df.columns]
    metadata_cols = (['indice'] + demographic_cols) if 'indice' in df.columns else demographic_cols
    metadata_df = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)

    features_df = pd.concat(
        [metadata_df, text_feats, kw_feats, pos_feats, entity_feats],
        axis=1,
    )

    features_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(
        'Features exportadas: %d documentos x %d columnas -> %s',
        len(features_df), len(features_df.columns), output_path,
    )

    return features_df


if __name__ == '__main__':
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    result = run_feature_pipeline()
    logger.info('Pipeline finalizado. Shape: %s', result.shape)
