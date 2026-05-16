import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from config import Paths
from .ner import (
    read_json,
    clean_entities,
    aggregate_entities,
    merge_similar_entities,
    enrichment_text,
)
from .vectorization import (
    compute_tfidf,
    get_top_tfidf_words,
    extract_yake,
    build_yake_vocab,
)
from .BERTopic import BERTopic_analysis

logger = logging.getLogger(__name__)

_ANALYSIS_PATHS = [
    Paths.SPANISH_ANALYSIS_JSON,
    Paths.ENGLISH_ANALYSIS_JSON,
    Paths.MIXED_ANALYSIS_JSON,
]


def _build_features(texts: list[str]) -> dict:
    X_tfidf, vectorizer = compute_tfidf(texts)
    yake_keywords = extract_yake(texts)
    yake_vocab = build_yake_vocab(yake_keywords)
    vectorizer_yake = CountVectorizer(vocabulary=yake_vocab, binary=True)
    X_yake = vectorizer_yake.transform(texts)
    return {
        "X_tfidf": X_tfidf,
        "vectorizer": vectorizer,
        "X_yake": X_yake,
        "yake_vocab": yake_vocab,
        "vectorizer_yake": vectorizer_yake,
    }


def _extract_group_ner() -> list[dict]:
    all_data = []
    for path in _ANALYSIS_PATHS:
        if path.exists():
            all_data.extend(read_json(str(path)))
    cleaned = clean_entities(all_data)
    aggregate = aggregate_entities(cleaned)
    merged = merge_similar_entities(aggregate)
    return enrichment_text(merged, all_data)


def _build_doc_entity_map() -> dict[int, list[str]]:
    from preprocessing.individual_functions import normalize_ner
    doc_entities: dict[int, list[str]] = {}
    for path in _ANALYSIS_PATHS:
        if not path.exists():
            continue
        data = read_json(str(path))
        for row in data:
            idx = row["indice"]
            ents = [
                normalize_ner(e["text"])
                for e in row["entities"]
                if len(e["text"]) >= 3
            ]
            if idx not in doc_entities:
                doc_entities[idx] = []
            doc_entities[idx].extend(ents)
    return doc_entities


def _enrich_texts_with_ner(df: pd.DataFrame, doc_entities: dict[int, list[str]]) -> list[str]:
    enriched_texts = []
    for _, row in df.iterrows():
        idx = row["indice"]
        text = row["comentario_clean"]
        ents = doc_entities.get(idx, [])
        ent_tokens = " ".join([f"__ent_{e}__" for e in ents])
        enriched_texts.append(text + " " + ent_tokens)
    return enriched_texts


def pipe() -> None:
    Paths.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    Paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Extrayendo grupos NER...")
    analysis_ner = _extract_group_ner()

    df = pd.read_csv(Paths.NORMALIZED_SPANISH_CSV)
    df["comentario_clean"] = df["comentario_clean"].fillna("").astype(str)

    doc_entities = _build_doc_entity_map()
    texts = _enrich_texts_with_ner(df, doc_entities)

    logger.info("Construyendo features TF-IDF y YAKE...")
    features = _build_features(texts)

    with open(Paths.NER_GROUPS_JSON, "w", encoding="utf-8") as f:
        json.dump(analysis_ner, f, indent=4, ensure_ascii=False)

    top_words = get_top_tfidf_words(features["X_tfidf"], features["vectorizer"])
    with open(Paths.ENTITIES_TOP_WORDS_JSON, "w", encoding="utf-8") as f:
        json.dump(top_words, f, indent=4)

    logger.info("Matrices TF-IDF shape=%s | YAKE shape=%s", features["X_tfidf"].shape, features["X_yake"].shape)

    joblib.dump(features["vectorizer"],      Paths.TFIDF_PKL)
    joblib.dump(features["vectorizer_yake"], Paths.YAKE_PKL)

    logger.info("Extrayendo embeddings con BERTopic...")
    topic_bert = BERTopic_analysis(None, None, None, texts)
    embedding = topic_bert.embedding_extraction(None, None)
    np.save(Paths.DOCS_WITH_TOPICS_NPY, embedding)

    topics, probs = topic_bert.fit()
    topic_info = topic_bert.get_topics()

    df["topic"] = topics
    df.to_csv(Paths.DOCS_WITH_TOPICS_CSV, index=False)
    topic_info.to_csv(Paths.TOPICS_CSV, index=False)
    topic_bert.model.save(str(Paths.BERTOPIC_MODEL_DIR))

    logger.info("pipe() completado. Tópicos encontrados: %d", len(topic_info))


def pipe_microtopics() -> None:
    df = pd.read_csv(Paths.DOCS_WITH_TOPICS_CSV)
    df["comentario_clean"] = df["comentario_clean"].fillna("").astype(str)
    micro_results = []
    doc_entities = _build_doc_entity_map()

    for region in tqdm(df["location"].unique(), desc="Regions"):
        for topic in tqdm(df["topic"].unique(), desc=f"Topics {region}", leave=False):
            subset = df[(df["location"] == region) & (df["topic"] == topic)]
            if len(subset) < 30:
                continue
            texts = _enrich_texts_with_ner(subset, doc_entities)
            model = BERTopic_analysis(None, None, None, texts)
            topics_micro, _ = model.fit()
            subset = subset.copy()
            subset["microtopic"] = topics_micro
            subset["parent_topic"] = topic
            subset["region"] = region
            micro_results.append(subset)

    if micro_results:
        df_micro = pd.concat(micro_results, ignore_index=True)
        df_micro.to_csv(Paths.MICROTOPICS_CSV, index=False)
        logger.info("Microtópicos guardados: %s", Paths.MICROTOPICS_CSV)
