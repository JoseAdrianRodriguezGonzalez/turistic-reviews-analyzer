import logging
from collections import Counter

import numpy as np
import yake
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def compute_tfidf(texts: list[str]) -> tuple:
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2), max_features=None)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def get_top_tfidf_words(X, vectorizer, top_k: int = 10) -> list[dict]:
    sums = X.sum(axis=0)
    words = vectorizer.get_feature_names_out()
    ranking = [
        {"word": words[i], "score": float(sums[0, i])}
        for i in range(len(words))
    ]
    return sorted(ranking, key=lambda x: x["score"], reverse=True)[:top_k]


def extract_yake(texts: list[str], top_k: int = 5) -> list[list[str]]:
    kw_extractor = yake.KeywordExtractor(lan="es", n=2, top=top_k)
    return [[kw for kw, score in kw_extractor.extract_keywords(t)] for t in texts]


def yake_to_vector(yake_list: list[str], vocab: list[str]) -> list[int]:
    counter = Counter(yake_list)
    return [counter.get(word, 0) for word in vocab]


def build_yake_vocab(all_keywords: list[list[str]], min_freq: int = 2) -> list[str]:
    flat = [kw for doc in all_keywords for kw in doc]
    counter = Counter(flat)
    return [k for k, v in counter.items() if v >= min_freq]
