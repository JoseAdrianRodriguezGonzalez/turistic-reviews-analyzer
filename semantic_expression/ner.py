import json
import logging
from collections import Counter, defaultdict

import pandas as pd
from rapidfuzz import fuzz

from preprocessing.individual_functions import normalize_ner

logger = logging.getLogger(__name__)


def read_json(src: str) -> list[dict]:
    with open(src, encoding='utf-8') as f:
        return json.load(f)


def clean_entities(data: list[dict]) -> list[dict]:
    cleaned = []
    valid = {"GPE", "LOC", "FAC", "ORG"}
    for row in data:
        index = row["indice"]
        for entity in row["entities"]:
            if entity["label"] not in valid:
                continue
            if len(entity["text"]) < 3:
                continue
            cleaned.append({
                "index": index,
                "text": normalize_ner(entity["text"]),
                "label": entity["label"],
            })
    return cleaned


def aggregate_entities(cleaned: list[dict]) -> list[dict]:
    entity_map: dict = defaultdict(lambda: {"count": 0, "indices": set()})
    for item in cleaned:
        key = (item["text"], item["label"])
        entity_map[key]["count"] += 1
        entity_map[key]["indices"].add(item["index"])
    result = [
        {"text": text, "label": label, "count": val["count"], "indices": list(val["indices"])}
        for (text, label), val in entity_map.items()
    ]
    result.sort(key=lambda x: x["count"], reverse=True)
    return result


def merge_similar_entities(entities: list[dict], threshold: int = 90) -> list[dict]:
    merged = []
    used = [False] * len(entities)
    for i, base in enumerate(entities):
        if used[i]:
            continue
        new_entity = {
            "text": base["text"],
            "label": base["label"],
            "count": base["count"],
            "indices": set(base["indices"]),
        }
        for j in range(i + 1, len(entities)):
            if used[j]:
                continue
            if fuzz.ratio(base["text"], entities[j]["text"]) >= threshold:
                new_entity["count"] += entities[j]["count"]
                new_entity["indices"].update(entities[j]["indices"])
                used[j] = True
        used[i] = True
        new_entity["indices"] = list(new_entity["indices"])
        merged.append(new_entity)
    return merged


def enrichment_text(groups: list[dict], original: list[dict], top_k: int = 5) -> list[dict]:
    original_map = {row["indice"]: row for row in original}
    for i, base in enumerate(groups):
        phrases = []
        for index in base["indices"]:
            if index not in original_map:
                continue
            phrases.extend(original_map[index]["noun_phrases"])
        counter = Counter(phrases)
        groups[i]["top_noun_phrases"] = [p for p, _ in counter.most_common(top_k)]
        groups[i]["noun_phrases_freq"] = dict(counter)
    return groups
