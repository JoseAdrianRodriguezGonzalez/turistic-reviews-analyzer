import logging

import pandas as pd

from .translation import Translator

logger = logging.getLogger(__name__)


def normalize_language(
    df: pd.DataFrame,
    text_col: str = "comentario_clean",
    lang_col: str = "lang",
) -> pd.DataFrame:
    translator = Translator()
    df = df.copy()
    mask_translate = df[lang_col].isin(["en", "mix", "mixed"])
    texts_to_translate = df.loc[mask_translate, text_col].fillna("").astype(str).tolist()
    if texts_to_translate:
        logger.info("Traduciendo %d textos...", len(texts_to_translate))
        translated = translator.translate_batch(texts_to_translate)
        df.loc[mask_translate, text_col] = translated
    return df
