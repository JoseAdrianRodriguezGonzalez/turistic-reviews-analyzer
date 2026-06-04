import logging

import pandas as pd

from config import Params
from .translation import Translator

logger = logging.getLogger(__name__)


def normalize_language(
    df: pd.DataFrame,
    text_col: str | None = None,
    lang_col: str | None = None,
) -> pd.DataFrame:
    if text_col is None:
        text_col = Params.COLUMNA_TEXTO
    if lang_col is None:
        lang_col = Params.COLUMNA_LANG
    translator = Translator()
    df = df.copy()
    mask_translate = df[lang_col].isin(Params.TRANSLATION_LANG_FILTER)
    texts_to_translate = df.loc[mask_translate, text_col].fillna("").astype(str).tolist()
    if texts_to_translate:
        logger.info("Traduciendo %d textos...", len(texts_to_translate))
        translated = translator.translate_batch(texts_to_translate)
        df.loc[mask_translate, text_col] = translated
    return df
