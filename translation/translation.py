import logging

import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

from config import Params, resolve_device

logger = logging.getLogger(__name__)


class Translator:
    def __init__(self, model_name: str | None = None):
        if model_name is None:
            model_name = Params.TRANSLATION_MODEL
        self.device = resolve_device(Params.DEVICE)
        logger.info("Cargando modelo de traducción: %s (device=%s)", model_name, self.device)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

    def translate_batch(self, texts: list[str], batch_size: int | None = None) -> list[str]:
        if batch_size is None:
            batch_size = Params.TRANSLATION_BATCH_SIZE
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Traduciendo"):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=Params.TRANSLATION_MAX_LENGTH,
            ).to(self.device)
            outputs = self.model.generate(**inputs, num_beams=Params.TRANSLATION_NUM_BEAMS)
            decoded = [self.tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
            results.extend(decoded)
        return results
