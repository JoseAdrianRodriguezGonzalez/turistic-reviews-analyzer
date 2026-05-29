import logging

import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Translator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-es"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Cargando modelo de traducción: %s (device=%s)", model_name, self.device)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

    def translate_batch(self, texts: list[str], batch_size: int = 32) -> list[str]:
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Traduciendo"):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            outputs = self.model.generate(**inputs)
            decoded = [self.tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
            results.extend(decoded)
        return results
