"""Symptom extraction using a local BioBERT encoder and cosine similarity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class SymptomExtractor:
    """Extracts likely symptom mentions from free text using embedding similarity."""

    def __init__(
        self,
        symptom_library_path: str | Path,
        model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        similarity_threshold: float = 0.48,
    ) -> None:
        self.symptom_library_path = Path(symptom_library_path)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.symptoms = self._load_symptom_library()
        self.symptom_embeddings = self._encode_sentences(self.symptoms)

    def _load_symptom_library(self) -> List[str]:
        with self.symptom_library_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _encode_sentences(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def extract_symptoms(self, text: str) -> list[str]:
        if not text.strip():
            return []

        text_embedding = self._encode_sentences([text])
        similarities = torch.mm(text_embedding, self.symptom_embeddings.T).squeeze(0).cpu().numpy()

        detected = []
        for symptom, score in zip(self.symptoms, similarities):
            direct_mention = symptom.lower() in text.lower()
            if direct_mention or score >= self.similarity_threshold:
                detected.append((symptom, float(score)))

        detected.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in detected]

    def similarity_table(self, text: str) -> list[tuple[str, float]]:
        """Returns symptom similarity scores for explainability."""
        if not text.strip():
            return []

        text_embedding = self._encode_sentences([text])
        similarities = torch.mm(text_embedding, self.symptom_embeddings.T).squeeze(0).cpu().numpy()
        rows = list(zip(self.symptoms, similarities.tolist()))
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows
