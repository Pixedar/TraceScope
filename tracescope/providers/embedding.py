"""
Embedding providers – abstract interface + OpenAI implementation.

Switch embedding models by passing a different provider to the pipeline:

    from tracescope.providers.embedding import OpenAIEmbedding
    provider = OpenAIEmbedding(api_key="sk-...", model="text-embedding-3-large")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from openai import OpenAI


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier (used as key in vector store)."""
        ...

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns 1-D float array."""
        ...

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed a list of texts. Returns (N, D) array.

        Default implementation calls embed() in a loop.
        Subclasses should override for batch API support.
        """
        return np.array([self.embed(t) for t in texts])


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embeddings via the official SDK.

    Supports text-embedding-3-small, text-embedding-3-large,
    text-embedding-ada-002, etc.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self._model = model
        self._client = OpenAI(api_key=api_key)

    def model_name(self) -> str:
        return self._model

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Batch embed using OpenAI's batch API (up to 2048 inputs)."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                input=batch,
                model=self._model,
            )
            sorted_data = sorted(response.data, key=lambda d: d.index)
            all_embeddings.extend([d.embedding for d in sorted_data])
        return np.array(all_embeddings, dtype=np.float32)
