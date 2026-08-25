"""Lazy local BGE-small embedding service."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from fastembed import TextEmbedding


def _cached_model_path() -> str | None:
    """Resolve fastembed's ONNX snapshot from the local Hugging Face cache.

    fastembed revalidates the model revision over the network on every
    process start; on networks that block huggingface.co that call is a
    connection reset even though the weights are fully cached. Passing the
    cached snapshot path skips every network touch."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download("qdrant/bge-small-en-v1.5-onnx-q",
                                 local_files_only=True)
    except Exception:
        return None


class LocalEmbedder:
    model_name = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            # A single CPU worker avoids an ONNX Runtime recursive-mutex crash
            # observed during interpreter shutdown on macOS/Python 3.13.
            self._model = TextEmbedding(
                model_name=self.model_name,
                threads=1,
                providers=["CPUExecutionProvider"],
                specific_model_path=_cached_model_path(),
            )
        return self._model

    def encode(self, texts: Iterable[str]) -> list[np.ndarray]:
        values = list(texts)
        if not values:
            return []
        return [np.asarray(vector, dtype=np.float32)
                for vector in self.model.embed(values)]

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
