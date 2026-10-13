"""Sentence embedding model, preferring the fine-tuned Helix checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from helix.config import BASE_EMBED, EMBED_DIR

_CACHE: dict[str, "Embedder"] = {}


class Embedder:
    """Normalized sentence embeddings for retrieval."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        path = Path(model_path) if model_path else _default_path()
        self.path = str(path)
        self.model = SentenceTransformer(self.path)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def _default_path() -> Path | str:
    if (EMBED_DIR / "config.json").exists() or (EMBED_DIR / "modules.json").exists():
        return EMBED_DIR
    return BASE_EMBED


def get_embedder() -> Embedder:
    """Return the process-wide embedder."""
    path = str(_default_path())
    if path not in _CACHE:
        _CACHE[path] = Embedder(path)
    return _CACHE[path]
