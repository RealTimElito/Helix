"""Paths and model locations for Helix."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
DOCS_DIR = ROOT / "docs"
MODELS_DIR = ROOT / "models"

BASE_LLM = MODELS_DIR / "Qwen3-4B-Instruct-2507"
ADAPTER_DIR = MODELS_DIR / "helix-lora"
EMBED_DIR = MODELS_DIR / "helix-embed"
BASE_EMBED = "sentence-transformers/all-MiniLM-L6-v2"

DB_PATH = DATA_DIR / "helix.db"
EMBED_STORE = DATA_DIR / "embeddings"
SFT_DIR = DATA_DIR / "sft"
RETRIEVAL_DIR = DATA_DIR / "retrieval"
SAMPLE_DIR = DATA_DIR / "samples"

MAX_SFT_LENGTH = 1536
CITATION_RE = r"\[([A-Za-z0-9_-]+)\s+p\.(\d+)\]"


def ensure_dirs() -> None:
    """Create runtime directories if they are missing."""
    for path in (
        DATA_DIR,
        RUNS_DIR,
        DOCS_DIR,
        EMBED_STORE,
        SFT_DIR,
        RETRIEVAL_DIR,
        SAMPLE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
