"""Small text helpers shared by retrieval, training, and grounding checks."""

from __future__ import annotations

import json
import re

_STOP = frozenset(
    """
    a an the of to for and or in on with by from that this which where after
    before their its is was were be as at into over under than then not no
    when what how who did does do are were been being it they them these those
    about than versus vs
    """.split()
)


def content_tokens(text: str) -> set[str]:
    """Lowercase tokens with stopwords and short fragments removed."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in _STOP and len(word) > 2}


def jaccard(left: str, right: str) -> float:
    """Jaccard overlap of content tokens."""
    a = content_tokens(left)
    b = content_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def supported(text: str, source: str, overlap: float = 0.6) -> bool:
    """Whether a statement is contained in or lexically covered by a source."""
    if not text or not source:
        return False
    if " ".join(text.split()).lower() in " ".join(source.split()).lower():
        return True
    tokens = content_tokens(text)
    if not tokens:
        return False
    covered = tokens & content_tokens(source)
    return (len(covered) / len(tokens)) >= overlap


def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    raw = text.strip()
    if "```" in raw:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object in model output")
    return json.loads(raw[start : end + 1])


def normalize_sentence(text: str) -> str:
    """Normalize a sentence for set comparisons."""
    return re.sub(r"\s+", " ", text.strip().lower())
