"""Citation checks that drop unsupported structured fields."""

from __future__ import annotations

import re

from helix.config import CITATION_RE
from helix.textutil import supported

_CITE = re.compile(CITATION_RE)


def index_chunks(chunks: list[dict]) -> dict[tuple[str, int], str]:
    """Map (cite id, page) to combined source text."""
    grouped: dict[tuple[str, int], list[str]] = {}
    for chunk in chunks:
        key = (str(chunk["cite_id"]), int(chunk["page"]))
        grouped.setdefault(key, []).append(chunk["text"])
    return {key: "\n".join(parts) for key, parts in grouped.items()}


def parse_citation(citation: str) -> tuple[str, int] | None:
    match = _CITE.search(citation or "")
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _keep_text(item: dict, key: str, sources: dict[tuple[str, int], str]) -> bool:
    parsed = parse_citation(str(item.get("citation", "")))
    if parsed is None or parsed not in sources:
        return False
    return supported(str(item.get(key, "")), sources[parsed])


def ground_payload(payload: dict, chunks: list[dict]) -> dict:
    """Return a copy of the payload with unsupported items removed."""
    sources = index_chunks(chunks)
    cleaned = dict(payload)
    dropped: list[dict] = []
    for field, text_key in (
        ("claims", "text"),
        ("limitations", "text"),
        ("risks", "statement"),
    ):
        kept = []
        for item in payload.get(field) or []:
            if isinstance(item, dict) and _keep_text(item, text_key, sources):
                if field == "risks":
                    action = str(item.get("action", ""))
                    parsed = parse_citation(str(item.get("citation", "")))
                    if action and parsed and not supported(action, sources[parsed]):
                        item = {**item, "action": ""}
                        dropped.append({"field": "risks.action", "text": action})
                kept.append(item)
            else:
                dropped.append({"field": field, "item": item})
        if field in payload:
            cleaned[field] = kept
    if "aspects" in payload:
        aspects = []
        for aspect in payload.get("aspects") or []:
            observations = []
            for item in aspect.get("observations") or []:
                if isinstance(item, dict) and _keep_text(item, "text", sources):
                    observations.append(item)
                else:
                    dropped.append({"field": "aspects", "item": item})
            aspects.append({**aspect, "observations": observations})
        cleaned["aspects"] = aspects
    if payload.get("task") == "qa":
        parsed = parse_citation(str(payload.get("citation", "")))
        answer = str(payload.get("answer", ""))
        ok = parsed is not None and parsed in sources and supported(answer, sources[parsed])
        cleaned["supported"] = bool(ok)
        if not ok:
            dropped.append({"field": "qa", "text": answer})
            cleaned["answer"] = ""
    summary = str(payload.get("summary") or payload.get("decision") or "")
    if summary and not any(supported(summary, text) for text in sources.values()):
        # Multi-sentence decisions are checked piece by piece.
        pieces = [part.strip() for part in re.split(r"(?<=[.])\s+", summary) if part.strip()]
        kept_pieces = [
            piece for piece in pieces if any(supported(piece, text) for text in sources.values())
        ]
        if "summary" in payload:
            cleaned["summary"] = " ".join(kept_pieces)
        if "decision" in payload and payload.get("decision") == summary:
            cleaned["decision"] = " ".join(kept_pieces)
        if len(kept_pieces) != len(pieces):
            dropped.append({"field": "summary", "text": summary})
    cleaned["grounding"] = {
        "dropped": dropped,
        "dropped_count": len(dropped),
        "faithfulness": _faithfulness(cleaned, dropped),
    }
    return cleaned


def _faithfulness(cleaned: dict, dropped: list[dict]) -> float:
    kept = 0
    for field in ("claims", "limitations", "risks"):
        kept += len(cleaned.get(field) or [])
    for aspect in cleaned.get("aspects") or []:
        kept += len(aspect.get("observations") or [])
    if cleaned.get("task") == "qa" and cleaned.get("supported"):
        kept += 1
    total = kept + len(dropped)
    if total == 0:
        return 1.0
    return round(kept / total, 3)
