"""Citation checks that drop statements the cited page does not support."""

from __future__ import annotations

import re

from helix.config import CITATION_RE
from helix.textutil import supported

_CITE = re.compile(CITATION_RE)


def index_chunks(chunks: list[dict]) -> dict[tuple[str, int], str]:
    """Map (cite id, page) to combined text."""
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


def check_item(text: str, citation: str, sources: dict[tuple[str, int], str]) -> bool:
    parsed = parse_citation(citation)
    if parsed is None or parsed not in sources:
        return False
    return supported(text, sources[parsed])


def ground_payload(payload: dict, chunks: list[dict]) -> dict:
    """Return a copy of the payload with unsupported items removed."""
    sources = index_chunks(chunks)
    dropped: list[dict] = []
    cleaned = dict(payload)

    def keep_objects(items: list, text_key: str) -> list:
        kept = []
        for item in items or []:
            if not isinstance(item, dict):
                dropped.append({"item": item, "reason": "not an object"})
                continue
            text = str(item.get(text_key, ""))
            citation = str(item.get("citation", ""))
            action = str(item.get("action", "")) if "action" in item else ""
            ok = check_item(text, citation, sources)
            if ok and action and not check_item(action, citation, sources):
                ok = False
            if ok:
                kept.append(item)
            else:
                dropped.append({"item": item, "reason": "citation not supported"})
        return kept

    if "claims" in cleaned:
        cleaned["claims"] = keep_objects(cleaned.get("claims"), "text")
    if "limitations" in cleaned:
        cleaned["limitations"] = keep_objects(cleaned.get("limitations"), "text")
    if "risks" in cleaned:
        cleaned["risks"] = keep_objects(cleaned.get("risks"), "statement")
    if "aspects" in cleaned:
        aspects = []
        for aspect in cleaned.get("aspects") or []:
            if not isinstance(aspect, dict):
                continue
            observations = keep_objects(aspect.get("observations"), "text")
            aspects.append({**aspect, "observations": observations})
        cleaned["aspects"] = aspects
    if cleaned.get("task") == "qa":
        answer = str(cleaned.get("answer", ""))
        citation = str(cleaned.get("citation", ""))
        ok = check_item(answer, citation, sources)
        cleaned["supported"] = bool(ok)
        if not ok:
            dropped.append({"item": {"answer": answer, "citation": citation}, "reason": "qa unsupported"})
            cleaned["answer"] = ""
    cited_fields = ("summary", "decision")
    for field in cited_fields:
        value = cleaned.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        if not _prose_supported(value, sources):
            dropped.append({"item": {field: value}, "reason": "prose not in sources"})
            cleaned[field] = ""
    questions = []
    for question in cleaned.get("open_questions") or []:
        if _prose_supported(str(question), sources):
            questions.append(question)
        else:
            dropped.append({"item": question, "reason": "question not in sources"})
    if "open_questions" in cleaned:
        cleaned["open_questions"] = questions
    if cleaned.get("notes"):
        cleaned["notes"] = _sync_notes(cleaned)
    total = _count_checkable(payload)
    kept = total - len(dropped)
    score = (kept / total) if total else 1.0
    return {
        "payload": cleaned,
        "dropped": dropped,
        "faithfulness": round(score, 3),
    }


def _sync_notes(cleaned: dict) -> list[dict]:
    """Keep each note block aligned with the lines the citation check kept."""
    kept_claims = {
        id(item) for item in cleaned.get("claims") or [] if isinstance(item, dict)
    }
    kept_limits = {
        id(item) for item in cleaned.get("limitations") or [] if isinstance(item, dict)
    }
    kept_questions = list(cleaned.get("open_questions") or [])
    synced = []
    for note in cleaned.get("notes") or []:
        if not isinstance(note, dict):
            continue
        block = {
            "title": note.get("title") or "",
            "cite_id": note.get("cite_id") or "",
            "claims": [
                item for item in note.get("claims") or [] if id(item) in kept_claims
            ],
            "limitations": [
                item for item in note.get("limitations") or [] if id(item) in kept_limits
            ],
            "open_questions": [
                question
                for question in note.get("open_questions") or []
                if question in kept_questions
            ],
        }
        synced.append(block)
    return synced


def _prose_supported(text: str, sources: dict[tuple[str, int], str]) -> bool:
    body = _CITE.sub("", text).strip()
    if not body:
        return True
    sentences = [
        part.strip() for part in re.split(r"(?<=[.])\s+", body) if part.strip()
    ]
    return all(
        any(supported(sentence, source, overlap=0.55) for source in sources.values())
        for sentence in sentences
    )


def _count_checkable(payload: dict) -> int:
    count = 0
    for key in ("claims", "limitations", "risks", "open_questions"):
        count += len(payload.get(key) or [])
    for aspect in payload.get("aspects") or []:
        if isinstance(aspect, dict):
            count += len(aspect.get("observations") or [])
    if payload.get("task") == "qa":
        count += 1
    for field in ("summary", "decision"):
        if payload.get(field):
            count += 1
    return count
