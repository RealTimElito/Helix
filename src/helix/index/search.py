"""Hybrid keyword and dense retrieval with reciprocal rank fusion."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from helix.index.store import get_chunks, load_embeddings
from helix.models.embedder import get_embedder


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Combine ranked id lists. Smaller rank index is better."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for index, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + index + 1)
    return scores


def diversify_by_document(chunks: list[dict], limit: int) -> list[dict]:
    """Prefer the best chunk of each document, up to `limit` documents.

    `chunks` is already ranked best-first. When more documents are present
    than `limit`, each selected document contributes its best chunk. When
    fewer documents matched, later chunks from those documents fill the
    remaining slots.
    """
    if limit <= 0:
        return []
    primary: list[dict] = []
    extra: list[dict] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = str(chunk.get("document_id") or chunk.get("cite_id") or chunk["id"])
        if key in seen:
            extra.append(chunk)
            continue
        seen.add(key)
        primary.append(chunk)
    if len(primary) >= limit:
        return primary[:limit]
    return (primary + extra)[:limit]


def search_collection(
    collection_id: str,
    query: str,
    limit: int = 6,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """Return fused chunks for a query, optionally restricted to documents."""
    chunks = get_chunks(collection_id, document_ids)
    if not chunks or not query.strip():
        return []
    by_id = {chunk["id"]: chunk for chunk in chunks}
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    keyword_scores = bm25.get_scores(query.lower().split())
    keyword_order = [
        chunks[index]["id"]
        for index in np.argsort(-np.asarray(keyword_scores))
        if keyword_scores[index] > 0
    ]
    dense_order: list[str] = []
    stored = load_embeddings(collection_id)
    if stored is not None:
        ids, vectors = stored
        id_to_row = {item_id: index for index, item_id in enumerate(ids)}
        rows = [id_to_row[chunk["id"]] for chunk in chunks if chunk["id"] in id_to_row]
        if rows:
            matrix = vectors[rows]
            query_vec = get_embedder().encode([query])[0]
            sims = matrix @ query_vec
            order = np.argsort(-sims)
            dense_order = [chunks[i]["id"] for i in order if sims[i] > 0.05]
    fused = reciprocal_rank_fusion([keyword_order, dense_order] if dense_order else [keyword_order])
    ranked = sorted(fused, key=fused.get, reverse=True)
    results = []
    for chunk_id in ranked:
        chunk = dict(by_id[chunk_id])
        chunk["cite"] = f"[{chunk['cite_id']} p.{chunk['page']}]"
        chunk["score"] = round(fused[chunk_id], 4)
        results.append(chunk)
    if document_ids:
        return results[:limit]
    return diversify_by_document(results, limit)


def load_collection_pages(collection_id: str) -> list[dict]:
    """Return every stored page, both pages of each note.

    Unfiltered briefs and extracts use this instead of a top-k search so
    every note in the collection can be quoted. Held-out notes stay out
    because they are not in the demo index.
    """
    ordered = sorted(
        get_chunks(collection_id),
        key=lambda chunk: (
            str(chunk.get("cite_id") or ""),
            _page_number(chunk),
            str(chunk.get("id") or ""),
        ),
    )
    return [_copied_chunk(chunk) for chunk in ordered]


def collection_note_count(collection_id: str) -> int:
    """Count distinct notes stored in a collection."""
    return len(_note_ids(get_chunks(collection_id)))


def _note_ids(chunks: list[dict]) -> list[str]:
    order: list[str] = []
    for chunk in chunks:
        key = _document_key(chunk)
        if key and key not in order:
            order.append(key)
    return order


def with_other_pages(collection_id: str, selected: list[dict]) -> list[dict]:
    """Add each document's other page after its best retrieved chunk.

    Unfiltered briefs otherwise keep only the page that matched the query.
    Page 1 is preferred when that page was not the hit. The result stays
    within eight documents and two chunks each. Document-filtered results
    should not be passed here.
    """
    if not selected:
        return []
    catalog = get_chunks(collection_id)
    if not catalog:
        return selected
    return attach_other_pages(selected, catalog)


def attach_other_pages(
    selected: list[dict],
    catalog: list[dict],
    max_docs: int = 8,
    max_per_doc: int = 2,
) -> list[dict]:
    """Keep chosen chunks and add another page from the same document."""
    grouped: dict[str, list[dict]] = {}
    for chunk in catalog:
        grouped.setdefault(_document_key(chunk), []).append(chunk)
    scored = {str(chunk["id"]): chunk for chunk in selected if chunk.get("id")}
    output: list[dict] = []
    seen: set[str] = set()
    per_doc: dict[str, int] = {}
    pages: dict[str, set[int]] = {}
    doc_order: list[str] = []
    for chunk in selected:
        key = _document_key(chunk)
        if key not in per_doc:
            if len(doc_order) >= max_docs:
                continue
            doc_order.append(key)
            per_doc[key] = 0
            pages[key] = set()
        if per_doc[key] >= max_per_doc:
            continue
        chunk_id = str(chunk.get("id") or "")
        if chunk_id and chunk_id in seen:
            continue
        output.append(_copied_chunk(chunk))
        if chunk_id:
            seen.add(chunk_id)
        per_doc[key] += 1
        pages[key].add(_page_number(chunk))
        if per_doc[key] >= max_per_doc:
            continue
        sibling = _other_page(grouped.get(key, []), seen, scored, pages[key])
        if sibling is None:
            continue
        output.append(sibling)
        sibling_id = str(sibling.get("id") or "")
        if sibling_id:
            seen.add(sibling_id)
        per_doc[key] += 1
        pages[key].add(_page_number(sibling))
    return output


def _document_key(chunk: dict) -> str:
    return str(chunk.get("document_id") or chunk.get("cite_id") or chunk["id"])


def _page_number(chunk: dict) -> int:
    try:
        return int(chunk.get("page") or 0)
    except (TypeError, ValueError):
        return 0


def _copied_chunk(chunk: dict) -> dict:
    item = dict(chunk)
    has_cite = item.get("cite")
    has_label = item.get("cite_id") is not None and item.get("page") is not None
    if not has_cite and has_label:
        item["cite"] = f"[{item['cite_id']} p.{item['page']}]"
    item.setdefault("score", 0.0)
    return item


def _other_page(
    candidates: list[dict],
    seen: set[str],
    scored: dict[str, dict],
    chosen_pages: set[int],
) -> dict | None:
    """Pick a page that is not already selected, preferring page 1."""
    pending: list[dict] = []
    for chunk in candidates:
        chunk_id = str(chunk.get("id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        pending.append(scored.get(chunk_id, chunk))
    if not pending:
        return None

    def sort_key(chunk: dict) -> tuple:
        page = _page_number(chunk)
        different_page = 0 if page not in chosen_pages else 1
        summary = 0 if page == 1 else 1
        score = float(chunk.get("score") or 0.0)
        return (different_page, summary, -score, page or 99)

    pending.sort(key=sort_key)
    return _copied_chunk(pending[0])
