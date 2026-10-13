"""Build contrastive retrieval pairs and a distractor corpus."""

from __future__ import annotations

import json

from helix.config import RETRIEVAL_DIR, ensure_dirs
from helix.corpus.library import documents
from helix.textutil import jaccard

_PREFIXES = (
    "Which source supports this:",
    "Look for evidence that",
    "Find the write-up where",
    "Relevant passage for:",
)

_SYSTEMS = (
    "Kestrel Index",
    "Marrow Rank",
    "Plover Gate",
    "Quill Monitor",
    "Yarrow Fusion",
    "Nimbus Latch",
    "Orchard Queue",
    "Vellum Chunker",
)
_FIGURES = (
    "11 percent",
    "64 ms",
    "0.22",
    "3.4 GB",
    "40 minutes",
    "rank 4",
    "0.51 cutoff",
    "19 cases",
)


def distractor_chunks() -> list[dict]:
    """Near-miss passages that share a technical tone but no gold sentence."""
    chunks = []
    index = 0
    for system in _SYSTEMS:
        for figure in _FIGURES:
            text = (
                f"{system} reported an internal figure of {figure} during a bench "
                f"exercise that is not part of the Helix source library. The note "
                f"does not describe Northline, Bramble, Citation Latch, Retry Loom, "
                f"Scanfold, Weekglass, LoRA Ridge, Figure Drift, Pairwise Ledger, "
                f"Severity Ladder, Chunk Window, Topic Drift Watch, Schema Kiln, "
                f"Action Tether, Heading Probe, Judge Caliper, Vendor Window, "
                f"Incident Margin, Negative Mine, or Patch Desk."
            )
            chunks.append({
                "id": f"DIST-{index:03d}",
                "text": text,
                "cite_id": f"DIST-{index:03d}",
                "page": 1,
            })
            index += 1
    return chunks


def build_pairs() -> tuple[list[dict], list[dict], list[dict]]:
    """Return train pairs, eval queries, and the full retrieval corpus."""
    library = documents()
    corpus = []
    for doc in library:
        for chunk in doc.chunks():
            corpus.append({
                "id": f"{doc.doc_id}-p{chunk['page']}",
                "text": chunk["text"],
                "cite_id": doc.doc_id,
                "page": chunk["page"],
                "split": doc.split,
            })
    corpus.extend(distractor_chunks())
    train_pairs: list[dict] = []
    eval_queries: list[dict] = []
    pages = {item["id"]: item["text"] for item in corpus}
    for doc in library:
        for query in doc.queries:
            positive_id = None
            for chunk in doc.chunks():
                if query.span in chunk["text"]:
                    positive_id = f"{doc.doc_id}-p{chunk['page']}"
                    break
            if positive_id is None:
                raise ValueError(f"Missing span for {doc.doc_id}")
            overlap = jaccard(query.text, pages[positive_id])
            row = {
                "query": query.text,
                "positive_id": positive_id,
                "positive": pages[positive_id],
                "overlap": round(overlap, 3),
                "doc_id": doc.doc_id,
            }
            if doc.split == "train":
                train_pairs.append(row)
                for prefix in _PREFIXES:
                    train_pairs.append({
                        **row,
                        "query": f"{prefix} {query.text}",
                    })
            else:
                eval_queries.append(row)
    return train_pairs, eval_queries, corpus


def write_datasets() -> dict:
    """Write retrieval jsonl files and return a short summary."""
    ensure_dirs()
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
    train_pairs, eval_queries, corpus = build_pairs()
    _dump(RETRIEVAL_DIR / "train.jsonl", train_pairs)
    _dump(RETRIEVAL_DIR / "eval.jsonl", eval_queries)
    _dump(RETRIEVAL_DIR / "corpus.jsonl", corpus)
    summary = {
        "train_pairs": len(train_pairs),
        "eval_queries": len(eval_queries),
        "corpus": len(corpus),
        "mean_eval_overlap": round(
            sum(row["overlap"] for row in eval_queries) / max(1, len(eval_queries)),
            3,
        ),
    }
    (RETRIEVAL_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _dump(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    print(json.dumps(write_datasets(), indent=2))


if __name__ == "__main__":
    main()
