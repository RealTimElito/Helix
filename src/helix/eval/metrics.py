"""Retrieval and structured-answer metrics."""

from __future__ import annotations

import math

import numpy as np

from helix.textutil import normalize_sentence


def recall_ndcg(query_vectors: np.ndarray, corpus_vectors: np.ndarray, relevant: list[str], corpus_ids: list[str], ks: tuple[int, ...] = (3, 5)) -> dict[str, float]:
    """Mean recall and nDCG for one relevant id per query."""
    scores = query_vectors @ corpus_vectors.T
    id_to_index = {item_id: index for index, item_id in enumerate(corpus_ids)}
    totals = {f"recall@{k}": 0.0 for k in ks}
    totals.update({f"ndcg@{k}": 0.0 for k in ks})
    for row, gold in enumerate(relevant):
        order = np.argsort(-scores[row])
        ranked = [corpus_ids[int(index)] for index in order]
        gold_index = id_to_index[gold]
        for k in ks:
            top = ranked[:k]
            hit = gold in top
            totals[f"recall@{k}"] += 1.0 if hit else 0.0
            if hit:
                rank = top.index(gold)
                totals[f"ndcg@{k}"] += 1.0 / math.log2(rank + 2)
    count = max(1, len(relevant))
    return {key: round(value / count, 4) for key, value in totals.items()}


def set_f1(gold: list[str], pred: list[str]) -> float:
    """F1 over normalized strings."""
    gold_set = {normalize_sentence(item) for item in gold if item}
    pred_set = {normalize_sentence(item) for item in pred if item}
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    true_positive = len(gold_set & pred_set)
    precision = true_positive / len(pred_set)
    recall = true_positive / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _texts(items: list | None, key: str) -> list[str]:
    values = []
    for item in items or []:
        if isinstance(item, dict) and item.get(key):
            values.append(str(item[key]))
        elif isinstance(item, str):
            values.append(item)
    return values


def structured_score(gold: dict, pred: dict) -> float:
    """Task-specific overlap between a gold JSON object and a prediction."""
    task = gold.get("task")
    if task == "qa":
        answer_ok = normalize_sentence(str(gold.get("answer", ""))) == normalize_sentence(
            str(pred.get("answer", ""))
        )
        cite_ok = str(gold.get("citation", "")).replace(" ", "") == str(
            pred.get("citation", "")
        ).replace(" ", "")
        return float(answer_ok) * 0.7 + float(cite_ok) * 0.3
    if task == "compare":
        gold_obs = []
        pred_obs = []
        for aspect in gold.get("aspects") or []:
            gold_obs.extend(_texts(aspect.get("observations"), "text"))
        for aspect in pred.get("aspects") or []:
            if isinstance(aspect, dict):
                pred_obs.extend(_texts(aspect.get("observations"), "text"))
        return set_f1(gold_obs, pred_obs)
    parts = [
        set_f1(_texts(gold.get("claims"), "text"), _texts(pred.get("claims"), "text")),
    ]
    if task in {"extract", "brief"}:
        parts.append(set_f1(
            _texts(gold.get("limitations"), "text"),
            _texts(pred.get("limitations"), "text"),
        ))
        parts.append(set_f1(
            list(gold.get("open_questions") or []),
            list(pred.get("open_questions") or []),
        ))
    if task in {"brief", "risk"}:
        gold_risks = [
            f"{item.get('statement', '')}|{item.get('severity', '')}|{item.get('action', '')}"
            for item in gold.get("risks") or []
        ]
        pred_risks = [
            f"{item.get('statement', '')}|{item.get('severity', '')}|{item.get('action', '')}"
            for item in pred.get("risks") or []
            if isinstance(item, dict)
        ]
        parts.append(set_f1(gold_risks, pred_risks))
        parts.append(float(
            normalize_sentence(str(gold.get("decision", "")))
            == normalize_sentence(str(pred.get("decision", "")))
        ))
    return sum(parts) / len(parts)
