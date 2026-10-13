"""Contrastive fine-tuning of a MiniLM retriever on Helix paraphrases."""

from __future__ import annotations

import json
import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

from helix.config import BASE_EMBED, DOCS_DIR, EMBED_DIR, RUNS_DIR, ensure_dirs
from helix.data.build_retrieval import build_pairs
from helix.eval.metrics import recall_ndcg


def self_features(model: SentenceTransformer, texts: list[str]) -> dict:
    """Tokenize a batch and move tensors to the model device."""
    batch = model.preprocess(texts)
    return {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return np.asarray(
        model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )


def evaluate(model: SentenceTransformer, queries: list[dict], corpus: list[dict]) -> dict[str, float]:
    query_vectors = _encode(model, [row["query"] for row in queries])
    corpus_vectors = _encode(model, [row["text"] for row in corpus])
    return recall_ndcg(
        query_vectors,
        corpus_vectors,
        [row["positive_id"] for row in queries],
        [row["id"] for row in corpus],
    )


def train(epochs: int = 8, batch_size: int = 32, lr: float = 2e-5, seed: int = 7) -> dict:
    """Fine-tune the embedder and return baseline versus tuned metrics."""
    ensure_dirs()
    train_pairs, eval_queries, corpus = build_pairs()
    rng = random.Random(seed)
    negatives_pool = [row["text"] for row in corpus]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline = SentenceTransformer(BASE_EMBED, device=device)
    before = evaluate(baseline, eval_queries, corpus)
    del baseline
    model = SentenceTransformer(BASE_EMBED, device=device)
    loss_fn = MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    steps = 0
    losses = []
    for epoch in range(epochs):
        order = list(range(len(train_pairs)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size]
            if len(batch_ids) < 4:
                continue
            anchors = [train_pairs[index]["query"] for index in batch_ids]
            positives = [train_pairs[index]["positive"] for index in batch_ids]
            negatives = []
            for index in batch_ids:
                positive = train_pairs[index]["positive"]
                negative = rng.choice(negatives_pool)
                while negative == positive:
                    negative = rng.choice(negatives_pool)
                negatives.append(negative)
            features = [self_features(model, texts) for texts in (anchors, positives, negatives)]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(features, torch.zeros(len(anchors), device=model.device))
            loss.backward()
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().cpu()))
        print(f"epoch {epoch + 1} mean_loss {sum(losses[-20:]) / max(1, len(losses[-20:])):.4f}", flush=True)
    model.save_pretrained(EMBED_DIR)
    after = evaluate(model, eval_queries, corpus)
    report = {
        "base": BASE_EMBED,
        "train_pairs": len(train_pairs),
        "eval_queries": len(eval_queries),
        "corpus": len(corpus),
        "epochs": epochs,
        "steps": steps,
        "mean_loss": round(sum(losses) / max(1, len(losses)), 4),
        "before": before,
        "after": after,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "embed_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_card(report)
    print(json.dumps(report, indent=2))
    return report


def _write_card(report: dict) -> None:
    before = report["before"]
    after = report["after"]
    card = f"""# Helix embedding model card

## Model

Contrastive adaptation of `{report["base"]}` with in-batch and explicit negatives. The checkpoint is `models/helix-embed`.

## Data

{report["train_pairs"]} training pairs from paraphrased questions over the fourteen training notes. Evaluation uses {report["eval_queries"]} held-out paraphrases. The retrieval corpus includes every note page plus same-tone distractors ({report["corpus"]} passages).

## Training

- Epochs: {report["epochs"]}
- Optimizer steps: {report["steps"]}
- Mean loss: {report["mean_loss"]}
- Loss: MultipleNegativesRankingLoss

## Held-out retrieval

| Metric | Base MiniLM | Helix embed |
| --- | --- | --- |
| recall@3 | {before["recall@3"]} | {after["recall@3"]} |
| recall@5 | {before["recall@5"]} | {after["recall@5"]} |
| nDCG@5 | {before["ndcg@5"]} | {after["ndcg@5"]} |

## Limitations

The query set is small and written against this library. The gain is evidence on these paraphrases, not a general retrieval benchmark.
"""
    (DOCS_DIR / "model_card_embed.md").write_text(card, encoding="utf-8")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
