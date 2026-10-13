# Helix embedding model card

## Model

Contrastive adaptation of `sentence-transformers/all-MiniLM-L6-v2` with in-batch and explicit negatives. The checkpoint is `models/helix-embed`.

## Data

280 training pairs from paraphrased questions over the fourteen training notes. Evaluation uses 24 held-out paraphrases. The retrieval corpus includes every note page plus same-tone distractors (104 passages).

## Training

- Epochs: 8
- Optimizer steps: 72
- Mean loss: 0.9621
- Loss: MultipleNegativesRankingLoss

## Held-out retrieval

| Metric | Base MiniLM | Helix embed |
| --- | --- | --- |
| recall@3 | 0.625 | 0.7083 |
| recall@5 | 0.7083 | 0.8333 |
| nDCG@5 | 0.5785 | 0.6934 |

## Limitations

The query set is small and written against this library. The gain is evidence on these paraphrases, not a general retrieval benchmark.
