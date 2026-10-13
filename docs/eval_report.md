# Helix evaluation report

## Setup

Held-out notes are `HX-015` through `HX-020`. Those pages are absent from supervised training and from embedding-pair training. The language-model comparison gives both models the same source pages and the same instructions. A win requires the tuned structured score to exceed the base score by more than 0.02.

Retrieval is measured separately on paraphrased questions against the full page corpus plus distractors.

## Retrieval

| Metric | Base MiniLM | Helix embed |
| --- | --- | --- |
| recall@5 | 0.7083 | 0.8333 |
| nDCG@5 | 0.5785 | 0.6934 |

## Structured accuracy

Macro score is the unweighted mean of per-example task scores (set overlap on fields, plus exact decision or quoted-answer checks).

| | Base Qwen3-4B-Instruct | Helix LoRA |
| --- | --- | --- |
| Examples | 62 | 62 |
| JSON parsed | 1.0 | 1.0 |
| Macro score | 0.7344 | 0.9648 |

| Task | Base | Helix |
| --- | --- | --- |
| brief | 0.7933 | 0.96 |
| compare | 0.6722 | 0.93 |
| extract | 0.7327 | 0.9444 |
| qa | 0.6562 | 1.0 |
| risk | 0.8333 | 0.9722 |

## Preference

- Tuned wins: 49
- Base wins: 2
- Ties: 11
- Win rate excluding ties: 0.9608
- Win rate including ties: 0.7903

## End-to-end agent

The tuned model was run through the LangGraph agent (plan, retrieve, draft, citation check, one repair) on each held-out note, with that note's pages supplied as the retrieved sources.

- Notes: 6
- Mean citation faithfulness after the checker: 1.0
- Mean structured overlap with the gold brief: 0.9867

## Reading the numbers

Structured score and preference use exact source sentences. A model that paraphrases a correct idea scores lower than one that quotes the source. That matches the product requirement that briefs stay copyable back to a page.

Faithfulness is the fraction of checkable fields that survived the citation overlap check. It is not a human preference study.
