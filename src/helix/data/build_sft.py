"""Build supervised examples for Helix brief, extraction, and comparison tasks."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from helix.config import DOCS_DIR, SFT_DIR, ensure_dirs
from helix.corpus.library import COMPARISONS, citation_for, documents
from helix.prompts import messages

TRAIN_QUESTIONS = {
    "extract": [
        "What are the key technical claims, limitations, and open questions?",
        "Extract the findings, caveats, and unresolved questions.",
        "List the claims and the limits of the evidence.",
        "Pull structured findings from this note.",
        "What did the authors establish, and what did they leave open?",
        "Identify claims, limitations, and open questions only.",
        "Turn this note into structured findings.",
        "What should a reviewer record as claims and caveats?",
    ],
    "brief": [
        "Write a decision brief with citations.",
        "Summarize the note for an operator who must decide.",
        "Produce a grounded brief covering claims, risks, and the decision.",
        "Draft the brief an analyst would send upstairs.",
        "Give me the decision-ready summary of this source.",
        "Prepare a cited brief with risks and next actions.",
        "What is the operator takeaway, with support?",
        "Build a short brief that stays inside the source.",
    ],
    "risk": [
        "Flag risks and the sourced actions.",
        "What could go wrong, and what action is written down?",
        "Give a risk summary with prioritized actions.",
        "Which risks are stated, and how severe are they?",
        "List exposures and the recommended actions from the source.",
        "What should be flagged before a decision?",
        "Produce the risk section only.",
        "Identify severity, statement, and action for each risk.",
    ],
    "qa": [
        "Which sentence answers this: {hint}?",
        "Find the source sentence for this question: {hint}",
        "Answer from the sources only: {hint}",
        "Quote the supporting sentence. Question: {hint}",
    ],
}

EVAL_QUESTIONS = {
    "extract": [
        "What claims, limitations, and open questions are actually supported?",
        "Structure the findings without adding outside knowledge.",
    ],
    "brief": [
        "Draft a cited decision brief from these sources.",
        "What brief would you hand a decision-maker?",
    ],
    "risk": [
        "Summarize the risks and the actions the source already states.",
        "Prioritize the written risks and keep their actions.",
    ],
    "qa": [
        "Quote the sentence that answers: {hint}",
        "From the sources, what answers this: {hint}",
    ],
    "compare": [
        "How do these approaches differ on evaluation methodology and reported results?",
        "Compare the measurement setup and the figures these sources report.",
    ],
}


def _by_id() -> dict:
    return {doc.doc_id: doc for doc in documents()}


def _claim_items(doc, sentences: list[str] | tuple[str, ...]) -> list[dict]:
    return [
        {"text": sentence, "citation": citation_for(doc, sentence)}
        for sentence in sentences
    ]


def _risk_items(doc) -> list[dict]:
    return [
        {
            "statement": risk.statement,
            "severity": risk.severity,
            "action": risk.action,
            "citation": citation_for(doc, risk.statement),
        }
        for risk in doc.risks
    ]


def _gold(task: str, docs: list) -> dict:
    if task == "extract":
        doc = docs[0]
        return {
            "task": "extract",
            "claims": _claim_items(doc, doc.claims),
            "limitations": _claim_items(doc, doc.limitations),
            "open_questions": list(doc.open_questions),
        }
    if task == "brief":
        claims = []
        limitations = []
        questions = []
        risks = []
        decisions = []
        for doc in docs:
            claims.extend(_claim_items(doc, doc.claims))
            limitations.extend(_claim_items(doc, doc.limitations))
            questions.extend(doc.open_questions)
            risks.extend(_risk_items(doc))
            decisions.append(doc.decision)
        summary = " ".join(decisions)
        return {
            "task": "brief",
            "summary": summary,
            "claims": claims,
            "limitations": limitations,
            "open_questions": questions,
            "risks": risks,
            "decision": summary,
        }
    if task == "risk":
        doc = docs[0]
        return {
            "task": "risk",
            "risks": _risk_items(doc),
            "decision": doc.decision,
        }
    if task == "qa":
        raise RuntimeError("qa gold is built per question")
    if task == "compare":
        aspects = []
        for name, attr in (
            ("evaluation methodology", "method"),
            ("reported results", "result"),
        ):
            observations = []
            for doc in docs:
                sentence = getattr(doc, attr)
                observations.append({
                    "doc_id": doc.doc_id,
                    "text": sentence,
                    "citation": citation_for(doc, sentence),
                })
            aspects.append({"name": name, "observations": observations})
        summary = " ".join(doc.decision for doc in docs)
        return {"task": "compare", "aspects": aspects, "summary": summary}
    raise KeyError(task)


def _example(task: str, question: str, docs: list, gold: dict, split: str) -> dict:
    chunks = []
    for doc in docs:
        chunks.extend(doc.chunks())
    return {
        "messages": messages(task, question, chunks) + [
            {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)}
        ],
        "meta": {
            "split": split,
            "task": task,
            "doc_ids": [doc.doc_id for doc in docs],
        },
    }


def build_examples(seed: int = 7) -> tuple[list[dict], list[dict]]:
    """Create train and held-out supervised examples."""
    rng = random.Random(seed)
    library = documents()
    by_id = _by_id()
    train: list[dict] = []
    eval_rows: list[dict] = []
    for doc in library:
        split_rows = train if doc.split == "train" else eval_rows
        question_bank = TRAIN_QUESTIONS if doc.split == "train" else EVAL_QUESTIONS
        for task in ("extract", "brief", "risk"):
            for question in question_bank[task]:
                split_rows.append(
                    _example(task, question, [doc], _gold(task, [doc]), doc.split)
                )
        qa_bank = question_bank["qa"]
        spans = list(doc.claims) + list(doc.limitations) + [doc.decision]
        for span in spans:
            hint = " ".join(span.split()[:8])
            for template in qa_bank:
                gold = {
                    "task": "qa",
                    "answer": span,
                    "citation": citation_for(doc, span),
                    "supported": True,
                }
                split_rows.append(
                    _example(task="qa", question=template.format(hint=hint), docs=[doc], gold=gold, split=doc.split)
                )
    for pair in COMPARISONS:
        left = by_id[pair["left"]]
        right = by_id[pair["right"]]
        docs = [left, right]
        gold = _gold("compare", docs)
        if left.split == "train":
            questions = [
                pair["question"],
                "Compare evaluation methodology and reported results.",
                "Where do the measurement setups and the reported figures differ?",
                "What did each source measure, and what did each source report?",
            ]
        else:
            questions = EVAL_QUESTIONS["compare"]
        for question in questions:
            row = _example("compare", question, docs, gold, left.split)
            (train if left.split == "train" else eval_rows).append(row)
    train_docs = [doc for doc in library if doc.split == "train"]
    pair_indexes = list(range(len(train_docs)))
    extra_pairs = []
    for i, left in enumerate(train_docs):
        for right in train_docs[i + 1 :]:
            extra_pairs.append((left, right))
    rng.shuffle(extra_pairs)
    for left, right in extra_pairs[:24]:
        gold = _gold("brief", [left, right])
        for question in TRAIN_QUESTIONS["brief"][:4]:
            train.append(_example("brief", question, [left, right], gold, "train"))
    expanded: list[dict] = []
    for row in train:
        expanded.append(row)
        for prefix in ("Using only the sources, ", "For the decision desk: "):
            cloned = copy.deepcopy(row)
            content = cloned["messages"][1]["content"]
            cloned["messages"][1]["content"] = content.replace(
                "Question: ", f"Question: {prefix}", 1
            )
            expanded.append(cloned)
    rng.shuffle(expanded)
    rng.shuffle(eval_rows)
    return expanded, eval_rows


def write_datasets(path: Path | None = None) -> dict:
    """Write jsonl splits and return counts."""
    ensure_dirs()
    out = path or SFT_DIR
    out.mkdir(parents=True, exist_ok=True)
    train, eval_rows = build_examples()
    for name, rows in ("train", train), ("eval", eval_rows):
        target = out / f"{name}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "train": len(train),
        "eval": len(eval_rows),
        "train_tasks": _count_tasks(train),
        "eval_tasks": _count_tasks(eval_rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_dataset_card(summary)
    return summary


def _count_tasks(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        task = row["meta"]["task"]
        counts[task] = counts.get(task, 0) + 1
    return counts


def _write_dataset_card(summary: dict) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    card = f"""# Helix supervised dataset card

## Summary

Helix instruction data is built from an original library of twenty technical notes. Fourteen notes are used for training. Six notes (`HX-015` through `HX-020`) are held out entirely, including every page of those notes.

Gold answers are derived from the note metadata. Claim, limitation, method, result, risk, and decision strings are copied into the page text, so a correct answer is a source sentence with a citation.

## Counts

- Training examples: {summary["train"]}
- Held-out examples: {summary["eval"]}
- Training tasks: {json.dumps(summary["train_tasks"])}
- Held-out tasks: {json.dumps(summary["eval_tasks"])}

## Tasks

- `extract`: claims, limitations, and open questions
- `brief`: decision brief with risks and citations
- `risk`: severity, statement, and sourced action
- `qa`: quote the supporting sentence
- `compare`: methodology and reported results, cited per document

## Construction

Question wording is templated. Training questions are also prefixed with two instruction wrappers so the model sees the same gold under slightly different requests. Held-out questions use wrappers that do not appear in training.

The notes are original. They are not excerpts from papers, advisories, or other copyrighted sources.

## Intended use

Supervised fine-tuning of an open instruct model for grounded technical briefs. The held-out split is for structured accuracy and preference comparisons against the base model.

## Limitations

The library is small and stylistically related, even though the held-out notes use different facts. Results describe this Helix task mixture, not general question answering.
"""
    (DOCS_DIR / "dataset_card.md").write_text(card, encoding="utf-8")


def main() -> None:
    summary = write_datasets()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
