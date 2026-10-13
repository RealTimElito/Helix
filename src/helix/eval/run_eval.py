"""Compare the base model with the Helix adapter on held-out notes."""

from __future__ import annotations

import json
import random

from helix.agent.graph import run_agent
from helix.config import DOCS_DIR, RUNS_DIR, SFT_DIR, ensure_dirs
from helix.eval.metrics import structured_score
from helix.models.llm import LocalLLM
from helix.textutil import parse_json_object

_MARGIN = 0.02


def _load_eval() -> list[dict]:
    rows = []
    with (SFT_DIR / "eval.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sample(rows: list[dict], per_task: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["meta"]["task"], []).append(row)
    chosen = []
    for task, items in grouped.items():
        rng.shuffle(items)
        chosen.extend(items[:per_task])
    return chosen


def _score_model(llm: LocalLLM, rows: list[dict]) -> list[dict]:
    scored = []
    batch_size = 4
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [row["messages"][:-1] for row in batch]
        try:
            outputs = llm.generate_batch(prompts, max_new_tokens=640)
        except Exception as exc:  # noqa: BLE001
            outputs = [f"ERROR {exc}" for _ in batch]
        for row, output in zip(batch, outputs):
            gold = json.loads(row["messages"][-1]["content"])
            try:
                pred = parse_json_object(output)
                score = structured_score(gold, pred)
                parsed = True
            except Exception:  # noqa: BLE001
                pred = {}
                score = 0.0
                parsed = False
            scored.append({
                "task": row["meta"]["task"],
                "doc_ids": row["meta"]["doc_ids"],
                "score": round(score, 4),
                "parsed": parsed,
            })
        done = min(start + batch_size, len(rows))
        print(f"scored {done}/{len(rows)}", flush=True)
    return scored


def _summarize(scored: list[dict]) -> dict:
    by_task: dict[str, list[float]] = {}
    for row in scored:
        by_task.setdefault(row["task"], []).append(row["score"])
    macro = sum(row["score"] for row in scored) / max(1, len(scored))
    return {
        "n": len(scored),
        "macro": round(macro, 4),
        "parsed_rate": round(sum(row["parsed"] for row in scored) / max(1, len(scored)), 4),
        "by_task": {
            task: round(sum(values) / len(values), 4) for task, values in sorted(by_task.items())
        },
    }


def _preference(base: list[dict], tuned: list[dict]) -> dict:
    wins = ties = losses = 0
    for left, right in zip(base, tuned):
        delta = right["score"] - left["score"]
        if delta > _MARGIN:
            wins += 1
        elif delta < -_MARGIN:
            losses += 1
        else:
            ties += 1
    decisive = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_excluding_ties": round(wins / decisive, 4) if decisive else 0.0,
        "win_rate_including_ties": round(wins / max(1, wins + losses + ties), 4),
        "margin": _MARGIN,
    }


def _end_to_end(llm: LocalLLM, limit: int = 6) -> dict:
    from helix.corpus.library import documents

    heldout = [doc for doc in documents() if doc.split == "eval"][:limit]
    faithfulness = []
    scores = []

    def generate(prompt: list[dict], max_new_tokens: int) -> str:
        return llm.generate(prompt, max_new_tokens=max_new_tokens)

    for doc in heldout:
        chunks = doc.chunks()
        for chunk in chunks:
            chunk["score"] = 1.0

        def search(_collection, _query, search_limit, _docs, chunks=chunks):
            return chunks[:search_limit]

        result = run_agent(
            question="What are the key technical claims, limitations, and open questions?",
            mode="brief",
            collection_id="eval",
            document_ids=None,
            generate=generate,
            search=search,
        )
        faithfulness.append(result["faithfulness"])
        gold = {
            "task": "brief",
            "claims": [{"text": claim} for claim in doc.claims],
            "limitations": [{"text": item} for item in doc.limitations],
            "open_questions": list(doc.open_questions),
            "risks": [
                {"statement": risk.statement, "severity": risk.severity, "action": risk.action}
                for risk in doc.risks
            ],
            "decision": doc.decision,
        }
        scores.append(structured_score(gold, result["answer"]))
        print(f"e2e {doc.doc_id} faith {result['faithfulness']}", flush=True)
    return {
        "n": len(heldout),
        "mean_faithfulness": round(sum(faithfulness) / max(1, len(faithfulness)), 4),
        "mean_structured": round(sum(scores) / max(1, len(scores)), 4),
    }


def _write_report(report: dict) -> None:
    base = report["base"]
    tuned = report["tuned"]
    pref = report["preference"]
    embed = report.get("embed")
    embed_lines = "Embedding metrics were not found. Run training first."
    if embed:
        before, after = embed["before"], embed["after"]
        embed_lines = (
            f"| recall@5 | {before['recall@5']} | {after['recall@5']} |\n"
            f"| nDCG@5 | {before['ndcg@5']} | {after['ndcg@5']} |"
        )
    task_rows = []
    for task in sorted(set(base["by_task"]) | set(tuned["by_task"])):
        task_rows.append(
            f"| {task} | {base['by_task'].get(task, '—')} | {tuned['by_task'].get(task, '—')} |"
        )
    e2e = report["e2e_tuned"]
    text = f"""# Helix evaluation report

## Setup

Held-out notes are `HX-015` through `HX-020`. Those pages are absent from supervised training and from embedding-pair training. The language-model comparison gives both models the same source pages and the same instructions. A win requires the tuned structured score to exceed the base score by more than {pref["margin"]}.

Retrieval is measured separately on paraphrased questions against the full page corpus plus distractors.

## Retrieval

| Metric | Base MiniLM | Helix embed |
| --- | --- | --- |
{embed_lines}

## Structured accuracy

Macro score is the unweighted mean of per-example task scores (set overlap on fields, plus exact decision or quoted-answer checks).

| | Base Qwen3-4B-Instruct | Helix LoRA |
| --- | --- | --- |
| Examples | {base["n"]} | {tuned["n"]} |
| JSON parsed | {base["parsed_rate"]} | {tuned["parsed_rate"]} |
| Macro score | {base["macro"]} | {tuned["macro"]} |

| Task | Base | Helix |
| --- | --- | --- |
{chr(10).join(task_rows)}

## Preference

- Tuned wins: {pref["wins"]}
- Base wins: {pref["losses"]}
- Ties: {pref["ties"]}
- Win rate excluding ties: {pref["win_rate_excluding_ties"]}
- Win rate including ties: {pref["win_rate_including_ties"]}

## End-to-end agent

The tuned model was run through the LangGraph agent (plan, retrieve, draft, citation check, one repair) on each held-out note, with that note's pages supplied as the retrieved sources.

- Notes: {e2e["n"]}
- Mean citation faithfulness after the checker: {e2e["mean_faithfulness"]}
- Mean structured overlap with the gold brief: {e2e["mean_structured"]}

## Reading the numbers

Structured score and preference use exact source sentences. A model that paraphrases a correct idea scores lower than one that quotes the source. That matches the product requirement that briefs stay copyable back to a page.

Faithfulness is the fraction of checkable fields that survived the citation overlap check. It is not a human preference study.
"""
    (DOCS_DIR / "eval_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    rows = _sample(_load_eval(), per_task=16, seed=7)
    print(f"eval examples {len(rows)}", flush=True)
    base_llm = LocalLLM()
    base_scores = _score_model(base_llm, rows)
    base_llm.close()
    from helix.config import ADAPTER_DIR

    tuned_llm = LocalLLM(adapter=ADAPTER_DIR)
    tuned_scores = _score_model(tuned_llm, rows)
    e2e = _end_to_end(tuned_llm)
    tuned_llm.close()
    embed = None
    embed_path = RUNS_DIR / "embed_metrics.json"
    if embed_path.exists():
        embed = json.loads(embed_path.read_text(encoding="utf-8"))
    report = {
        "base": _summarize(base_scores),
        "tuned": _summarize(tuned_scores),
        "preference": _preference(base_scores, tuned_scores),
        "e2e_tuned": e2e,
        "embed": embed,
        "pairs": [
            {"base": left["score"], "tuned": right["score"], "task": left["task"]}
            for left, right in zip(base_scores, tuned_scores)
        ],
    }
    (RUNS_DIR / "llm_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_report(report)
    print(json.dumps({k: report[k] for k in ("base", "tuned", "preference", "e2e_tuned")}, indent=2))


if __name__ == "__main__":
    main()
