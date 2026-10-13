"""Shared prompts for supervised training and the agent."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are Helix, a research and decision copilot for technical documents. "
    "Use only the sources provided. Copy claim, limitation, risk, method, and "
    "result sentences from the sources instead of paraphrasing them. "
    "Every copied sentence needs a citation in the form [DOC_ID p.N]. "
    "If a point is not supported, omit it. Output one JSON object and no other text."
)

TASK_INSTRUCTIONS = {
    "extract": (
        "Return JSON with keys task, claims, limitations, open_questions. "
        "task must be \"extract\". "
        "claims and limitations are lists of objects {text, citation}. "
        "open_questions is a list of strings."
    ),
    "brief": (
        "Return JSON with keys task, summary, claims, limitations, "
        "open_questions, risks, decision. task must be \"brief\". "
        "summary and decision are strings. "
        "claims and limitations are lists of {text, citation}. "
        "risks are lists of {statement, severity, action, citation}. "
        "severity is high, medium, or low. open_questions is a list of strings."
    ),
    "risk": (
        "Return JSON with keys task, risks, decision. task must be \"risk\". "
        "risks are objects {statement, severity, action, citation}. "
        "severity is high, medium, or low. decision is a string."
    ),
    "qa": (
        "Return JSON with keys task, answer, citation, supported. "
        "task must be \"qa\". answer is a source sentence. "
        "supported is true only when the sources contain the answer."
    ),
    "compare": (
        "Return JSON with keys task, aspects, summary. task must be \"compare\". "
        "aspects is a list of {name, observations}. "
        "observations are objects {doc_id, text, citation}. "
        "Cover evaluation methodology and reported results."
    ),
}


def sources_block(chunks: list[dict]) -> str:
    """Format retrieved chunks the same way training examples do."""
    parts: list[str] = []
    for chunk in chunks:
        cite = chunk.get("cite") or (
            f"[{chunk['cite_id']} p.{chunk['page']}]"
        )
        if not cite.startswith("["):
            cite = f"[{cite}]"
        parts.append(f"{cite}\n{chunk['text'].strip()}")
    return "\n\n".join(parts)


def user_message(task: str, question: str, chunks: list[dict]) -> str:
    """Build the user turn for a grounded Helix task."""
    instruction = TASK_INSTRUCTIONS[task]
    return (
        f"Task: {task}\n"
        f"Question: {question}\n\n"
        f"Sources:\n{sources_block(chunks)}\n\n"
        f"{instruction}"
    )


def messages(task: str, question: str, chunks: list[dict]) -> list[dict]:
    """Chat messages for one Helix example."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(task, question, chunks)},
    ]
