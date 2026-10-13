"""Multi-step Helix agent: plan, retrieve, draft, and grounding repair."""

from __future__ import annotations

import re
from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from helix.agent.faithfulness import ground_payload
from helix.index.search import collection_note_count, load_collection_pages
from helix.prompts import messages
from helix.textutil import parse_json_object

_QUOTE_MODES = frozenset({"brief", "extract", "risk"})
_SIBLING_MODES = frozenset({"brief", "extract"})
_LIST_FIELDS = ("claims", "limitations", "risks", "open_questions")
_MIN_WORDS = 6
_CLAIMS_PER_DOCUMENT = 2
_NOT_FOUND = "Not found in the source."
_TITLE_PREFIXES = (
    "minutes for ",
    "lab note for ",
    "memorandum on ",
    "advisory:",
)
_CLAIM_BOILERPLATE = (
    "the group recorded these points",
    "the observations follow",
)
_DECISION_MARKERS = (
    "decision recorded:",
    "operator decision:",
    "the working decision is:",
    "recommended posture:",
)
_LIMIT_CUES = (
    "limit",
    "caveat",
    "does not",
    "unresolved",
    "not show",
    "only whether",
    "should not be generalized",
)
_LEADING_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9'’/ -]{0,48}:\s+")
_MEASUREMENT_LABELS = (
    "method note",
    "how it was measured",
    "evaluation setup discussed",
    "assessment method",
    "reported result",
    "what came back",
    "figures entered in the minutes",
    "observed outcome",
)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:\.\d+[^.!?\n]*)*[.!?]")
_PAGE_TITLE_RE = re.compile(
    r"(?i)^(minutes for |lab note for |memorandum on |advisory:\s+)(.+?)\."
)
_RISK_RE = re.compile(
    r"Severity\s+(high|medium|low):\s*(.+?[.])\s*Recommended action:\s*(.+?[.])",
    re.IGNORECASE,
)


class AgentState(TypedDict, total=False):
    question: str
    mode: str
    collection_id: str
    document_ids: list[str]
    search_query: str
    chunks: list[dict]
    draft: dict
    grounded: dict
    dropped: list[dict]
    faithfulness: float
    repairs: int
    needs_repair: bool
    notes_in_collection: int
    parse_failed: bool
    trajectory: list[dict]
    error: str


GenerateFn = Callable[[list[dict], int], str]
SearchFn = Callable[[str, str, int, list[str] | None], list[dict]]


def build_graph(generate: GenerateFn, search: SearchFn):
    """Compile the Helix tool graph."""

    def plan(state: AgentState) -> AgentState:
        mode = state.get("mode") or "brief"
        question = state["question"]
        if _collection_quote_task({**state, "mode": mode}):
            trajectory = list(state.get("trajectory") or [])
            trajectory.append({
                "tool": "plan",
                "detail": "Checking every note in the collection; no search rewrite.",
                "search_query": question,
                "mode": mode,
            })
            return {
                "search_query": question,
                "mode": mode,
                "trajectory": trajectory,
                "repairs": 0,
            }
        search_query = question
        note = "Using the question as the search query."
        try:
            raw = generate(
                [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the user question as a short search query. "
                            "Return JSON {\"search_query\": \"...\"} only."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                80,
            )
            parsed = parse_json_object(raw)
            if parsed.get("search_query"):
                search_query = str(parsed["search_query"])
                note = "Rewrote the question into a search query."
        except Exception as exc:  # noqa: BLE001 - recovery is the point of this node
            note = f"Query rewrite failed ({exc.__class__.__name__}); using the raw question."
        trajectory = list(state.get("trajectory") or [])
        trajectory.append({
            "tool": "plan",
            "detail": note,
            "search_query": search_query,
            "mode": mode,
        })
        return {"search_query": search_query, "mode": mode, "trajectory": trajectory, "repairs": 0}

    def retrieve(state: AgentState) -> AgentState:
        if _collection_quote_task(state):
            chunks = load_collection_pages(state["collection_id"])
            note_count = len({_document_key(chunk) for chunk in chunks})
            detail = f"Loaded {len(chunks)} pages from {note_count} notes."
        else:
            chunks = search(
                state["collection_id"],
                state.get("search_query") or state["question"],
                8,
                state.get("document_ids"),
            )
            if state.get("document_ids") and len(chunks) < 2:
                chunks = search(
                    state["collection_id"],
                    state["question"],
                    8,
                    state.get("document_ids"),
                )
            note_count = collection_note_count(state["collection_id"])
            detail = f"Retrieved {len(chunks)} chunks."
        trajectory = list(state.get("trajectory") or [])
        trajectory.append({
            "tool": "retrieve",
            "detail": detail,
            "citations": [chunk["cite"] for chunk in chunks],
        })
        return {
            "chunks": chunks,
            "trajectory": trajectory,
            "notes_in_collection": note_count,
        }

    def draft(state: AgentState) -> AgentState:
        return _draft(state, generate, repair=False)

    def repair(state: AgentState) -> AgentState:
        return _draft(state, generate, repair=True)

    def verify(state: AgentState) -> AgentState:
        draft = state.get("draft") or {}
        chunks = state.get("chunks") or []
        result = ground_payload(draft, chunks) if draft else {
            "payload": {},
            "dropped": [],
            "faithfulness": 0.0,
        }
        parse_failed = bool(state.get("parse_failed"))
        if parse_failed and not _supported_content(result["payload"]):
            result["faithfulness"] = 0.0
        repairs = int(state.get("repairs") or 0)
        needs = repairs < 1 and bool(chunks) and (bool(result["dropped"]) or parse_failed)
        filled = False
        mode = state.get("mode") or "brief"
        if not needs and _needs_quotes(mode, chunks, result["payload"]):
            quoted = quote_payload(mode, chunks)
            result = ground_payload(quoted, chunks)
            if not _has_list_content(result["payload"]):
                result["faithfulness"] = 0.0
            filled = _has_list_content(result["payload"])
        trajectory = list(state.get("trajectory") or [])
        detail = (
            f"Faithfulness {result['faithfulness']}. "
            f"Dropped {len(result['dropped'])} unsupported items."
        )
        if filled:
            detail += " Quoted source sentences into the empty answer."
        trajectory.append({
            "tool": "cite_check",
            "detail": detail,
            "faithfulness": result["faithfulness"],
        })
        update: AgentState = {
            "grounded": result["payload"],
            "dropped": result["dropped"],
            "faithfulness": result["faithfulness"],
            "needs_repair": needs,
            "trajectory": trajectory,
        }
        if _collection_quote_task(state) and "notes" in result["payload"]:
            result["payload"]["notes"] = fill_note_gaps(result["payload"]["notes"])
        if filled:
            update["error"] = ""
        return update

    def route(state: AgentState) -> str:
        if state.get("needs_repair"):
            return "repair"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("draft", draft)
    graph.add_node("verify", verify)
    graph.add_node("repair", repair)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "verify")
    graph.add_conditional_edges("verify", route, {"repair": "repair", END: END})
    graph.add_edge("repair", "verify")
    return graph.compile()


def _draft(state: AgentState, generate: GenerateFn, repair: bool) -> AgentState:
    mode = state.get("mode") or "brief"
    chunks = state.get("chunks") or []
    trajectory = list(state.get("trajectory") or [])
    if not chunks:
        trajectory.append({"tool": mode, "detail": "No sources retrieved."})
        return {
            "draft": {
                "task": mode,
                "summary": "No supporting sources were retrieved.",
                "claims": [],
                "decision": "",
            },
            "trajectory": trajectory,
            "error": "no sources",
        }
    if not repair and _collection_quote_task(state):
        trajectory.append({
            "tool": mode,
            "detail": f"Drafted a {mode} from quoted source sentences.",
        })
        return {
            "draft": quote_payload(mode, chunks),
            "trajectory": trajectory,
            "error": "",
            "parse_failed": False,
        }
    prompt = messages(mode if mode in {"extract", "brief", "risk", "qa", "compare"} else "brief", state["question"], chunks)
    if repair and state.get("parse_failed"):
        prompt = prompt + [{
            "role": "user",
            "content": (
                "The previous response was not valid JSON. Rewrite the JSON. "
                "Keep only sentences copied from the sources, with citations."
            ),
        }]
    elif repair:
        rejected = state.get("dropped") or []
        prompt = prompt + [{
            "role": "user",
            "content": (
                "The citation check rejected these items because the cited page "
                f"did not support them: {rejected}. Rewrite the JSON. Keep only "
                "sentences copied from the sources, with citations."
            ),
        }]
    try:
        raw = generate(prompt, 700)
        parsed = parse_json_object(raw)
        detail = "Repaired the draft." if repair else f"Drafted a {mode} response."
        error = ""
        parse_failed = False
    except Exception as exc:  # noqa: BLE001
        parsed = {"task": mode, "summary": "", "claims": []}
        detail = f"Draft failed ({exc.__class__.__name__})."
        error = str(exc)
        parse_failed = True
    trajectory.append({"tool": "repair" if repair else mode, "detail": detail})
    update: AgentState = {
        "draft": parsed,
        "trajectory": trajectory,
        "error": error,
        "parse_failed": parse_failed,
    }
    if repair:
        update["repairs"] = int(state.get("repairs") or 0) + 1
        update["needs_repair"] = False
    return update


def _collection_quote_task(state: AgentState) -> bool:
    """Unfiltered collection briefs and extracts are drafted from quotes."""
    mode = state.get("mode") or "brief"
    return mode in _SIBLING_MODES and not state.get("document_ids")


def _needs_quotes(mode: str, chunks: list[dict], payload: dict) -> bool:
    """Whether a brief, extract, or risk answer should be filled from quotes."""
    return mode in _QUOTE_MODES and bool(chunks) and not _has_list_content(payload)


def _has_list_content(payload: dict) -> bool:
    """Return whether any claim, limitation, question, or risk survived."""
    return any(payload.get(key) for key in _LIST_FIELDS)


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < right and end > left for left, right in occupied)


def _summary_rank(chunk: dict) -> tuple[int, int]:
    """Sort summary and page 1 chunks ahead of method and result pages."""
    try:
        page = int(chunk.get("page") or 99)
    except (TypeError, ValueError):
        page = 99
    section = str(chunk.get("section") or "").lower()
    summary = page == 1 or "summary" in section
    return (0 if summary else 1, page)


def _is_measurement_sentence(sentence: str) -> bool:
    """Return whether a sentence is measurement setup or a reported figure."""
    lowered = sentence.lower().lstrip()
    return any(lowered.startswith(f"{label}:") for label in _MEASUREMENT_LABELS)


def _is_summary_page(chunk: dict) -> bool:
    """Return whether a chunk is the opening summary page of a note."""
    try:
        page = int(chunk.get("page") or 99)
    except (TypeError, ValueError):
        page = 99
    section = str(chunk.get("section") or "").lower()
    return page == 1 or "summary" in section


def _is_title_sentence(sentence: str) -> bool:
    """Return whether a sentence is a note title rather than a finding."""
    lowered = sentence.lower().lstrip()
    return any(lowered.startswith(prefix) for prefix in _TITLE_PREFIXES)


def _title_from_page(text: str) -> str:
    """Return the document title inside a page-one boilerplate sentence."""
    match = _PAGE_TITLE_RE.match(text.strip())
    if not match:
        return ""
    return match.group(2).strip()


def _note_title(group: list[dict]) -> str:
    """Prefer the stored document title over the page boilerplate prefix."""
    for chunk in group:
        title = str(chunk.get("title") or "").strip()
        if title:
            return title
    for chunk in group:
        parsed = _title_from_page(str(chunk.get("text") or ""))
        if parsed:
            return parsed
    return _document_key(group[0])


def _is_boilerplate_sentence(sentence: str) -> bool:
    """Return whether a sentence is page scaffolding rather than a finding."""
    lowered = " ".join(sentence.lower().split()).rstrip(".!?")
    return any(
        lowered == phrase or lowered.startswith(f"{phrase} ")
        for phrase in _CLAIM_BOILERPLATE
    )


def _is_decision_sentence(sentence: str) -> bool:
    """Return whether a sentence records a decision rather than a finding."""
    lowered = sentence.lower().lstrip()
    return any(lowered.startswith(marker) for marker in _DECISION_MARKERS)


def _strip_label(sentence: str, source: str) -> str | None:
    """Drop a short leading label and keep an exact span of the source."""
    match = _LEADING_LABEL.match(sentence)
    if not match:
        return sentence if sentence in source else None
    body = sentence[match.end():].strip()
    if body and body in source:
        return body
    return None


def _document_key(chunk: dict) -> str:
    """Identify a note so later pages share one selection slot."""
    return str(
        chunk.get("cite_id")
        or chunk.get("document_id")
        or chunk.get("cite")
        or ""
    )


def _groups_in_retrieval_order(chunks: list[dict]) -> list[list[dict]]:
    """Group by note in first-seen order, summary page first."""
    order: list[str] = []
    grouped: dict[str, list[dict]] = {}
    for chunk in chunks:
        key = _document_key(chunk)
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(chunk)
    return [sorted(grouped[key], key=_summary_rank) for key in order]


def _take_round(
    buckets: list[dict],
    kind: str,
    dest: list,
    cap: int,
    counts: dict[str, int] | None = None,
    per_doc: int | None = None,
) -> None:
    """Give every note one pending item before any note receives another."""
    for bucket in buckets:
        if len(dest) >= cap:
            return
        doc_key = bucket["doc"]
        already = counts.get(doc_key, 0) if counts is not None else 0
        if per_doc is not None and already >= per_doc:
            continue
        pending = bucket[kind]
        if not pending:
            continue
        dest.append(pending.pop(0))
        if counts is not None:
            counts[doc_key] = counts.get(doc_key, 0) + 1


def quote_payload(mode: str, chunks: list[dict]) -> dict:
    """Quote exact chunk sentences into claims, limitations, questions, and risks."""
    risks: list[dict] = []
    seen: set[str] = set()
    buckets: list[dict] = []

    def fresh(text: str) -> bool:
        key = " ".join(text.split()).lower()
        if key in seen:
            return False
        seen.add(key)
        return True

    for group in _groups_in_retrieval_order(chunks):
        bucket: dict = {
            "doc": _document_key(group[0]),
            "title": _note_title(group),
            "claims": [],
            "limitations": [],
            "questions": [],
        }
        for chunk in group:
            citation = f"[{chunk['cite_id']} p.{int(chunk['page'])}]"
            text = str(chunk.get("text") or "")
            occupied: list[tuple[int, int]] = []
            lowered_text = text.lower()
            if "severity" in lowered_text and "recommended action" in lowered_text:
                for match in _RISK_RE.finditer(text):
                    statement = match.group(2).strip()
                    action = match.group(3).strip()
                    if (
                        statement in text
                        and action in text
                        and not _is_title_sentence(statement)
                        and len(statement.split()) >= _MIN_WORDS
                        and len(action.split()) >= _MIN_WORDS
                        and fresh(statement)
                    ):
                        risks.append({
                            "statement": statement,
                            "severity": match.group(1).lower(),
                            "action": action,
                            "citation": citation,
                        })
                        occupied.append(match.span())
            skip_lede = False
            after_decision = False
            summary_page = _is_summary_page(chunk)
            for match in _SENTENCE_RE.finditer(text):
                if _overlaps(match.span(), occupied):
                    continue
                raw = match.group(0).strip()
                if _is_title_sentence(raw):
                    skip_lede = summary_page
                    continue
                if skip_lede:
                    skip_lede = False
                    continue
                if after_decision or _is_decision_sentence(raw):
                    after_decision = True
                    continue
                if _is_measurement_sentence(raw) or _is_boilerplate_sentence(raw):
                    continue
                quoted = _strip_label(raw, text)
                if quoted is None or len(quoted.split()) < _MIN_WORDS or not fresh(quoted):
                    continue
                if quoted.endswith("?"):
                    bucket["questions"].append(f"{quoted} {citation}")
                    continue
                if any(cue in raw.lower() for cue in _LIMIT_CUES):
                    bucket["limitations"].append({"text": quoted, "citation": citation})
                    continue
                bucket["claims"].append({"text": quoted, "citation": citation})
        buckets.append(bucket)

    claims: list[dict] = []
    limitations: list[dict] = []
    questions: list[str] = []
    claim_counts: dict[str, int] = {}
    caps = _selection_caps(len(buckets))
    _take_round(
        buckets, "claims", claims, caps["claims"],
        claim_counts, 1,
    )
    _take_round(buckets, "limitations", limitations, caps["limitations"])
    _take_round(buckets, "questions", questions, caps["open_questions"])
    _take_round(
        buckets, "claims", claims, caps["claims"],
        claim_counts, _CLAIMS_PER_DOCUMENT,
    )

    payload: dict = {
        "task": mode if mode in _QUOTE_MODES else "brief",
        "claims": claims,
        "limitations": limitations,
        "open_questions": questions,
        "risks": risks,
    }
    if payload["task"] in {"brief", "risk"}:
        payload["summary"] = ""
        payload["decision"] = ""
    if payload["task"] in _SIBLING_MODES:
        payload["notes"] = fill_note_gaps(
            _note_blocks(buckets, claims, limitations, questions)
        )
    return payload


def _selection_caps(note_count: int) -> dict[str, int]:
    """Fit one claim, limitation, and question per note, then a second claim."""
    slots = max(note_count, 1)
    return {
        "claims": slots * _CLAIMS_PER_DOCUMENT,
        "limitations": slots,
        "open_questions": slots,
    }


def _gap_line() -> dict:
    return {"text": _NOT_FOUND, "citation": "", "missing": True}


def fill_note_gaps(notes: list[dict]) -> list[dict]:
    """Keep every note and mark fields that had no quotable sentence."""
    filled: list[dict] = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        block = {
            "title": note.get("title") or "",
            "cite_id": note.get("cite_id") or "",
            "claims": list(note.get("claims") or []),
            "limitations": list(note.get("limitations") or []),
            "open_questions": list(note.get("open_questions") or []),
        }
        if not block["claims"]:
            block["claims"] = [_gap_line()]
        if not block["limitations"]:
            block["limitations"] = [_gap_line()]
        if not block["open_questions"]:
            block["open_questions"] = [_gap_line()]
        filled.append(block)
    return filled


def _cite_doc(citation: str) -> str:
    token = citation.strip().split()
    if not token:
        return ""
    return token[0].lstrip("[")


def _note_blocks(
    buckets: list[dict],
    claims: list[dict],
    limitations: list[dict],
    questions: list[str],
) -> list[dict]:
    """Group selected findings under each note, in retrieval order."""
    blocks: dict[str, dict] = {}
    order: list[str] = []
    for bucket in buckets:
        doc = bucket["doc"]
        if doc in blocks:
            continue
        order.append(doc)
        blocks[doc] = {
            "title": bucket["title"],
            "cite_id": doc,
            "claims": [],
            "limitations": [],
            "open_questions": [],
        }
    for item in claims:
        block = blocks.get(_cite_doc(str(item.get("citation") or "")))
        if block is not None:
            block["claims"].append(item)
    for item in limitations:
        block = blocks.get(_cite_doc(str(item.get("citation") or "")))
        if block is not None:
            block["limitations"].append(item)
    for question in questions:
        match = re.search(r"\[([A-Za-z0-9_-]+)\s+p\.\d+\]", question)
        if match is None:
            continue
        block = blocks.get(match.group(1))
        if block is not None:
            block["open_questions"].append(question)
    return [blocks[doc] for doc in order]


def _supported_content(payload: dict) -> bool:
    """Return whether grounding left any cited content."""
    if any(payload.get(field) for field in ("summary", "decision", "answer")):
        return True
    for key in ("claims", "limitations", "risks", "open_questions"):
        if payload.get(key):
            return True
    for aspect in payload.get("aspects") or []:
        if isinstance(aspect, dict) and aspect.get("observations"):
            return True
    return False


def _coverage(
    answer: dict,
    draft: dict,
    chunks: list[dict],
    notes_in_collection: int,
) -> dict:
    """Notes checked against the collection, and claims kept or removed."""
    notes = [note for note in answer.get("notes") or [] if isinstance(note, dict)]
    if notes:
        checked = len(notes)
    else:
        checked = len({_document_key(chunk) for chunk in chunks if _document_key(chunk)})
    grounded = [
        item
        for item in answer.get("claims") or []
        if isinstance(item, dict) and not item.get("missing")
    ]
    drafted = [
        (item.get("text"), item.get("citation"))
        for item in draft.get("claims") or []
        if isinstance(item, dict) and not item.get("missing")
    ]
    remaining = [(item.get("text"), item.get("citation")) for item in grounded]
    dropped = 0
    for key in drafted:
        if key in remaining:
            remaining.remove(key)
        else:
            dropped += 1
    return {
        "notes_checked": checked,
        "notes_in_collection": notes_in_collection,
        "claims_grounded": len(grounded),
        "claims_dropped": dropped,
    }


def run_agent(
    question: str,
    mode: str,
    collection_id: str,
    document_ids: list[str] | None,
    generate: GenerateFn,
    search: SearchFn,
) -> dict:
    """Execute one grounded Helix run."""
    graph = build_graph(generate, search)
    final = graph.invoke({
        "question": question,
        "mode": mode,
        "collection_id": collection_id,
        "document_ids": document_ids or [],
        "trajectory": [],
    })
    answer = final.get("grounded") or final.get("draft") or {}
    return {
        "question": question,
        "mode": mode,
        "answer": answer,
        "faithfulness": final.get("faithfulness", 0.0),
        "coverage": _coverage(
            answer,
            final.get("draft") or {},
            final.get("chunks") or [],
            int(final.get("notes_in_collection") or 0),
        ),
        "dropped": final.get("dropped") or [],
        "chunks": [
            {
                "cite": chunk["cite"],
                "cite_id": chunk["cite_id"],
                "page": chunk["page"],
                "section": chunk.get("section"),
                "text": chunk["text"],
                "score": chunk.get("score"),
            }
            for chunk in final.get("chunks") or []
        ],
        "trajectory": final.get("trajectory") or [],
        "error": final.get("error", ""),
    }
