"""Original technical notes used as Helix's seed corpus.

Every gold sentence is written into the page text so citations can be checked
by exact containment. Held-out ids never enter training.
"""

from __future__ import annotations

from dataclasses import dataclass


EVAL_IDS = frozenset({
    "HX-015",
    "HX-016",
    "HX-017",
    "HX-018",
    "HX-019",
    "HX-020",
})


@dataclass(frozen=True)
class Risk:
    statement: str
    severity: str
    action: str


@dataclass(frozen=True)
class Query:
    text: str
    span: str


@dataclass(frozen=True)
class Page:
    page: int
    section: str
    text: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    domain: str
    pages: tuple[Page, ...]
    claims: tuple[str, ...]
    limitations: tuple[str, ...]
    open_questions: tuple[str, ...]
    risks: tuple[Risk, ...]
    decision: str
    method: str
    result: str
    queries: tuple[Query, ...]

    @property
    def split(self) -> str:
        return "eval" if self.doc_id in EVAL_IDS else "train"

    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    def chunks(self) -> list[dict]:
        return [
            {
                "cite_id": self.doc_id,
                "page": page.page,
                "section": page.section,
                "text": page.text,
                "cite": f"[{self.doc_id} p.{page.page}]",
                "title": self.title,
            }
            for page in self.pages
        ]


def _locate(pages: tuple[Page, ...], sentence: str) -> str:
    for page in pages:
        if sentence in page.text:
            return f"p.{page.page}"
    raise ValueError(f"Sentence not in document: {sentence[:80]}")


def _render(raw: dict) -> Document:
    risks = tuple(Risk(**item) for item in raw["risks"])
    risk_text = " ".join(
        f"Severity {risk.severity}: {risk.statement} "
        f"Recommended action: {risk.action}"
        for risk in risks
    )
    claims = " ".join(raw["claims"])
    limits = " ".join(raw["limitations"])
    questions = " ".join(raw["open_questions"])
    style = raw["style"]
    title = raw["title"]
    if style == "memo":
        page1 = (
            f"Memorandum on {title}. {raw['lede']} {claims} "
            f"Operator decision: {raw['decision']}"
        )
        page2 = (
            f"Method note: {raw['method']} Reported result: {raw['result']} "
            f"Limits of the evidence: {limits} Open questions: {questions} "
            f"Risk review: {risk_text}"
        )
        sections = ("Summary", "Evidence")
    elif style == "note":
        page1 = (
            f"Lab note for {title}. {raw['lede']} The observations follow. "
            f"{claims} The working decision is: {raw['decision']}"
        )
        page2 = (
            f"How it was measured: {raw['method']} What came back: {raw['result']} "
            f"What this does not show: {limits} Still unresolved: {questions} "
            f"Operational risks: {risk_text}"
        )
        sections = ("Observations", "Measurement")
    elif style == "advisory":
        page1 = (
            f"Advisory: {title}. {raw['lede']} Findings: {claims} "
            f"Recommended posture: {raw['decision']}"
        )
        page2 = (
            f"Assessment method: {raw['method']} Observed outcome: {raw['result']} "
            f"Boundaries: {limits} Unresolved items: {questions} "
            f"Exposure: {risk_text}"
        )
        sections = ("Advisory", "Assessment")
    else:
        page1 = (
            f"Minutes for {title}. {raw['lede']} The group recorded these points. "
            f"{claims} Decision recorded: {raw['decision']}"
        )
        page2 = (
            f"Evaluation setup discussed: {raw['method']} "
            f"Figures entered in the minutes: {raw['result']} "
            f"Caveats entered: {limits} Questions parked: {questions} "
            f"Risks assigned: {risk_text}"
        )
        sections = ("Minutes", "Recorded detail")
    pages = (
        Page(1, sections[0], page1),
        Page(2, sections[1], page2),
    )
    _locate(pages, raw["decision"])
    for sentence in (
        raw["claims"]
        + raw["limitations"]
        + raw["open_questions"]
        + [raw["method"], raw["result"]]
    ):
        _locate(pages, sentence)
    for risk in risks:
        _locate(pages, risk.statement)
        _locate(pages, risk.action)
    queries = tuple(Query(item["q"], item["span"]) for item in raw["queries"])
    for query in queries:
        _locate(pages, query.span)
    return Document(
        doc_id=raw["doc_id"],
        title=title,
        domain=raw["domain"],
        pages=pages,
        claims=tuple(raw["claims"]),
        limitations=tuple(raw["limitations"]),
        open_questions=tuple(raw["open_questions"]),
        risks=risks,
        decision=raw["decision"],
        method=raw["method"],
        result=raw["result"],
        queries=queries,
    )


def _q(text: str, span: str) -> dict:
    return {"q": text, "span": span}


_RAW: list[dict] = [
    {
        "doc_id": "HX-001",
        "title": "Northline Cache for repeated lookups",
        "domain": "retrieval",
        "style": "memo",
        "lede": (
            "Analysts were waiting on repeated semantic lookups against a "
            "fixed desk corpus."
        ),
        "claims": [
            "Northline Cache reduced p95 lookup latency from 180 ms to 74 ms by retaining embeddings for repeated queries.",
            "The March replay showed a 2.1 point gain in recall at five after the cache warmed.",
        ],
        "limitations": [
            "The cache measurement used one corpus snapshot and did not cover streaming updates.",
        ],
        "open_questions": [
            "Does the latency gain hold when more than one analyst shares the same warm store?",
        ],
        "risks": [{
            "statement": "A stale cached vector can surface a superseded paragraph after a document is replaced.",
            "severity": "medium",
            "action": "Invalidate cache entries whenever a source file hash changes.",
        }],
        "decision": "Keep the warm store enabled for the analyst desk and bypass it for one-off research queries.",
        "method": "Latency was measured on 400 repeated queries against a frozen 12,000-chunk corpus, with a cold run before each cached run.",
        "result": "Median latency fell from 96 ms to 41 ms and recall at five rose from 0.71 to 0.73.",
        "queries": [
            _q(
                "Which write-up credits a warm store of reused vectors with making slow lookups about twice as quick?",
                "Northline Cache reduced p95 lookup latency from 180 ms to 74 ms by retaining embeddings for repeated queries.",
            ),
            _q(
                "Where is a frozen-corpus replay described as lifting fifth-rank recovery by roughly two points once the store was hot?",
                "The March replay showed a 2.1 point gain in recall at five after the cache warmed.",
            ),
            _q(
                "Which note says the timing trial never exercised live document arrivals?",
                "The cache measurement used one corpus snapshot and did not cover streaming updates.",
            ),
            _q(
                "What should happen to stored vectors after a file digest changes?",
                "Invalidate cache entries whenever a source file hash changes.",
            ),
        ],
    },
    {
        "doc_id": "HX-002",
        "title": "Bramble hybrid fusion",
        "domain": "retrieval",
        "style": "note",
        "lede": "Keyword search and dense search were missing different passages.",
        "claims": [
            "Bramble fusion with reciprocal rank fusion at k equals 60 beat either retriever alone on the held-out question set.",
            "Dense search recovered paraphrases while keyword search recovered rare identifiers such as ticket codes.",
        ],
        "limitations": [
            "The fusion trial did not tune k per domain, so the constant 60 may be suboptimal for very short notes.",
        ],
        "open_questions": [
            "Would a learned gate outperform a fixed fusion constant on mixed-length collections?",
        ],
        "risks": [{
            "statement": "Fusion can promote a keyword-only hit that is lexically similar but topically wrong.",
            "severity": "medium",
            "action": "Show both retriever ranks beside the fused list during review.",
        }],
        "decision": "Ship reciprocal rank fusion as the default desk search and keep the two raw lists available to reviewers.",
        "method": "Each question was scored with keyword search alone, dense search alone, and fused ranks, using the same top-20 cutoff.",
        "result": "Fused recall at ten was 0.84 compared with 0.76 for dense search and 0.71 for keyword search.",
        "queries": [
            _q(
                "Which lab note found that blending the two rankers with a fixed reciprocal constant beat running either ranker by itself?",
                "Bramble fusion with reciprocal rank fusion at k equals 60 beat either retriever alone on the held-out question set.",
            ),
            _q(
                "Where are ticket-style codes said to be a keyword strength while reworded questions favor the vector side?",
                "Dense search recovered paraphrases while keyword search recovered rare identifiers such as ticket codes.",
            ),
            _q(
                "Which measurement admits the fusion constant was never fit separately for each subject area?",
                "The fusion trial did not tune k per domain, so the constant 60 may be suboptimal for very short notes.",
            ),
            _q(
                "What review practice is urged so a purely lexical false friend stays visible?",
                "Show both retriever ranks beside the fused list during review.",
            ),
        ],
    },
    {
        "doc_id": "HX-003",
        "title": "Citation Latch faithfulness check",
        "domain": "grounding",
        "style": "memo",
        "lede": "Briefs were shipping sentences that no source page contained.",
        "claims": [
            "Citation Latch drops any sentence whose content words are not mostly present in the cited page.",
            "On the June audit, the checker cut unsupported sentences from 18 percent to 4 percent.",
        ],
        "limitations": [
            "The overlap rule misses contradictions that reuse the source vocabulary with a flipped conclusion.",
        ],
        "open_questions": [
            "Can a small entailment model catch reversals without adding more than 200 ms per brief?",
        ],
        "risks": [{
            "statement": "A strict overlap threshold will delete valid paraphrases that analysts would have accepted.",
            "severity": "low",
            "action": "Keep rejected sentences in an audit tray rather than discarding them silently.",
        }],
        "decision": "Run the overlap check before every brief is shown, and surface rejected lines to the analyst.",
        "method": "Two annotators marked unsupported sentences in 50 briefs, and the checker was scored against their union.",
        "result": "Precision of the rejection decision was 0.91 and recall of unsupported lines was 0.78.",
        "queries": [
            _q(
                "Which memorandum describes a gate that throws out lines when the cited page lacks most of their content words?",
                "Citation Latch drops any sentence whose content words are not mostly present in the cited page.",
            ),
            _q(
                "Where did an audit show unsupported lines falling from just under a fifth to about one in twenty-five?",
                "On the June audit, the checker cut unsupported sentences from 18 percent to 4 percent.",
            ),
            _q(
                "Which check is described as blind to a conclusion that flips while keeping the original wording?",
                "The overlap rule misses contradictions that reuse the source vocabulary with a flipped conclusion.",
            ),
            _q(
                "What should happen to lines the gate refuses, instead of deleting them with no trace?",
                "Keep rejected sentences in an audit tray rather than discarding them silently.",
            ),
        ],
    },
    {
        "doc_id": "HX-004",
        "title": "Retry Loom for tool failures",
        "domain": "agents",
        "style": "note",
        "lede": "Multi-step runs were aborting when a single retrieval call timed out.",
        "claims": [
            "Retry Loom repeats a failed tool call once with a narrower query before it returns a partial brief.",
            "Capped retries cut abandoned runs from 11 percent to 3 percent on the operations queue.",
        ],
        "limitations": [
            "The retry policy was only exercised on retrieval timeouts, not on malformed model outputs.",
        ],
        "open_questions": [
            "Should a second failure switch models or simply stop with the evidence collected so far?",
        ],
        "risks": [{
            "statement": "Unbounded retries can loop on a tool that fails for a permanent reason such as a missing file.",
            "severity": "high",
            "action": "Allow at most one retry and record the tool error in the trajectory.",
        }],
        "decision": "Adopt a single retry with a narrower query, then finish with whatever evidence is already in hand.",
        "method": "The queue replay compared a no-retry agent with a one-retry agent on 300 sessions that included injected timeouts.",
        "result": "Completed sessions rose from 89 percent to 97 percent and median added delay was 1.4 seconds.",
        "queries": [
            _q(
                "Which lab note says a failed tool is attempted again a single time with a tighter search before the run settles for a partial product?",
                "Retry Loom repeats a failed tool call once with a narrower query before it returns a partial brief.",
            ),
            _q(
                "Where did giving up less often on the operations queue drop abandoned work from about one in nine to about one in thirty?",
                "Capped retries cut abandoned runs from 11 percent to 3 percent on the operations queue.",
            ),
            _q(
                "Which policy write-up never tried the recovery path on badly shaped model text?",
                "The retry policy was only exercised on retrieval timeouts, not on malformed model outputs.",
            ),
            _q(
                "What cap prevents a permanently broken tool from being hammered?",
                "Allow at most one retry and record the tool error in the trajectory.",
            ),
        ],
    },
    {
        "doc_id": "HX-005",
        "title": "Scanfold table recovery",
        "domain": "documents",
        "style": "advisory",
        "lede": "Scanned quarterly packets hid the figures analysts needed inside ruled grids.",
        "claims": [
            "Scanfold recovers ruled grids by trusting vector lines when they exist and falling back to text alignment when they do not.",
            "On born-digital pages, cell accuracy reached 96 percent, but on phone photos it fell to 71 percent.",
        ],
        "limitations": [
            "Merged header cells were frequently split into extra columns on the phone-photo subset.",
        ],
        "open_questions": [
            "Would a lightweight detector trained on ruling lines close the gap between digital pages and phone photos?",
        ],
        "risks": [{
            "statement": "A misread total in a scanned budget table can flow into a decision brief without an obvious visual cue.",
            "severity": "high",
            "action": "Mark machine-read tables as unverified until an analyst confirms the header row.",
        }],
        "decision": "Use automatic table text for digital PDFs and require analyst confirmation for photo pages.",
        "method": "Cell accuracy was scored against 40 hand-transcribed tables, split evenly between digital PDFs and phone photos.",
        "result": "Digital cell accuracy was 0.96 and photo cell accuracy was 0.71, with header merges causing most photo errors.",
        "queries": [
            _q(
                "Which advisory recovers grids from drawing commands when present and from aligned text otherwise?",
                "Scanfold recovers ruled grids by trusting vector lines when they exist and falling back to text alignment when they do not.",
            ),
            _q(
                "Where is accuracy on camera images far below the accuracy on originally digital pages?",
                "On born-digital pages, cell accuracy reached 96 percent, but on phone photos it fell to 71 percent.",
            ),
            _q(
                "Which failure involves header cells that span several columns being broken apart?",
                "Merged header cells were frequently split into extra columns on the phone-photo subset.",
            ),
            _q(
                "What confirmation is required before a photographed grid is treated as trusted?",
                "Mark machine-read tables as unverified until an analyst confirms the header row.",
            ),
        ],
    },
    {
        "doc_id": "HX-006",
        "title": "Weekglass digest",
        "domain": "monitoring",
        "style": "memo",
        "lede": "Topic watches were producing long dumps instead of a Monday brief.",
        "claims": [
            "Weekglass keeps only items that are new since the previous digest or that change a prior severity.",
            "Analysts rated the one-page Monday brief as saving about 25 minutes versus reading the raw queue.",
        ],
        "limitations": [
            "The time-saved figure comes from self-report, not from a timed task study.",
        ],
        "open_questions": [
            "How often should a quiet topic still emit a digest that explicitly says nothing material changed?",
        ],
        "risks": [{
            "statement": "Suppressing unchanged items can hide a risk that remains severe but is no longer new.",
            "severity": "medium",
            "action": "Always retain items whose severity is high even when the text is unchanged.",
        }],
        "decision": "Publish a Monday digest limited to new items, severity changes, and any standing high-severity risks.",
        "method": "Four analysts scored five digest formats on usefulness after their normal Monday review.",
        "result": "The one-page format received a median usefulness of 4.5 out of 5, against 3.0 for the full queue paste.",
        "queries": [
            _q(
                "Which memorandum filters a watchlist down to fresh items and to cases where seriousness changed?",
                "Weekglass keeps only items that are new since the previous digest or that change a prior severity.",
            ),
            _q(
                "Where do readers credit a short start-of-week note with avoiding about half an hour of queue reading?",
                "Analysts rated the one-page Monday brief as saving about 25 minutes versus reading the raw queue.",
            ),
            _q(
                "Which savings claim rests on what people said rather than a clocked exercise?",
                "The time-saved figure comes from self-report, not from a timed task study.",
            ),
            _q(
                "What must stay visible even when the wording did not change, if the seriousness is already at the top rung?",
                "Always retain items whose severity is high even when the text is unchanged.",
            ),
        ],
    },
    {
        "doc_id": "HX-007",
        "title": "LoRA Ridge rank sweep",
        "domain": "training",
        "style": "note",
        "lede": "A rank sweep asked how much adapter capacity the brief task actually needed.",
        "claims": [
            "Raising LoRA rank from 8 to 16 improved structured field F1, while rank 32 did not improve it further.",
            "Rank 32 increased trainable parameters by about 2 times versus rank 16 with no gain on the held-out briefs.",
        ],
        "limitations": [
            "The sweep used one base model and one task mix, so the plateau may sit elsewhere for other bases.",
        ],
        "open_questions": [
            "Does a larger rank help only when the training mixture includes comparison as well as extraction?",
        ],
        "risks": [{
            "statement": "Training a higher rank on a small corpus can memorize citation strings from the training notes.",
            "severity": "medium",
            "action": "Keep entire documents, not individual examples, in the held-out split.",
        }],
        "decision": "Use rank 16 for the brief adapter unless a later mixture shows a clear gain from more capacity.",
        "method": "Ranks 8, 16, and 32 were trained for one epoch on the same examples and scored on documents withheld by identifier.",
        "result": "Structured field F1 was 0.62 at rank 8, 0.74 at rank 16, and 0.73 at rank 32.",
        "queries": [
            _q(
                "Which lab note saw field overlap improve when adapter capacity doubled from the smallest setting, then stall at the next doubling?",
                "Raising LoRA rank from 8 to 16 improved structured field F1, while rank 32 did not improve it further.",
            ),
            _q(
                "Where does extra capacity fail to help withheld briefs even though the parameter count roughly doubles again?",
                "Rank 32 increased trainable parameters by about 2 times versus rank 16 with no gain on the held-out briefs.",
            ),
            _q(
                "Which sweep never tried a second foundation model?",
                "The sweep used one base model and one task mix, so the plateau may sit elsewhere for other bases.",
            ),
            _q(
                "What split rule is meant to stop an adapter from memorizing source identifiers?",
                "Keep entire documents, not individual examples, in the held-out split.",
            ),
        ],
    },
    {
        "doc_id": "HX-008",
        "title": "Figure Drift diagram notes",
        "domain": "documents",
        "style": "advisory",
        "lede": "Architecture sketches were being described from captions alone.",
        "claims": [
            "Figure Drift treats a caption as a claim about the drawing only when the caption names the same components as the nearby text.",
            "Uncaptioned boxes were left as unresolved rather than assigned a guessed function.",
        ],
        "limitations": [
            "The rule does not read pixels, so a diagram with no nearby text remains opaque.",
        ],
        "open_questions": [
            "Which vision encoder is worth the latency if it is only applied to pages that contain drawings?",
        ],
        "risks": [{
            "statement": "Guessing the role of an unlabeled box can invent an architecture the paper never proposed.",
            "severity": "high",
            "action": "Refuse to name a component that is absent from both the caption and the body text.",
        }],
        "decision": "Describe drawings only from captions and body text that agree, and mark everything else unresolved.",
        "method": "A reviewer checked 30 diagram descriptions for invented components before and after the agreement rule.",
        "result": "Invented components fell from 12 descriptions to 2 descriptions out of 30.",
        "queries": [
            _q(
                "Which advisory trusts a caption's account of a sketch only when the same parts are named in the surrounding prose?",
                "Figure Drift treats a caption as a claim about the drawing only when the caption names the same components as the nearby text.",
            ),
            _q(
                "Where are unlabeled rectangles left unnamed instead of given a speculated job?",
                "Uncaptioned boxes were left as unresolved rather than assigned a guessed function.",
            ),
            _q(
                "Which rule openly cannot interpret a sketch that has no accompanying words?",
                "The rule does not read pixels, so a diagram with no nearby text remains opaque.",
            ),
            _q(
                "What must the system decline to do when a part appears in neither caption nor body?",
                "Refuse to name a component that is absent from both the caption and the body text.",
            ),
        ],
    },
    {
        "doc_id": "HX-009",
        "title": "Pairwise Ledger comparison protocol",
        "domain": "evaluation",
        "style": "minutes",
        "lede": "Side-by-side questions were collapsing into a single blended summary.",
        "claims": [
            "Pairwise Ledger requires every comparison aspect to cite each document separately.",
            "Blended summaries were rejected when a result from one paper was attributed to both.",
        ],
        "limitations": [
            "The protocol was tried on pairs, not on sets of four or more documents.",
        ],
        "open_questions": [
            "How should an aspect be shown when only one of the two documents reports it?",
        ],
        "risks": [{
            "statement": "A shared aspect name can imply the two studies measured the same quantity when they did not.",
            "severity": "medium",
            "action": "Quote each study's own result sentence under the aspect instead of rewriting a joint figure.",
        }],
        "decision": "Require separate citations for each document on methodology and on reported results.",
        "method": "Reviewers scored 20 pair answers for crossed attribution before and after the separate-citation rule.",
        "result": "Crossed attributions fell from 9 answers to 1 answer out of 20.",
        "queries": [
            _q(
                "Which minutes require each side of a contrast to carry its own source pointer?",
                "Pairwise Ledger requires every comparison aspect to cite each document separately.",
            ),
            _q(
                "Where is a merged write-up refused if a number from one paper is hung on both papers?",
                "Blended summaries were rejected when a result from one paper was attributed to both.",
            ),
            _q(
                "Which protocol never left the two-document case?",
                "The protocol was tried on pairs, not on sets of four or more documents.",
            ),
            _q(
                "What quoting practice avoids inventing a single combined figure?",
                "Quote each study's own result sentence under the aspect instead of rewriting a joint figure.",
            ),
        ],
    },
    {
        "doc_id": "HX-010",
        "title": "Severity Ladder for advisories",
        "domain": "security",
        "style": "advisory",
        "lede": "Incident notes used incompatible words for how urgent a fix was.",
        "claims": [
            "Severity Ladder maps outage of a control path to high, degraded visibility to medium, and documentation drift to low.",
            "Using the ladder cut disagreements between the two duty officers from 14 cases to 3 cases in April.",
        ],
        "limitations": [
            "The ladder does not score vulnerabilities that have no operational impact note.",
        ],
        "open_questions": [
            "Should customer-visible delay without an outage sit at medium or at high?",
        ],
        "risks": [{
            "statement": "Copying a vendor's own urgency word can bypass the ladder and reintroduce inconsistent labels.",
            "severity": "medium",
            "action": "Record the ladder label next to any vendor wording and use the ladder label in briefs.",
        }],
        "decision": "Label operational notes with the ladder and keep vendor wording only as a quotation.",
        "method": "Two duty officers labeled 40 historical notes independently, then repeated the task with the ladder definitions in view.",
        "result": "Agreement rose from 65 percent to 92 percent after the ladder was introduced.",
        "queries": [
            _q(
                "Which advisory assigns a top label to a broken control path, a middle label to weaker visibility, and a bottom label to paperwork drift?",
                "Severity Ladder maps outage of a control path to high, degraded visibility to medium, and documentation drift to low.",
            ),
            _q(
                "Where did a shared scale shrink officer disputes during April from fourteen files to three?",
                "Using the ladder cut disagreements between the two duty officers from 14 cases to 3 cases in April.",
            ),
            _q(
                "Which scale refuses to grade a weakness that lacks an operations-impact note?",
                "The ladder does not score vulnerabilities that have no operational impact note.",
            ),
            _q(
                "How should a supplier's own urgency word be kept from replacing the internal label?",
                "Record the ladder label next to any vendor wording and use the ladder label in briefs.",
            ),
        ],
    },
    {
        "doc_id": "HX-011",
        "title": "Chunk Window length ablation",
        "domain": "retrieval",
        "style": "note",
        "lede": "Passage length was changed while the embedding model stayed fixed.",
        "claims": [
            "Chunks near 900 characters beat both 300-character slices and 2,000-character slices on recall at five.",
            "Very short slices lost the sentence that actually stated the result.",
        ],
        "limitations": [
            "The ablation used English technical prose and may not transfer to slide decks with sparse text.",
        ],
        "open_questions": [
            "Does the best length change when tables are stored as separate chunks?",
        ],
        "risks": [{
            "statement": "A long chunk can bury a relevant sentence under unrelated paragraphs from the same section.",
            "severity": "low",
            "action": "Prefer section-bounded chunks before increasing the character budget.",
        }],
        "decision": "Index narrative pages in roughly 900-character section-bounded chunks.",
        "method": "The same 80 questions were retrieved three times, changing only the chunker length and overlap.",
        "result": "Recall at five was 0.61 at 300 characters, 0.78 at 900 characters, and 0.70 at 2,000 characters.",
        "queries": [
            _q(
                "Which lab note found a mid-sized passage length winning over both much shorter and much longer cuts?",
                "Chunks near 900 characters beat both 300-character slices and 2,000-character slices on recall at five.",
            ),
            _q(
                "Where did tiny slices drop the sentence that carried the outcome?",
                "Very short slices lost the sentence that actually stated the result.",
            ),
            _q(
                "Which length test never left English technical prose?",
                "The ablation used English technical prose and may not transfer to slide decks with sparse text.",
            ),
            _q(
                "What boundary should be respected before a passage is allowed to grow?",
                "Prefer section-bounded chunks before increasing the character budget.",
            ),
        ],
    },
    {
        "doc_id": "HX-012",
        "title": "Topic Drift Watch",
        "domain": "monitoring",
        "style": "memo",
        "lede": "A standing watch on agent reliability started collecting unrelated outages.",
        "claims": [
            "Topic Drift Watch flags a digest item when its embedding similarity to the topic seed falls below 0.42.",
            "The threshold caught 17 off-topic items in May while missing 2 items the editor still wanted.",
        ],
        "limitations": [
            "The seed was a single paragraph, so a topic that legitimately shifts vocabulary will look like drift.",
        ],
        "open_questions": [
            "Should the seed be updated automatically after an editor accepts a borderline item?",
        ],
        "risks": [{
            "statement": "A fixed threshold can drop a genuine development that uses new terminology.",
            "severity": "medium",
            "action": "Queue borderline items for a quick editor decision instead of auto-dropping them.",
        }],
        "decision": "Keep the similarity gate, and send borderline items to an editor rather than suppressing them.",
        "method": "An editor labeled May digest candidates as on-topic or off-topic, and the gate was compared with those labels.",
        "result": "Off-topic precision was 0.94 and on-topic recall was 0.90 at the 0.42 cutoff.",
        "queries": [
            _q(
                "Which memorandum marks a digest line as wandering when its closeness to the seed paragraph drops under a fixed cutoff near two-fifths?",
                "Topic Drift Watch flags a digest item when its embedding similarity to the topic seed falls below 0.42.",
            ),
            _q(
                "Where did May's gate remove seventeen wanderers and also skip two lines an editor still valued?",
                "The threshold caught 17 off-topic items in May while missing 2 items the editor still wanted.",
            ),
            _q(
                "Which gate is brittle because the reference text is only one paragraph?",
                "The seed was a single paragraph, so a topic that legitimately shifts vocabulary will look like drift.",
            ),
            _q(
                "What should happen to near-miss items instead of automatic removal?",
                "Queue borderline items for a quick editor decision instead of auto-dropping them.",
            ),
        ],
    },
    {
        "doc_id": "HX-013",
        "title": "Schema Kiln output contract",
        "domain": "training",
        "style": "minutes",
        "lede": "Downstream tools were breaking when brief keys changed without notice.",
        "claims": [
            "Schema Kiln freezes the brief keys for a release and rejects model output that omits a required key.",
            "After the freeze, parsing failures in the desk tool fell from 9 percent to under 1 percent.",
        ],
        "limitations": [
            "The contract check does not judge whether the values are faithful, only whether the keys exist.",
        ],
        "open_questions": [
            "How should optional fields be versioned without forcing every client to update on the same day?",
        ],
        "risks": [{
            "statement": "A model can satisfy the key check with empty lists and still look structurally valid.",
            "severity": "low",
            "action": "Fail a brief that has no claims when the sources are non-empty.",
        }],
        "decision": "Validate the frozen keys before display, and reject empty claim lists when sources were provided.",
        "method": "Client logs were compared for the two weeks before and after the key check was deployed.",
        "result": "Parse errors dropped from 9.1 percent of briefs to 0.7 percent of briefs.",
        "queries": [
            _q(
                "Which minutes lock the field names for a release and refuse payloads that leave out a required field?",
                "Schema Kiln freezes the brief keys for a release and rejects model output that omits a required key.",
            ),
            _q(
                "Where did desk-tool parse breakage nearly disappear after that lock?",
                "After the freeze, parsing failures in the desk tool fell from 9 percent to under 1 percent.",
            ),
            _q(
                "Which contract ignores whether the contents are true and only looks at the presence of fields?",
                "The contract check does not judge whether the values are faithful, only whether the keys exist.",
            ),
            _q(
                "When should a structurally neat brief still be refused if the inputs were not blank?",
                "Fail a brief that has no claims when the sources are non-empty.",
            ),
        ],
    },
    {
        "doc_id": "HX-014",
        "title": "Action Tether for next steps",
        "domain": "grounding",
        "style": "memo",
        "lede": "Suggested next steps were sometimes generic advice with no source.",
        "claims": [
            "Action Tether keeps a suggested action only when that action sentence appears in a cited source.",
            "Generic advice such as monitor the situation was removed unless a source used that instruction.",
        ],
        "limitations": [
            "The tether cannot propose a novel action that is reasonable but unstated.",
        ],
        "open_questions": [
            "Should analysts be able to pin an unstated action and have it stored as their own note?",
        ],
        "risks": [{
            "statement": "Dropping every unstated action can leave a severe risk with no next step in the brief.",
            "severity": "medium",
            "action": "If a high-severity risk has no sourced action, ask the analyst to supply one explicitly.",
        }],
        "decision": "Show only sourced actions, and request an analyst-written step when a high-severity risk has none.",
        "method": "Thirty briefs were checked for actions that did not occur in the cited pages.",
        "result": "Unsourced actions fell from 22 instances to 0 instances after the tether was applied.",
        "queries": [
            _q(
                "Which memorandum allows a next step only when the same sentence exists on the cited page?",
                "Action Tether keeps a suggested action only when that action sentence appears in a cited source.",
            ),
            _q(
                "Where is vague counsel stripped unless some source actually gave that counsel?",
                "Generic advice such as monitor the situation was removed unless a source used that instruction.",
            ),
            _q(
                "Which constraint cannot invent a sensible step that nobody wrote down?",
                "The tether cannot propose a novel action that is reasonable but unstated.",
            ),
            _q(
                "What should the analyst be asked to do if a top-rung risk has no written step?",
                "If a high-severity risk has no sourced action, ask the analyst to supply one explicitly.",
            ),
        ],
    },
    {
        "doc_id": "HX-015",
        "title": "Heading Probe versus hand rules",
        "domain": "documents",
        "style": "note",
        "lede": "Section titles were being guessed from capitalization alone.",
        "claims": [
            "Heading Probe labels a line as a heading only when it is short, has no trailing period, and is followed by a longer paragraph.",
            "The combined rule raised heading precision from 0.70 to 0.88 on the internal report set.",
        ],
        "limitations": [
            "All-capital legal clauses were still marked as headings even when they were body text.",
        ],
        "open_questions": [
            "Can a trained sequence tagger separate legal clauses from titles without a full layout model?",
        ],
        "risks": [{
            "statement": "A false heading splits a chunk in the middle of an argument and weakens retrieval.",
            "severity": "medium",
            "action": "Fall back to page-level chunks when fewer than two heading candidates exist.",
        }],
        "decision": "Use the combined heading rule, and fall back to whole pages when heading candidates are scarce.",
        "method": "Precision and recall were computed against 25 reports whose headings had been marked by an editor.",
        "result": "Heading precision moved from 0.70 to 0.88 and recall stayed near 0.81.",
        "queries": [
            _q(
                "Which lab note promotes a line to a title only if it is brief, lacks a final stop, and a longer block follows?",
                "Heading Probe labels a line as a heading only when it is short, has no trailing period, and is followed by a longer paragraph.",
            ),
            _q(
                "Where did stacking those cues lift title precision on the internal report collection?",
                "The combined rule raised heading precision from 0.70 to 0.88 on the internal report set.",
            ),
            _q(
                "Which residual error tags fully capitalized legal language as if it were a title?",
                "All-capital legal clauses were still marked as headings even when they were body text.",
            ),
            _q(
                "What backup segmentation is used when almost no title candidates appear?",
                "Fall back to page-level chunks when fewer than two heading candidates exist.",
            ),
        ],
    },
    {
        "doc_id": "HX-016",
        "title": "Judge Caliper preference checks",
        "domain": "evaluation",
        "style": "memo",
        "lede": "Automatic preference labels were drifting away from what analysts actually chose.",
        "claims": [
            "Judge Caliper treats a tuned answer as a win only when its field overlap exceeds the base answer by more than two points.",
            "Ties inside that band prevented a noisy judge from inflating the win rate.",
        ],
        "limitations": [
            "The caliper was computed on structured fields and does not capture prose clarity.",
        ],
        "open_questions": [
            "Would analysts still agree with the caliper on briefs that are equally complete but differently ordered?",
        ],
        "risks": [{
            "statement": "Counting ties as wins would overstate the value of a model that mostly matches the baseline.",
            "severity": "low",
            "action": "Report win rate both with ties excluded and with ties included in the denominator.",
        }],
        "decision": "Publish both win rates and keep a two-point margin before calling a preference.",
        "method": "Analyst choices on 40 pairs were compared with caliper labels that used a two-point field-overlap margin.",
        "result": "Agreement with analysts was 0.85 when ties were set aside and 0.78 when ties counted as non-wins.",
        "queries": [
            _q(
                "Which memorandum refuses to call a winner unless field overlap leads by more than a small margin?",
                "Judge Caliper treats a tuned answer as a win only when its field overlap exceeds the base answer by more than two points.",
            ),
            _q(
                "Where are near-equal pairs left undecided so a jittery scorer cannot pump the headline rate?",
                "Ties inside that band prevented a noisy judge from inflating the win rate.",
            ),
            _q(
                "Which score ignores how readable the prose feels?",
                "The caliper was computed on structured fields and does not capture prose clarity.",
            ),
            _q(
                "What pair of rates should be published so draws are not hidden?",
                "Report win rate both with ties excluded and with ties included in the denominator.",
            ),
        ],
    },
    {
        "doc_id": "HX-017",
        "title": "Vendor Window patch comparison",
        "domain": "security",
        "style": "advisory",
        "lede": "Two gateway vendors proposed different weekends for a signing change.",
        "claims": [
            "Northwind Gateway 4.2 rejects telemetry that fails signature checks, which created a two-hour monitoring gap in the March rehearsal.",
            "Sable Route 7.1 queues unsigned telemetry for review instead of dropping it, at the cost of a larger holding directory.",
        ],
        "limitations": [
            "Neither rehearsal measured analyst time spent clearing the queue or rebuilding the gap.",
        ],
        "open_questions": [
            "Can the two products be run in parallel for one cycle without double-counting alerts?",
        ],
        "risks": [{
            "statement": "Dropping unsigned telemetry without a queue can hide a collector outage during the change window.",
            "severity": "high",
            "action": "Confirm the trust store and keep a temporary queue before the Saturday change.",
        }],
        "decision": "Prefer a queue-during-change posture and verify the trust store before the weekend window.",
        "method": "Each vendor configuration was rehearsed once on a lab gateway with the same unsigned-sample stream.",
        "result": "Northwind produced a two-hour visibility gap while Sable retained the samples and used 18 GB of extra disk.",
        "queries": [
            _q(
                "Which advisory says a gateway build discards unchecked telemetry and once left monitors blind for a couple of hours in rehearsal?",
                "Northwind Gateway 4.2 rejects telemetry that fails signature checks, which created a two-hour monitoring gap in the March rehearsal.",
            ),
            _q(
                "Where does the other product hold unchecked telemetry for someone to inspect, while spending extra disk?",
                "Sable Route 7.1 queues unsigned telemetry for review instead of dropping it, at the cost of a larger holding directory.",
            ),
            _q(
                "Which comparison never timed how long people spent clearing leftovers?",
                "Neither rehearsal measured analyst time spent clearing the queue or rebuilding the gap.",
            ),
            _q(
                "What two preparations are urged before a weekend cutover?",
                "Confirm the trust store and keep a temporary queue before the Saturday change.",
            ),
        ],
    },
    {
        "doc_id": "HX-018",
        "title": "Incident Margin readout limits",
        "domain": "security",
        "style": "minutes",
        "lede": "After-action notes were being summarized as if they were complete causal analyses.",
        "claims": [
            "Incident Margin states that the readout explains the detection timeline and not the attacker's intent.",
            "Missing packet captures were listed as a limit on any claim about the initial access path.",
        ],
        "limitations": [
            "The minutes cover one exercise and should not be generalized to production incidents without the same logs.",
        ],
        "open_questions": [
            "Which additional log would be required before the initial access path could be described?",
        ],
        "risks": [{
            "statement": "Filling gaps in an incident timeline with plausible steps overstates what the logs show.",
            "severity": "high",
            "action": "Write unknown wherever a timeline step lacks a cited log line.",
        }],
        "decision": "Summarize only the detection timeline and mark every unsupported step as unknown.",
        "method": "The exercise readout was checked line by line against the preserved detection logs.",
        "result": "Four proposed causal steps had no log support and were relabeled unknown.",
        "queries": [
            _q(
                "Which minutes restrict the story to when detection happened and refuse to infer motive?",
                "Incident Margin states that the readout explains the detection timeline and not the attacker's intent.",
            ),
            _q(
                "Where are absent packet records treated as a bound on claims about how entry began?",
                "Missing packet captures were listed as a limit on any claim about the initial access path.",
            ),
            _q(
                "Which write-up warns against treating one drill as if it were every production case?",
                "The minutes cover one exercise and should not be generalized to production incidents without the same logs.",
            ),
            _q(
                "What word should replace a timeline step that has no cited log line?",
                "Write unknown wherever a timeline step lacks a cited log line.",
            ),
        ],
    },
    {
        "doc_id": "HX-019",
        "title": "Negative Mine hard examples",
        "domain": "retrieval",
        "style": "note",
        "lede": "Embedding training was ignoring near-miss passages that shared vocabulary.",
        "claims": [
            "Negative Mine adds same-domain passages with different numbers as explicit hard negatives.",
            "Adding those negatives lifted recall at three on paraphrased questions by four points.",
        ],
        "limitations": [
            "The mined negatives were written by hand, so the gain may shrink on automatically mined sets.",
        ],
        "open_questions": [
            "How many hard negatives per query are useful before the batch becomes mostly near misses?",
        ],
        "risks": [{
            "statement": "A hard negative that is actually relevant will push the model away from a correct page.",
            "severity": "medium",
            "action": "Drop any negative that contains the answer sentence before training.",
        }],
        "decision": "Train with one checked hard negative per paraphrased query and discard negatives that contain the answer.",
        "method": "Recall at three was measured on a paraphrased question set before and after one epoch with hard negatives.",
        "result": "Recall at three moved from 0.64 to 0.68 on the paraphrased set.",
        "queries": [
            _q(
                "Which lab note builds difficult distractors from the same subject area by changing the figures?",
                "Negative Mine adds same-domain passages with different numbers as explicit hard negatives.",
            ),
            _q(
                "Where did those distractors improve third-rank recovery on reworded questions by about four points?",
                "Adding those negatives lifted recall at three on paraphrased questions by four points.",
            ),
            _q(
                "Which gain might not survive if the distractors were harvested automatically?",
                "The mined negatives were written by hand, so the gain may shrink on automatically mined sets.",
            ),
            _q(
                "What must be excluded from the negative pool if it already states the answer?",
                "Drop any negative that contains the answer sentence before training.",
            ),
        ],
    },
    {
        "doc_id": "HX-020",
        "title": "Patch Desk change-window brief",
        "domain": "security",
        "style": "memo",
        "lede": "Leadership asked for a single page before the weekend signing change.",
        "claims": [
            "Patch Desk recommends a temporary review queue rather than a hard drop during the first change window.",
            "The brief limits its recommendation to the lab rehearsal evidence and the duty-officer ladder.",
        ],
        "limitations": [
            "No production traffic sample was available, so load during the window is unknown.",
        ],
        "open_questions": [
            "What queue depth would force a rollback during the window?",
        ],
        "risks": [{
            "statement": "A hard drop during the first window can open a monitoring gap if signatures fail closed.",
            "severity": "high",
            "action": "Keep the temporary queue and define a rollback if the queue depth grows without bound.",
        }],
        "decision": "Enter the weekend window with a temporary queue, a confirmed trust store, and an explicit rollback trigger.",
        "method": "The brief used the lab rehearsal outcome and the severity ladder, and did not add production estimates.",
        "result": "The recommended posture was queue-during-change, with rollback if the holding directory grows without a stated cap.",
        "queries": [
            _q(
                "Which memorandum favors a holding area over an immediate discard for the first cutover weekend?",
                "Patch Desk recommends a temporary review queue rather than a hard drop during the first change window.",
            ),
            _q(
                "Where is the advice confined to a lab rehearsal and an internal urgency scale?",
                "The brief limits its recommendation to the lab rehearsal evidence and the duty-officer ladder.",
            ),
            _q(
                "Which gap is the lack of a live traffic sample?",
                "No production traffic sample was available, so load during the window is unknown.",
            ),
            _q(
                "What two controls should be in place so a failed-closed signature check cannot blind the monitors?",
                "Keep the temporary queue and define a rollback if the queue depth grows without bound.",
            ),
        ],
    },
]


COMPARISONS: list[dict] = [
    {
        "left": "HX-001",
        "right": "HX-011",
        "question": "How do these two approaches differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-002",
        "right": "HX-001",
        "question": "How do these notes differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-003",
        "right": "HX-014",
        "question": "How do these grounding checks differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-004",
        "right": "HX-012",
        "question": "How do these operational designs differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-005",
        "right": "HX-008",
        "question": "How do these document-understanding notes differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-006",
        "right": "HX-010",
        "question": "How do these desk processes differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-007",
        "right": "HX-013",
        "question": "How do these training notes differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-009",
        "right": "HX-003",
        "question": "How do these review protocols differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-015",
        "right": "HX-016",
        "question": "How do these evaluation notes differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-017",
        "right": "HX-018",
        "question": "How do these security notes differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-019",
        "right": "HX-015",
        "question": "How do these method notes differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-020",
        "right": "HX-017",
        "question": "How do these change-window notes differ on evaluation methodology and reported results?",
    },
    {
        "left": "HX-016",
        "right": "HX-020",
        "question": "How do these decision notes differ on evaluation methodology and reported results?",
    },
]


def documents() -> list[Document]:
    """Return the validated seed library."""
    return [_render(raw) for raw in _RAW]


def validate_library() -> None:
    """Raise if gold strings, splits, or query overlap checks fail."""
    docs = documents()
    ids = [doc.doc_id for doc in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document ids")
    if not EVAL_IDS.issubset(set(ids)):
        raise ValueError("Missing held-out documents")
    by_id = {doc.doc_id: doc for doc in docs}
    for pair in COMPARISONS:
        left = by_id[pair["left"]]
        right = by_id[pair["right"]]
        if left.split != right.split:
            raise ValueError(f"Comparison crosses splits: {pair}")
    for doc in docs:
        for query in doc.queries:
            if query.text in doc.full_text():
                raise ValueError(f"Query is copied from {doc.doc_id}")


def citation_for(doc: Document, sentence: str) -> str:
    """Return the citation token for a sentence in a document."""
    for page in doc.pages:
        if sentence in page.text:
            return f"[{doc.doc_id} p.{page.page}]"
    raise ValueError(f"{sentence[:60]} not in {doc.doc_id}")
