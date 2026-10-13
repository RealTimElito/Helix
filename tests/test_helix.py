"""Unit tests for grounding, retrieval fusion, and dataset construction."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table

from helix.agent.faithfulness import ground_payload
from helix.agent.graph import quote_payload, run_agent
from helix.corpus.library import EVAL_IDS, documents, validate_library
from helix.data.build_retrieval import build_pairs
from helix.data.build_sft import build_examples
from helix.index import store
from helix.index.search import diversify_by_document, reciprocal_rank_fusion, search_collection
from helix.ingest.parse import chunk_pages, parse_path
from helix.textutil import jaccard, parse_json_object, supported


class HelixTests(unittest.TestCase):
    def test_library_and_datasets(self) -> None:
        validate_library()
        train, heldout = build_examples()
        self.assertGreaterEqual(len(train), 2000)
        self.assertGreater(len(heldout), 40)
        train_ids = {doc_id for row in train for doc_id in row["meta"]["doc_ids"]}
        eval_ids = {doc_id for row in heldout for doc_id in row["meta"]["doc_ids"]}
        self.assertTrue(train_ids.isdisjoint(eval_ids))
        gold = json.loads(train[0]["messages"][-1]["content"])
        self.assertIn("task", gold)

    def test_retrieval_pairs_are_paraphrases(self) -> None:
        _train, eval_queries, corpus = build_pairs()
        self.assertGreater(len(corpus), 40)
        mean = sum(row["overlap"] for row in eval_queries) / len(eval_queries)
        self.assertLess(mean, 0.45)

    def test_grounding_drops_unsupported_claims(self) -> None:
        chunks = [{
            "cite_id": "HX-001",
            "page": 1,
            "text": "Northline Cache reduced p95 lookup latency from 180 ms to 74 ms.",
        }]
        payload = {
            "task": "extract",
            "claims": [
                {
                    "text": "Northline Cache reduced p95 lookup latency from 180 ms to 74 ms.",
                    "citation": "[HX-001 p.1]",
                },
                {"text": "The cache works on every language.", "citation": "[HX-001 p.1]"},
            ],
        }
        result = ground_payload(payload, chunks)
        self.assertEqual(len(result["payload"]["claims"]), 1)
        self.assertEqual(len(result["dropped"]), 1)
        self.assertTrue(supported(payload["claims"][0]["text"], chunks[0]["text"]))

    def test_agent_repairs_once(self) -> None:
        calls = {"n": 0}

        def generate(messages: list[dict], _max_tokens: int) -> str:
            calls["n"] += 1
            blob = json.dumps(messages)
            if "search_query" in blob and calls["n"] == 1:
                return json.dumps({"search_query": "latency cache"})
            if "citation check rejected" in blob:
                return json.dumps({
                    "task": "brief",
                    "summary": "Keep the warm store.",
                    "claims": [{
                        "text": "Northline Cache reduced p95 lookup latency from 180 ms to 74 ms.",
                        "citation": "[HX-001 p.1]",
                    }],
                    "limitations": [],
                    "open_questions": [],
                    "risks": [],
                    "decision": "",
                })
            return json.dumps({
                "task": "brief",
                "summary": "This summary is invented.",
                "claims": [
                    {"text": "Invented claim.", "citation": "[HX-001 p.1]"},
                    {
                        "text": "Northline Cache reduced p95 lookup latency from 180 ms to 74 ms.",
                        "citation": "[HX-001 p.1]",
                    },
                ],
                "limitations": [],
                "open_questions": [],
                "risks": [],
                "decision": "Invented decision.",
            })

        def search(_collection, _query, limit, _docs):
            return [{
                "id": "c1",
                "cite_id": "HX-001",
                "page": 1,
                "section": "Summary",
                "text": "Northline Cache reduced p95 lookup latency from 180 ms to 74 ms. Keep the warm store.",
                "cite": "[HX-001 p.1]",
                "score": 1.0,
            }][:limit]

        result = run_agent("brief the cache note", "brief", "col", ["HX-001"], generate, search)
        tools = [step["tool"] for step in result["trajectory"]]
        self.assertIn("retrieve", tools)
        self.assertIn("repair", tools)
        self.assertEqual(len([tool for tool in tools if tool == "repair"]), 1)
        self.assertTrue(result["answer"]["claims"])

    def test_agent_retries_invalid_json_once(self) -> None:
        claim = "Northline Cache reduced p95 lookup latency from 180 ms to 74 ms."

        def generate(messages: list[dict], _max_tokens: int) -> str:
            blob = json.dumps(messages)
            if "search_query" in blob:
                return json.dumps({"search_query": "latency cache"})
            if "not valid JSON" in blob:
                return json.dumps({
                    "task": "brief",
                    "summary": "",
                    "claims": [{"text": claim, "citation": "[HX-001 p.1]"}],
                    "limitations": [],
                    "open_questions": [],
                    "risks": [],
                    "decision": "",
                })
            return "this is not json"

        def search(_collection, _query, limit, _docs):
            return [{
                "id": "c1",
                "cite_id": "HX-001",
                "page": 1,
                "section": "Summary",
                "text": claim,
                "cite": "[HX-001 p.1]",
                "score": 1.0,
            }][:limit]

        result = run_agent("brief the cache note", "brief", "col", ["HX-001"], generate, search)
        tools = [step["tool"] for step in result["trajectory"]]
        self.assertEqual(len([tool for tool in tools if tool == "repair"]), 1)
        self.assertEqual(result["answer"]["claims"][0]["text"], claim)
        checks = [step for step in result["trajectory"] if step["tool"] == "cite_check"]
        self.assertEqual(checks[0]["faithfulness"], 0.0)
        self.assertEqual(checks[-1]["faithfulness"], 1.0)

        def generate_broken(messages: list[dict], _max_tokens: int) -> str:
            blob = json.dumps(messages)
            if "search_query" in blob:
                return json.dumps({"search_query": "latency cache"})
            return "still not json"

        broken = run_agent("brief the cache note", "brief", "col", ["HX-001"], generate_broken, search)
        broken_tools = [step["tool"] for step in broken["trajectory"]]
        self.assertEqual(len([tool for tool in broken_tools if tool == "repair"]), 1)
        self.assertEqual(broken["answer"]["claims"][0]["text"], claim)
        self.assertEqual(broken["answer"]["claims"][0]["citation"], "[HX-001 p.1]")
        self.assertEqual(broken["faithfulness"], 1.0)
        self.assertFalse(broken["answer"]["summary"])

    def test_empty_brief_quotes_source_sentences(self) -> None:
        limitation = (
            "The measurement does not show a gain outside this snapshot "
            "and should not be generalized."
        )
        question = "Would a second analyst erase the latency gain?"
        citation = "[HX-013 p.2]"

        def generate(messages: list[dict], _max_tokens: int) -> str:
            if any("search_query" in message.get("content", "") for message in messages):
                return json.dumps({"search_query": "cache limits"})
            return json.dumps({
                "task": "brief",
                "summary": "The cache is universally better.",
                "claims": [{
                    "text": "The cache is universally better in every language.",
                    "citation": citation,
                }],
                "limitations": [],
                "open_questions": [],
                "risks": [],
                "decision": "Ship it everywhere.",
            })

        def search(_collection, _query, limit, _docs):
            return [{
                "id": "c1",
                "cite_id": "HX-013",
                "page": 2,
                "section": "Evidence",
                "text": f"{limitation} {question}",
                "cite": citation,
                "score": 1.0,
            }][:limit]

        page = [{
            "id": "c1",
            "cite_id": "HX-013",
            "document_id": "HX-013",
            "page": 2,
            "section": "Evidence",
            "text": f"{limitation} {question}",
            "cite": citation,
            "score": 1.0,
        }]

        def fake_chunks(_collection_id, document_ids=None):
            return list(page)

        with patch("helix.index.search.get_chunks", fake_chunks):
            result = run_agent(
                "What are the key technical claims, limitations, and open questions?",
                "brief",
                "col",
                None,
                generate,
                search,
            )
        tools = [step["tool"] for step in result["trajectory"]]
        self.assertEqual(tools, ["plan", "retrieve", "brief", "cite_check"])
        self.assertNotIn("repair", tools)
        self.assertEqual(result["answer"]["limitations"][0]["text"], limitation)
        self.assertEqual(result["answer"]["limitations"][0]["citation"], citation)
        self.assertIn(question, result["answer"]["open_questions"][0])
        self.assertIn(citation, result["answer"]["open_questions"][0])
        self.assertNotIn("universally", json.dumps(result["answer"]))
        self.assertEqual(result["faithfulness"], 1.0)

        def search_empty(_collection, _query, limit, _docs):
            return []

        blank = [{
            "id": "c1",
            "cite_id": "HX-013",
            "document_id": "HX-013",
            "page": 2,
            "section": "Evidence",
            "text": "Note.",
            "cite": citation,
            "score": 1.0,
        }]

        def blank_chunks(_collection_id, document_ids=None):
            return list(blank)

        with patch("helix.index.search.get_chunks", blank_chunks):
            empty = run_agent(
                "brief the cache note", "brief", "col", None, generate, search_empty,
            )
        self.assertEqual(empty["faithfulness"], 0.0)
        self.assertFalse(empty["answer"]["claims"])
        self.assertFalse(empty["answer"]["limitations"])
        self.assertFalse(empty["answer"]["open_questions"])

    def test_brief_claim_comes_from_summary_page(self) -> None:
        finding = "Chunk Window says 900-character chunks beat shorter and longer slices."
        method = (
            "How it was measured: The same 80 questions were retrieved three times "
            "with only the chunk length changed."
        )
        reported = "What came back: Recall at five was 0.61 at the short slice."
        limitation = (
            "The measurement does not show a gain outside this snapshot "
            "and should not be generalized."
        )
        question = "Would a second analyst erase the latency gain?"
        page1 = (
            "Lab note for Chunk Window. Analysts were comparing slice lengths. "
            f"{finding}"
        )
        page2 = f"{method} {reported} {limitation} {question}"
        catalog = [
            {
                "id": "p1",
                "document_id": "doc-a",
                "cite_id": "HX-013",
                "page": 1,
                "section": "Summary",
                "text": page1,
            },
            {
                "id": "p2",
                "document_id": "doc-a",
                "cite_id": "HX-013",
                "page": 2,
                "section": "Measurement",
                "text": page2,
            },
        ]

        def generate(_messages: list[dict], _max_tokens: int) -> str:
            return "still not json"

        def search(_collection, _query, limit, _docs):
            return [{
                "id": "p2",
                "document_id": "doc-a",
                "cite_id": "HX-013",
                "page": 2,
                "section": "Measurement",
                "text": page2,
                "cite": "[HX-013 p.2]",
                "score": 1.0,
            }][:limit]

        def fake_chunks(_collection_id, document_ids=None):
            if document_ids:
                return [chunk for chunk in catalog if chunk["document_id"] in document_ids]
            return list(catalog)

        with patch("helix.index.search.get_chunks", fake_chunks):
            result = run_agent(
                "What are the key technical claims, limitations, and open questions?",
                "brief",
                "col",
                None,
                generate,
                search,
            )
        self.assertEqual({chunk["page"] for chunk in result["chunks"]}, {1, 2})
        self.assertEqual(
            [item["text"] for item in result["answer"]["claims"]],
            [finding],
        )
        self.assertEqual(result["answer"]["claims"][0]["citation"], "[HX-013 p.1]")
        claim_blob = " ".join(item["text"] for item in result["answer"]["claims"])
        self.assertNotIn("How it was measured", claim_blob)
        self.assertNotIn("What came back", claim_blob)
        self.assertNotIn("retrieved three times", claim_blob)
        self.assertNotIn("Recall at five", claim_blob)
        self.assertEqual(result["answer"]["limitations"][0]["text"], limitation)
        self.assertIn(question, result["answer"]["open_questions"][0])
        self.assertEqual(result["faithfulness"], 1.0)

    def test_brief_skips_titles_ledes_and_decisions(self) -> None:
        by_id = {doc.doc_id: doc for doc in documents()}
        chunks = []
        for doc_id in ("HX-013", "HX-011", "HX-018"):
            chunks.extend(by_id[doc_id].chunks())
        payload = quote_payload("brief", chunks)
        claim_texts = [item["text"] for item in payload["claims"]]
        limit_texts = [item["text"] for item in payload["limitations"]]
        finding = (
            "Chunks near 900 characters beat both 300-character slices and "
            "2,000-character slices on recall at five."
        )
        for banned in (
            "Minutes for Schema Kiln output contract.",
            "Downstream tools were breaking when brief keys changed without notice.",
            "Validate the frozen keys before display, and reject empty claim lists when sources were provided.",
            "Lab note for Chunk Window length ablation.",
            "The group recorded these points.",
            "The observations follow.",
        ):
            self.assertNotIn(banned, claim_texts)
        self.assertIn(finding, claim_texts)
        self.assertEqual(
            next(item["citation"] for item in payload["claims"] if item["text"] == finding),
            "[HX-011 p.1]",
        )
        self.assertNotIn("Minutes for Incident Margin readout limits.", limit_texts)
        self.assertNotIn("Minutes for Incident Margin readout limits.", claim_texts)
        claim_blob = " ".join(claim_texts)
        self.assertNotIn("How it was measured", claim_blob)
        self.assertNotIn("What came back", claim_blob)
        self.assertNotIn("Method note", claim_blob)
        self.assertNotIn("Reported result", claim_blob)
        counts: dict[str, int] = {}
        for item in payload["claims"]:
            doc_id = item["citation"].split()[0].lstrip("[")
            counts[doc_id] = counts.get(doc_id, 0) + 1
        self.assertTrue(counts)
        self.assertLessEqual(max(counts.values()), 2)
        self.assertLessEqual(len(payload["claims"]), 16)
        self.assertTrue(payload["risks"])
        self.assertTrue(payload["open_questions"])

    def test_collection_brief_covers_each_note_without_repair(self) -> None:
        order = (
            "HX-013", "HX-011", "HX-018", "HX-014",
            "HX-009", "HX-003", "HX-010", "HX-015",
        )
        by_id = {doc.doc_id: doc for doc in documents()}
        chunks = []
        for doc_id in order:
            for chunk in by_id[doc_id].chunks():
                chunks.append({
                    **chunk,
                    "id": f"{doc_id}-p{chunk['page']}",
                    "document_id": doc_id,
                    "score": 1.0,
                })
        payload = quote_payload("brief", chunks)
        claim_texts = [item["text"] for item in payload["claims"]]
        limit_texts = [item["text"] for item in payload["limitations"]]
        question_blob = " ".join(payload["open_questions"])
        self.assertIn(
            "Chunks near 900 characters beat both 300-character slices and "
            "2,000-character slices on recall at five.",
            claim_texts,
        )
        self.assertIn(
            "Blended summaries were rejected when a result from one paper "
            "was attributed to both.",
            claim_texts,
        )
        self.assertTrue(any("Citation Latch" in text for text in claim_texts))
        self.assertTrue(any("Severity Ladder" in text for text in claim_texts))
        self.assertTrue(any("Heading Probe" in text for text in claim_texts))
        for banned in (
            "Minutes for Schema Kiln output contract.",
            "Lab note for Chunk Window length ablation.",
            "Minutes for Incident Margin readout limits.",
            "Memorandum on Action Tether for next steps.",
            "Minutes for Pairwise Ledger comparison protocol.",
            "Memorandum on Citation Latch faithfulness check.",
            "Advisory: Severity Ladder for advisories.",
            "Lab note for Heading Probe versus hand rules.",
            "The group recorded these points.",
            "The observations follow.",
            "Validate the frozen keys before display, and reject empty claim lists when sources were provided.",
            "Index narrative pages in roughly 900-character section-bounded chunks.",
            "Summarize only the detection timeline and mark every unsupported step as unknown.",
            "Show only sourced actions, and request an analyst-written step when a high-severity risk has none.",
            "Require separate citations for each document on methodology and on reported results.",
            "Run the overlap check before every brief is shown, and surface rejected lines to the analyst.",
            "Label operational notes with the ladder and keep vendor wording only as a quotation.",
            "Use the combined heading rule, and fall back to whole pages when heading candidates are scarce.",
        ):
            self.assertNotIn(banned, claim_texts)
            self.assertNotIn(banned, limit_texts)
        for expected in (
            "The tether cannot propose a novel action that is reasonable but unstated.",
            "The protocol was tried on pairs, not on sets of four or more documents.",
            "The overlap rule misses contradictions that reuse the source vocabulary with a flipped conclusion.",
            "The ladder does not score vulnerabilities that have no operational impact note.",
            "All-capital legal clauses were still marked as headings even when they were body text.",
        ):
            self.assertIn(expected, limit_texts)
        for expected in (
            "How should an aspect be shown when only one of the two documents reports it?",
            "Can a small entailment model catch reversals without adding more than 200 ms per brief?",
            "Should customer-visible delay without an outage sit at medium or at high?",
            "Can a trained sequence tagger separate legal clauses from titles without a full layout model?",
        ):
            self.assertIn(expected, question_blob)
        for doc_id in order:
            self.assertTrue(
                any(doc_id in item["citation"] for item in payload["limitations"]),
                doc_id,
            )
            self.assertTrue(
                any(doc_id in item for item in payload["open_questions"]),
                doc_id,
            )
        counts: dict[str, int] = {}
        for item in payload["claims"]:
            doc_id = item["citation"].split()[0].lstrip("[")
            counts[doc_id] = counts.get(doc_id, 0) + 1
        self.assertLessEqual(max(counts.values()), 2)
        self.assertIn("HX-009", counts)

        calls = {"n": 0}

        def generate(messages: list[dict], _max_tokens: int) -> str:
            calls["n"] += 1
            blob = json.dumps(messages)
            if "search_query" in blob and calls["n"] == 1:
                return json.dumps({"search_query": "collection brief"})
            return "not json and not a grounded draft"

        def search(_collection, _query, limit, _docs):
            page_one = [chunk for chunk in chunks if chunk["page"] == 1]
            return page_one[:limit]

        def fake_chunks(_collection_id, document_ids=None):
            if document_ids:
                return [chunk for chunk in chunks if chunk["document_id"] in document_ids]
            return list(chunks)

        with patch("helix.index.search.get_chunks", fake_chunks):
            result = run_agent(
                "What are the key technical claims, limitations, and open questions?",
                "brief",
                "missing-collection",
                None,
                generate,
                search,
            )
        tools = [step["tool"] for step in result["trajectory"]]
        self.assertEqual(tools, ["plan", "retrieve", "brief", "cite_check"])
        self.assertNotIn("repair", tools)
        self.assertEqual(calls["n"], 0)
        self.assertEqual(result["faithfulness"], 1.0)
        self.assertIn(
            "Blended summaries were rejected when a result from one paper "
            "was attributed to both.",
            [item["text"] for item in result["answer"]["claims"]],
        )
        checks = [step for step in result["trajectory"] if step["tool"] == "cite_check"]
        self.assertEqual(checks[0]["faithfulness"], 1.0)
        self._assert_notes_grouped(payload)
        self._assert_notes_grouped(result["answer"])

    def test_demo_collection_brief_checks_every_note(self) -> None:
        demo = [doc for doc in documents() if doc.doc_id not in EVAL_IDS]
        self.assertEqual(len(demo), 14)
        chunks = []
        for doc in demo:
            for chunk in doc.chunks():
                chunks.append({
                    **chunk,
                    "id": f"{doc.doc_id}-p{chunk['page']}",
                    "document_id": doc.doc_id,
                    "score": 1.0,
                })
        payload = quote_payload("brief", chunks)
        notes = payload["notes"]
        cite_ids = {note["cite_id"] for note in notes}
        self.assertEqual(len(notes), 14)
        self.assertEqual(cite_ids, {doc.doc_id for doc in demo})
        self.assertTrue(cite_ids.isdisjoint(EVAL_IDS))
        claim_texts = [item["text"] for item in payload["claims"]]
        self.assertIn(
            "Chunks near 900 characters beat both 300-character slices and "
            "2,000-character slices on recall at five.",
            claim_texts,
        )
        self.assertIn(
            "Blended summaries were rejected when a result from one paper "
            "was attributed to both.",
            claim_texts,
        )
        self.assertTrue(any("Citation Latch" in text for text in claim_texts))
        for banned in (
            "Minutes for Schema Kiln output contract.",
            "Lab note for Chunk Window length ablation.",
            "Memorandum on Citation Latch faithfulness check.",
            "The group recorded these points.",
            "The observations follow.",
            "Decision recorded:",
            "The working decision is:",
            "Operator decision:",
            "Recommended posture:",
        ):
            self.assertNotIn(banned, claim_texts)
        seen: list[str] = []
        claim_docs = {item["citation"].split()[0].lstrip("[") for item in payload["claims"]}
        for item in payload["claims"]:
            doc_id = item["citation"].split()[0].lstrip("[")
            if doc_id in seen:
                self.assertEqual(set(seen), claim_docs)
                break
            seen.append(doc_id)
        for note in notes:
            self.assertTrue(note["claims"])
            self.assertTrue(note["limitations"])
            self.assertTrue(note["open_questions"])
            self.assertFalse(note["title"].lower().startswith("minutes for "))
            self.assertFalse(note["title"].lower().startswith("lab note for "))
            self.assertFalse(note["title"].lower().startswith("memorandum on "))
            self.assertFalse(note["title"].lower().startswith("advisory:"))

        def generate(_messages: list[dict], _max_tokens: int) -> str:
            raise AssertionError("collection brief should not call the model")

        def search(*_args):
            raise AssertionError("collection brief should load every page")

        def fake_chunks(_collection_id, document_ids=None):
            if document_ids:
                return [chunk for chunk in chunks if chunk["document_id"] in document_ids]
            return list(chunks)

        with patch("helix.index.search.get_chunks", fake_chunks):
            result = run_agent(
                "What are the key technical claims, limitations, and open questions?",
                "brief",
                "demo",
                None,
                generate,
                search,
            )
        self.assertEqual(
            [step["tool"] for step in result["trajectory"]],
            ["plan", "retrieve", "brief", "cite_check"],
        )
        self.assertEqual(result["faithfulness"], 1.0)
        self.assertEqual(result["coverage"]["notes_checked"], 14)
        self.assertEqual(result["coverage"]["notes_in_collection"], 14)
        self.assertEqual(
            result["coverage"]["claims_grounded"],
            len(result["answer"]["claims"]),
        )
        self.assertEqual(result["coverage"]["claims_dropped"], 0)
        self.assertEqual(
            {note["cite_id"] for note in result["answer"]["notes"]},
            cite_ids,
        )
        self.assertTrue(
            {note["cite_id"] for note in result["answer"]["notes"]}.isdisjoint(EVAL_IDS)
        )

    def test_note_without_quotable_sentences_is_kept(self) -> None:
        chunks = [{
            "id": "blank",
            "cite_id": "HX-099",
            "document_id": "HX-099",
            "page": 1,
            "section": "Summary",
            "title": "Blank Note",
            "text": (
                "Minutes for Blank Note. "
                "Decision recorded: Do nothing useful here today."
            ),
            "cite": "[HX-099 p.1]",
        }]
        payload = quote_payload("brief", chunks)
        self.assertEqual(len(payload["notes"]), 1)
        note = payload["notes"][0]
        self.assertEqual(note["title"], "Blank Note")
        for field in ("claims", "limitations", "open_questions"):
            self.assertEqual(note[field][0]["text"], "Not found in the source.")
            self.assertTrue(note[field][0]["missing"])
            self.assertFalse(note[field][0]["citation"])
        self.assertFalse(payload["claims"])
        self.assertNotIn("Do nothing", json.dumps(payload["claims"]))

    def _assert_notes_grouped(self, payload: dict) -> None:
        notes = payload["notes"]
        window = next(note for note in notes if note["cite_id"] == "HX-011")
        window_claims = [item["text"] for item in window["claims"]]
        self.assertIn(
            "Chunks near 900 characters beat both 300-character slices and "
            "2,000-character slices on recall at five.",
            window_claims,
        )
        self.assertIn(
            "Very short slices lost the sentence that actually stated the result.",
            window_claims,
        )
        self.assertEqual(
            window["limitations"][0]["text"],
            "The ablation used English technical prose and may not transfer "
            "to slide decks with sparse text.",
        )
        self.assertIn("HX-011", window["limitations"][0]["citation"])
        self.assertIn("Does the best length change", window["open_questions"][0])
        self.assertIn("[HX-011", window["open_questions"][0])
        ledger = next(note for note in notes if note["title"] == "Pairwise Ledger comparison protocol")
        self.assertTrue(
            any("Blended summaries" in item["text"] for item in ledger["claims"])
        )
        latch = next(note for note in notes if note["cite_id"] == "HX-003")
        self.assertEqual(latch["title"], "Citation Latch faithfulness check")
        self.assertTrue(latch["claims"])
        self.assertTrue(any("Citation Latch" in item["text"] for item in latch["claims"]))
        for note in notes:
            title = note["title"]
            self.assertFalse(title.lower().startswith("minutes for "))
            self.assertFalse(title.lower().startswith("lab note for "))
            self.assertFalse(title.lower().startswith("memorandum on "))
            self.assertFalse(title.lower().startswith("advisory:"))
            blob = " ".join(
                item["text"] for item in note["claims"] + note["limitations"]
            )
            self.assertNotIn("Decision recorded", blob)
            self.assertNotIn("The working decision is", blob)
            self.assertNotIn("Operator decision", blob)
            self.assertNotIn("Recommended posture", blob)
        self.assertTrue(payload["risks"])
        self.assertIn("severity", payload["risks"][0])
        self.assertIn("action", payload["risks"][0])
        self.assertIn("citation", payload["risks"][0])

    def test_grouped_brief_titles_come_from_the_page(self) -> None:
        by_id = {doc.doc_id: doc for doc in documents()}
        chunks = []
        for doc_id in ("HX-011", "HX-009", "HX-003"):
            for chunk in by_id[doc_id].chunks():
                chunks.append({key: value for key, value in chunk.items() if key != "title"})
        payload = quote_payload("brief", chunks)
        self._assert_notes_grouped(payload)

    def test_demo_seed_excludes_heldout_notes(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "helix.db"
            embed_dir = Path(tmp) / "embeddings"
            embed_dir.mkdir()
            with patch.object(store, "DB_PATH", db_path), patch.object(
                store, "EMBED_STORE", embed_dir
            ), patch.object(store, "_encode_collection"):
                collection = store.seed_demo()
                cite_ids = {row["cite_id"] for row in store.list_documents(collection["id"])}
                self.assertEqual(len(cite_ids), 14)
                self.assertTrue(cite_ids.isdisjoint(EVAL_IDS))
                self.assertEqual(store.list_collections()[0]["chunks"], 28)
                self.assertEqual(store.seed_demo()["id"], collection["id"])
                self.assertEqual(len(store.list_documents(collection["id"])), 14)

                eval_doc = next(doc for doc in documents() if doc.doc_id in EVAL_IDS)
                store.add_document(
                    collection_id=collection["id"],
                    filename=f"{eval_doc.doc_id}.txt",
                    title=eval_doc.title,
                    source_type="seed",
                    pages=[
                        {"page": page.page, "section": page.section, "text": page.text}
                        for page in eval_doc.pages
                    ],
                    cite_id=eval_doc.doc_id,
                )
                store.seed_demo()
                cite_ids = {row["cite_id"] for row in store.list_documents(collection["id"])}
                self.assertNotIn(eval_doc.doc_id, cite_ids)
                self.assertEqual(len(cite_ids), 14)
                self.assertEqual(store.list_collections()[0]["chunks"], 28)

    def test_diversified_ranking_keeps_other_documents(self) -> None:
        ranked = []
        for index in range(8):
            ranked.append({
                "id": f"a{index}",
                "document_id": "doc-a",
                "cite_id": "HX-A",
                "score": 1.0 - (index * 0.01),
            })
        for offset, doc_id in enumerate(("doc-b", "doc-c", "doc-d")):
            ranked.append({
                "id": doc_id,
                "document_id": doc_id,
                "cite_id": doc_id,
                "score": 0.5 - (offset * 0.01),
            })
        selected = diversify_by_document(ranked, 8)
        self.assertEqual(
            [chunk["document_id"] for chunk in selected[:4]],
            ["doc-a", "doc-b", "doc-c", "doc-d"],
        )
        self.assertEqual(sum(chunk["document_id"] == "doc-a" for chunk in selected), 5)

        chunks = [
            {
                "id": "a2",
                "document_id": "doc-a",
                "cite_id": "HX-A",
                "page": 2,
                "section": "Evidence",
                "text": "zebra " * 12,
            },
            {
                "id": "a1",
                "document_id": "doc-a",
                "cite_id": "HX-A",
                "page": 1,
                "section": "Summary",
                "text": "zebra " * 4,
            },
            {
                "id": "b1",
                "document_id": "doc-b",
                "cite_id": "HX-B",
                "page": 1,
                "section": "Summary",
                "text": "zebra beta",
            },
        ]
        chunks.extend(
            {
                "id": f"filler-{index}",
                "document_id": "doc-a" if index < 3 else f"doc-f{index}",
                "cite_id": "HX-A" if index < 3 else f"HX-F{index}",
                "page": 1,
                "section": "Other",
                "text": "unrelated catalog notes about hardware",
            }
            for index in range(6)
        )

        def fake_chunks(_collection_id, document_ids=None):
            if document_ids:
                return [chunk for chunk in chunks if chunk["document_id"] in document_ids]
            return list(chunks)

        with patch("helix.index.search.get_chunks", fake_chunks), patch(
            "helix.index.search.load_embeddings", lambda _collection_id: None
        ):
            broad = search_collection("col", "zebra", limit=2)
            focused = search_collection("col", "zebra", limit=2, document_ids=["doc-a"])
        self.assertEqual({chunk["document_id"] for chunk in broad}, {"doc-a", "doc-b"})
        self.assertEqual({chunk["document_id"] for chunk in focused}, {"doc-a"})
        self.assertEqual(len(focused), 2)

    def test_rrf_and_json(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
        self.assertGreater(fused["b"], fused["a"])
        self.assertEqual(parse_json_object("note\n```json\n{\"a\": 1}\n```")["a"], 1)
        self.assertLess(jaccard("warm store of reused vectors", "p95 lookup latency embeddings"), 0.5)

    def test_pdf_table_and_chunker(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.pdf"
            doc = SimpleDocTemplate(str(path), pagesize=letter)
            doc.build([Table([["Subset", "Accuracy"], ["Digital", "0.96"]])])
            pages = parse_path(path)
            self.assertTrue(any("0.96" in page["text"] for page in pages))
            chunks = chunk_pages([{
                "page": 1,
                "section": "S",
                "text": "word " * 2000,
            }])
            self.assertGreater(len(chunks), 1)

    def test_seed_notes_stay_one_chunk_per_page(self) -> None:
        for doc in documents():
            pages = [
                {"page": page.page, "section": page.section, "text": page.text}
                for page in doc.pages
            ]
            chunks = chunk_pages(pages)
            self.assertEqual(len(chunks), len(doc.pages))
            self.assertEqual(
                [chunk["page"] for chunk in chunks],
                [page.page for page in doc.pages],
            )

    def test_collection_counts_do_not_cross_join(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "helix.db"
            with patch.object(store, "DB_PATH", db_path), patch.object(
                store, "_encode_collection"
            ):
                collection = store.create_collection("Sample library")
                for index in range(2):
                    store.add_document(
                        collection_id=collection["id"],
                        filename=f"HX-00{index}.txt",
                        title=f"Note {index}",
                        source_type="seed",
                        pages=[
                            {"page": 1, "section": "Summary", "text": "Page one is short."},
                            {"page": 2, "section": "Evidence", "text": "Page two is short."},
                        ],
                        cite_id=f"HX-00{index}",
                    )
                listed = store.list_collections()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["documents"], 2)
        self.assertEqual(listed[0]["chunks"], 4)


if __name__ == "__main__":
    unittest.main()
