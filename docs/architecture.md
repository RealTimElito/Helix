# Helix architecture

Helix turns a collection of technical documents into cited briefs. Each stage is a separate module so retrieval, the language model, and the citation check can be evaluated on their own.

## Ingestion

PDFs are read with PyMuPDF. pdfplumber adds any detected tables as text. Plain text and Markdown are stored as a single page. Images are stored with an explicit note that this build does not OCR them. Pages longer than 1,600 characters are split on sentence boundaries.

## Retrieval

Chunks are stored in SQLite. A fine-tuned MiniLM encoder writes normalized vectors beside the database. Search ranks a keyword index and the dense index separately, then combines them with reciprocal rank fusion (k = 60).

The encoder is trained with a contrastive loss on paraphrased questions from the training notes. Held-out notes and handwritten distractors are reserved for recall and nDCG.

## Language model

The generator is Qwen3-4B-Instruct-2507 with a rank-16 LoRA adapter on the attention projections. Supervised targets are JSON briefs, extractions, risk notes, quoted answers, and pairwise comparisons. Loss is applied only to the assistant tokens. Training notes and evaluation notes do not overlap.

## Agent

A LangGraph workflow runs four tools:

1. `plan` rewrites the question into a search query, except for an unfiltered collection brief or extract, which skips the rewrite.
2. `retrieve` calls hybrid search for a selected note, a comparison, a risk note, or a direct question. An unfiltered collection brief or extract loads every page in that collection instead of a top-k subset.
3. `draft` asks the fine-tuned model for JSON in the same format used for training. An unfiltered collection brief or extract quotes sentences from the pages and does not call the model.
4. `cite_check` drops any field whose citation is missing or whose sentence is not supported by that page. One repair pass is allowed after a model draft. The quote path does not repair.

Faithfulness is that citation check: each shown line occurs on the cited page. A separate coverage count reports notes checked against notes in the collection, and claims kept against claims dropped. The demo index contains the fourteen training notes. Held-out notes `HX-015` through `HX-020` stay out of that index.

Trajectories are stored with the answer.

## Application

FastAPI serves collections, uploads, questions, and topic monitors. The React interface uses TanStack Query for collections, documents, and monitors. The first screen is the question box; the library and the monitor list are separate views over the same collection id.

## What this build does not do

There is no production OCR, no scheduled cron daemon, and no hosted demo video. Monitors run when an analyst presses "Run synthesis". A walkthrough script is in `docs/demo_script.md`.
