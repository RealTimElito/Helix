# Helix

Helix is a local fact-checking desk. You upload notes, ask for claims, limitations, and open questions, and the answer shows only sentences that appear on a cited page.

The demo collection is fourteen original training notes, `HX-001` through `HX-014`. Held-out notes `HX-015` through `HX-020` stay in the library for evaluation and are not in the demo index.

A whole-collection brief or extract does not call the language model. It loads every note in the selected collection, quotes sentences from those pages, and checks that each shown line occurs on the cited page. Other modes (a single note, compare, risk, and a direct question) load Qwen3-4B on CPU and can take several minutes the first time. A GPU is optional; the server uses one when PyTorch can see it.

Faithfulness is the citation check. One hundred percent means each shown line appears on its cited page. It does not mean the model agreed with the notes, and it does not by itself mean every note was read. The coverage line reports notes checked against notes in the collection, and claims kept against claims dropped.

## Run the demo

Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

Modes other than the whole-collection brief also need PyTorch, Transformers, PEFT, and Sentence-Transformers in that environment. Install those from their own projects. CPU is enough.

Build the interface, then start the API on port 8020:

```bash
cd frontend && npm install && npm run build && cd ..
.venv/bin/python -m uvicorn helix.api.main:app --host 127.0.0.1 --port 8020
```

Open `http://127.0.0.1:8020`. The first launch seeds the fourteen-note demo collection.

Base weights are not in this repository. Download `Qwen/Qwen3-4B-Instruct-2507` into `models/Qwen3-4B-Instruct-2507`. Those weights stay under the Qwen license. The optional LoRA adapter lives in `models/helix-lora`. The embedder starts from `sentence-transformers/all-MiniLM-L6-v2`, which has its own license, and a fine-tuned copy can sit in `models/helix-embed`.

Sample PDFs for the upload path are in `data/samples`. Regenerate them with:

```bash
.venv/bin/python -m helix.corpus.export_pdfs
```

## Tests

```bash
.venv/bin/python -m unittest tests.test_helix
```

## Layout

- `src/helix/ingest` parses PDFs, text, and tables
- `src/helix/index` stores chunks and runs hybrid search
- `src/helix/agent` plans, retrieves or loads the collection, drafts, and checks citations
- `src/helix/api` is the FastAPI server
- `frontend` is the React interface
- `docs` holds the dataset card, model cards, architecture, and evaluation report

Training commands live under `src/helix/train` and `src/helix/data`. They are separate from running the demo.
