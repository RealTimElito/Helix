"""FastAPI application for upload, grounded questions, and monitors."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from helix import __version__
from helix.agent.graph import run_agent
from helix.config import ADAPTER_DIR, EMBED_DIR, ROOT
from helix.index.search import search_collection
from helix.index.store import (
    add_upload,
    create_collection,
    create_monitor,
    delete_document,
    get_monitor,
    list_collections,
    list_documents,
    list_monitors,
    reindex_all,
    save_monitor_brief,
    save_trajectory,
    seed_demo,
)
from helix.models.embedder import get_embedder
from helix.models.llm import generate_locked, get_llm

_MODES = {"brief", "extract", "risk", "qa", "compare"}
_DIST = ROOT / "frontend" / "dist"


class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: str = "brief"
    document_ids: list[str] = Field(default_factory=list)


class MonitorIn(BaseModel):
    collection_id: str
    topic: str = Field(min_length=3, max_length=240)


def _preload_llm() -> None:
    try:
        get_llm()
    except Exception as exc:  # noqa: BLE001
        print(f"LLM preload failed: {exc}", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    seed_demo()
    get_embedder()
    reindex_all()
    threading.Thread(target=_preload_llm, daemon=True).start()
    yield


app = FastAPI(title="Helix", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    from helix.models.llm import _SINGLETON

    return {
        "version": __version__,
        "embed": "helix" if (EMBED_DIR / "modules.json").exists() else "base",
        "adapter": (ADAPTER_DIR / "adapter_config.json").exists(),
        "llm": "ready" if _SINGLETON is not None else "loading",
    }


@app.get("/api/collections")
def collections() -> list[dict]:
    return list_collections()


@app.post("/api/collections")
def post_collection(body: CollectionIn) -> dict:
    return create_collection(body.name)


@app.get("/api/collections/{collection_id}/documents")
def documents(collection_id: str) -> list[dict]:
    return list_documents(collection_id)


@app.post("/api/collections/{collection_id}/documents")
async def upload_document(collection_id: str, file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".markdown", ".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail="Upload a PDF, text file, or image.")
    destination = ROOT / "data" / "uploads"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / Path(file.filename or "upload.txt").name
    target.write_bytes(await file.read())
    try:
        return add_upload(collection_id, target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/documents/{document_id}")
def remove_document(document_id: str) -> dict:
    delete_document(document_id)
    return {"ok": True}


@app.post("/api/collections/{collection_id}/ask")
def ask(collection_id: str, body: AskIn) -> dict:
    if body.mode not in _MODES:
        raise HTTPException(status_code=400, detail="Unknown mode.")
    result = run_agent(
        question=body.question,
        mode=body.mode,
        collection_id=collection_id,
        document_ids=body.document_ids,
        generate=generate_locked,
        search=search_collection,
    )
    save_trajectory(collection_id, body.question, body.mode, result)
    return result


@app.get("/api/monitors")
def monitors(collection_id: str | None = None) -> list[dict]:
    return list_monitors(collection_id)


@app.post("/api/monitors")
def post_monitor(body: MonitorIn) -> dict:
    return create_monitor(body.collection_id, body.topic)


@app.post("/api/monitors/{monitor_id}/run")
def run_monitor(monitor_id: str) -> dict:
    monitor = get_monitor(monitor_id)
    if monitor is None:
        raise HTTPException(status_code=404, detail="Monitor not found.")
    question = (
        f"Draft a weekly synthesis on {monitor['topic']} with new developments, "
        "implications, risks, and sourced actions from this collection."
    )
    result = run_agent(
        question=question,
        mode="brief",
        collection_id=monitor["collection_id"],
        document_ids=None,
        generate=generate_locked,
        search=search_collection,
    )
    save_monitor_brief(monitor_id, result)
    result["monitor_id"] = monitor_id
    return result


if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_DIST / "index.html")
