"""FastAPI backend for the hybrid RAG app (FAISS + Neo4j + OpenAI).

PDFs are read from the server-side documents folder (config.PDF_DIR), not uploaded.
Run:  uvicorn main:app --port 8000
"""
import json
import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
import llm
from hybrid_store import get_store

# State of the "ingest PDFs from the documents folder" job (single process).
sync_lock = threading.Lock()
sync_state: dict = {"status": "idle", "progress": 0.0, "message": "", "files": [], "error": None}


def _run_sync():
    if not sync_lock.acquire(blocking=False):
        return  # a sync is already running
    try:
        sync_state.update(status="running", progress=0.0, message="Scanning documents folder...",
                          files=[], error=None)

        def progress(frac: float, msg: str):
            sync_state.update(progress=frac, message=msg)

        result = get_store().sync_folder(progress)
        message = "Done" if result["files"] else f"No PDFs found in {result['folder']}"
        sync_state.update(status="done", progress=1.0, message=message, files=result["files"])
    except Exception as e:
        sync_state.update(status="error", error=str(e))
    finally:
        sync_lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_store()  # load the embedding model and connect to Neo4j once, at startup
    if config.AUTO_SYNC:  # ingest new PDFs in the background; the API is usable meanwhile
        threading.Thread(target=_run_sync, daemon=True).start()
    yield


app = FastAPI(title="Hybrid RAG API", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return get_store().stats()


@app.get("/documents")
def documents():
    return get_store().list_documents()


@app.post("/sync", status_code=202)
def sync(background: BackgroundTasks):
    """Scan the documents folder and ingest any PDF not already stored."""
    if sync_lock.locked():
        return {"status": "running"}
    sync_state.update(status="running", progress=0.0, message="Starting...", files=[], error=None)
    background.add_task(_run_sync)
    return {"status": "started"}


@app.get("/sync/status")
def sync_status():
    return sync_state


@app.post("/chat")
def chat(req: ChatRequest):
    """Streams newline-delimited JSON: one 'sources' event, then 'token' events."""
    ctx = get_store().retrieve(req.question)
    history = [m.model_dump() for m in req.history]

    def events():
        yield json.dumps({"type": "sources", "data": ctx}) + "\n"
        try:
            for token in llm.answer_stream(req.question, ctx, history):
                yield json.dumps({"type": "token", "data": token}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.delete("/reset")
def reset():
    if sync_lock.locked():
        raise HTTPException(409, "A document sync is running. Try again when it finishes.")
    get_store().reset()
    sync_state.update(status="idle", progress=0.0, message="", files=[], error=None)
    return {"status": "reset"}
