"""
Single-file hybrid store:
  * Vector DB       -> FAISS (cosine similarity, persisted to disk)
  * Knowledge graph -> Neo4j (Chunk / Entity nodes, RELATED edges)

Each PDF is written to BOTH; one question queries BOTH.
"""
import hashlib
import io
import json
import os
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from neo4j import GraphDatabase
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config
import llm

GRAPH_SYSTEM = """You extract knowledge graphs from text.
Return ONLY JSON, no prose and no markdown fences, in this shape:
{"entities": [{"name": "...", "type": "Person|Organization|Product|Concept|Location|Event|Other"}],
 "relations": [{"source": "...", "target": "...", "type": "SHORT_UPPER_SNAKE_VERB"}]}
Rules: use the full, canonical name of each entity; every relation's source and
target must also appear in entities; only include facts stated in the text;
at most 15 entities and 20 relations."""

ENTITY_SYSTEM = (
    'Extract the key named entities and concepts from the question. '
    'Return ONLY JSON: {"entities": ["...", "..."]}'
)


# ---------------------------------------------------------------- PDF helpers
def load_pdf(data: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(data))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
    chunks = []
    for page_no, text in pages:
        text = " ".join(text.split())
        for start in range(0, len(text), step):
            piece = text[start:start + config.CHUNK_SIZE].strip()
            if len(piece) > 50:
                chunks.append({"text": piece, "page": page_no})
    return chunks


# ---------------------------------------------------------------- the store
class HybridStore:
    def __init__(self):
        # --- vector side (FAISS) ---
        os.makedirs(config.FAISS_PATH, exist_ok=True)
        self._index_file = os.path.join(config.FAISS_PATH, "index.faiss")
        self._meta_file = os.path.join(config.FAISS_PATH, "meta.json")
        self.model = SentenceTransformer(config.EMBED_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._lock = threading.RLock()  # API serves chat while ingestion runs
        self._load_vectors()
        self._manifest_file = os.path.join(config.FAISS_PATH, "manifest.json")
        self._load_manifest()

        # --- graph side (Neo4j) ---
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        self.driver.verify_connectivity()
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                  "FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            s.run("CREATE CONSTRAINT entity_key IF NOT EXISTS "
                  "FOR (e:Entity) REQUIRE e.key IS UNIQUE")

    # ============================ VECTOR DB (FAISS) ============================
    def _load_vectors(self):
        if os.path.exists(self._index_file) and os.path.exists(self._meta_file):
            self.index = faiss.read_index(self._index_file)
            with open(self._meta_file, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self._empty_vectors()

    def _empty_vectors(self):
        # Inner product on L2-normalised vectors == cosine similarity.
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        self.meta = {"next_id": 0, "records": {}}

    def _save_vectors(self):
        faiss.write_index(self.index, self._index_file)
        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(self.meta, f)

    # Manifest: which documents are fully stored in BOTH databases (keyed by content hash).
    def _load_manifest(self):
        self.manifest = {}
        if os.path.exists(self._manifest_file):
            with open(self._manifest_file, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)

    def _save_manifest(self):
        with open(self._manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return vecs.astype("float32")

    def _vector_delete_doc(self, doc_id: str):
        ids = [int(i) for i, r in self.meta["records"].items() if r["doc_id"] == doc_id]
        if ids:
            self.index.remove_ids(np.array(ids, dtype="int64"))
            for i in ids:
                del self.meta["records"][str(i)]

    def _vector_add(self, doc_id: str, texts: list[str], metadatas: list[dict]):
        vecs = self._embed(texts)  # slow part: done outside the lock
        with self._lock:
            self._vector_delete_doc(doc_id)  # re-ingesting replaces, never duplicates
            start = self.meta["next_id"]
            ids = np.arange(start, start + len(texts), dtype="int64")
            self.index.add_with_ids(vecs, ids)
            for i, text, md in zip(ids, texts, metadatas):
                self.meta["records"][str(int(i))] = {"doc_id": doc_id, "text": text, **md}
            self.meta["next_id"] = start + len(texts)
            self._save_vectors()

    def vector_search(self, question: str, k: int = config.TOP_K) -> list[dict]:
        qvec = self._embed([question])
        with self._lock:
            n = int(self.index.ntotal)
            if n == 0:
                return []
            scores, ids = self.index.search(qvec, min(k, n))
            return [
                {**self.meta["records"][str(int(i))], "score": float(s)}
                for s, i in zip(scores[0], ids[0]) if i != -1
            ]

    # ========================== KNOWLEDGE GRAPH (Neo4j) ==========================
    @staticmethod
    def _extract_graph(text: str) -> dict:
        data = llm.parse_json(llm.complete(GRAPH_SYSTEM, text, max_tokens=4000, json_mode=True))
        entities = {}
        for e in data.get("entities", []):
            name = str(e.get("name", "")).strip()
            if len(name) >= 2:
                entities[name.lower()] = {
                    "name": name, "key": name.lower(), "type": e.get("type", "Other"),
                }
        relations = []
        for r in data.get("relations", []):
            s, t = str(r.get("source", "")).strip(), str(r.get("target", "")).strip()
            if s.lower() in entities and t.lower() in entities:
                relations.append({
                    "source": s.lower(), "target": t.lower(),
                    "type": str(r.get("type", "RELATED_TO")).strip().upper()[:50],
                })
        return {"entities": list(entities.values()), "relations": relations}

    @staticmethod
    def _graph_write(session, chunk_id, text, doc_name, page, graph):
        session.run(
            "MERGE (c:Chunk {id:$cid}) SET c.text=$text, c.doc=$doc, c.page=$page",
            cid=chunk_id, text=text, doc=doc_name, page=page,
        )
        session.run(
            """
            MATCH (c:Chunk {id:$cid})
            UNWIND $entities AS e
            MERGE (n:Entity {key:e.key})
              ON CREATE SET n.name = e.name, n.type = e.type
            MERGE (c)-[:MENTIONS]->(n)
            """,
            cid=chunk_id, entities=graph["entities"],
        )
        if graph["relations"]:
            session.run(
                """
                UNWIND $rels AS r
                MATCH (a:Entity {key:r.source}), (b:Entity {key:r.target})
                MERGE (a)-[x:RELATED {type:r.type}]->(b)
                SET x.doc = $doc
                """,
                rels=graph["relations"], doc=doc_name,
            )

    def graph_search(self, question: str, max_facts: int = 25, max_chunks: int = 3) -> dict:
        try:
            raw = llm.parse_json(llm.complete(ENTITY_SYSTEM, question, max_tokens=1000, json_mode=True))
        except Exception:
            raw = {}
        keys = [str(e).lower().strip() for e in raw.get("entities", [])]
        keys = [k for k in keys if len(k) >= 3]
        if not keys:
            return {"entities": [], "facts": [], "chunks": []}

        with self.driver.session() as s:
            facts = [
                f"({r['s']}) -[{r['t']}]-> ({r['o']})"
                for r in s.run(
                    """
                    MATCH (e:Entity) WHERE any(k IN $keys WHERE e.key CONTAINS k)
                    MATCH (e)-[r:RELATED]-(:Entity)
                    RETURN DISTINCT startNode(r).name AS s, r.type AS t, endNode(r).name AS o
                    LIMIT $n
                    """,
                    keys=keys, n=max_facts,
                )
            ]
            chunks = [
                {"text": r["text"], "doc": r["doc"], "page": r["page"]}
                for r in s.run(
                    """
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                    WHERE any(k IN $keys WHERE e.key CONTAINS k)
                    RETURN c.text AS text, c.doc AS doc, c.page AS page,
                           count(DISTINCT e) AS hits
                    ORDER BY hits DESC LIMIT $n
                    """,
                    keys=keys, n=max_chunks,
                )
            ]
        return {"entities": keys, "facts": facts, "chunks": chunks}

    # ============================== PUBLIC API ==============================
    def ingest_pdf(self, file_name: str, data: bytes, progress=None) -> dict:
        """Store one PDF in BOTH the vector DB and the knowledge graph."""
        def report(frac, msg):
            if progress:
                progress(min(frac, 1.0), msg)

        doc_id = hashlib.sha1(data).hexdigest()[:10]
        if doc_id in self.manifest:  # identical content already stored in both DBs
            return {"skipped": True, "chunks": self.manifest[doc_id]["chunks"]}
        chunks = chunk_pages(load_pdf(data))
        if not chunks:
            raise ValueError("No extractable text found (is this a scanned PDF?).")
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]

        report(0.0, "Embedding chunks into the FAISS index...")
        self._vector_add(
            doc_id,
            [c["text"] for c in chunks],
            [{"chunk_id": cid, "doc": file_name, "page": c["page"]}
             for cid, c in zip(ids, chunks)],
        )
        report(0.2, "Vector DB done. Building knowledge graph...")

        failed = 0
        with self.driver.session() as session:
            for i, (cid, chunk) in enumerate(zip(ids, chunks)):
                try:
                    graph = self._extract_graph(chunk["text"])
                except Exception:
                    graph, failed = {"entities": [], "relations": []}, failed + 1
                self._graph_write(session, cid, chunk["text"], file_name, chunk["page"], graph)
                report(0.2 + 0.8 * (i + 1) / len(chunks),
                       f"Graph extraction {i + 1}/{len(chunks)}")
        self.manifest[doc_id] = {
            "file": file_name,
            "chunks": len(chunks),
            "graph_failures": failed,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._save_manifest()
        return {"skipped": False, "chunks": len(chunks), "graph_failures": failed}

    def sync_folder(self, progress=None) -> dict:
        """Ingest every PDF in config.PDF_DIR that isn't already in both databases."""
        folder = Path(config.PDF_DIR)
        if not folder.is_dir():
            raise FileNotFoundError(f"Documents folder not found: {folder}")
        pdfs = sorted(p for p in folder.iterdir()
                      if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith("."))
        total, results = len(pdfs), []
        for n, path in enumerate(pdfs):
            def sub(frac, msg, n=n, name=path.name):
                if progress:
                    progress((n + frac) / total, f"{name}: {msg}")
            try:
                res = self.ingest_pdf(path.name, path.read_bytes(), sub)
            except Exception as e:
                res = {"error": str(e)}
            results.append({"file": path.name, **res})
            if progress:
                progress((n + 1) / total, f"{path.name}: done")
        return {"folder": str(folder), "files": results}

    def list_documents(self) -> list[dict]:
        return sorted(
            ({"file": d["file"], "chunks": d["chunks"], "graph_failures": d["graph_failures"]}
             for d in self.manifest.values()),
            key=lambda d: d["file"],
        )

    def retrieve(self, question: str) -> dict:
        """Hybrid retrieval: vector similarity + knowledge-graph traversal."""
        return {"vector": self.vector_search(question), "graph": self.graph_search(question)}

    def stats(self) -> dict:
        with self.driver.session() as s:
            row = s.run(
                "MATCH (e:Entity) WITH count(e) AS entities "
                "OPTIONAL MATCH ()-[r:RELATED]->() "
                "RETURN entities, count(r) AS relations"
            ).single()
        return {
            "chunks": int(self.index.ntotal),
            "entities": row["entities"],
            "relations": row["relations"],
        }

    def reset(self):
        with self._lock:
            self._empty_vectors()
            self._save_vectors()
            self.manifest = {}
            self._save_manifest()
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")


@lru_cache(maxsize=1)
def get_store() -> HybridStore:
    return HybridStore()
