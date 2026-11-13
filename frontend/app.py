"""Streamlit chat frontend. Talks to the FastAPI backend over HTTP only.
PDFs are NOT uploaded here: the backend loads them from the project's documents/ folder.

Run:  BACKEND_URL=http://localhost:8000 streamlit run app.py
"""
import json
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="My Chat Bot", page_icon="💬", layout="wide")
st.title("💬 My Chat Bot")
st.caption("Answers come from a vector DB (FAISS) and a knowledge graph (Neo4j).")


def api(method: str, path: str, timeout: int = 15, **kwargs):
    r = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    r.raise_for_status()
    return r.json()


# ---------- backend check ----------
try:
    api("GET", "/health", timeout=3)
except Exception:
    st.error(f"Backend not reachable at {BACKEND_URL}. "
             "Start it with `uvicorn main:app --port 8000` from the backend folder.")
    st.stop()

st.session_state.setdefault("messages", [])


def stream_answer(prompt: str, history: list[dict], holder: dict):
    """Yield answer tokens; stash the retrieved sources in `holder`."""
    payload = {
        "question": prompt,
        "history": [{"role": m["role"], "content": m["content"]} for m in history],
    }
    with requests.post(f"{BACKEND_URL}/chat", json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "sources":
                holder["ctx"] = event["data"]
            elif event["type"] == "token":
                yield event["data"]
            elif event["type"] == "error":
                raise RuntimeError(event["data"])


def show_sources(ctx: dict):
    with st.expander("Sources used"):
        st.markdown("**Vector search**")
        for p in ctx["vector"]:
            st.markdown(f"- `{p['doc']}` p.{p['page']} (similarity {p['score']:.2f})")
            st.caption(p["text"][:300] + "...")
        g = ctx["graph"]
        st.markdown("**Knowledge graph**")
        if g["entities"]:
            st.caption("Matched entities: " + ", ".join(g["entities"]))
        for f in g["facts"][:10]:
            st.markdown(f"- {f}")
        if not g["facts"]:
            st.caption("No graph facts matched this question.")


# Refreshes on its own every 5s without interrupting the chat.
@st.fragment(run_every=5)
def status_panel():
    try:
        sync = api("GET", "/sync/status")
        stats = api("GET", "/stats")
        docs = api("GET", "/documents")
    except Exception:
        st.warning("Could not load status from the backend.")
        return

    if sync["status"] == "running":
        st.progress(min(sync["progress"], 1.0), text=sync["message"])
        st.caption("You can keep chatting; new documents become searchable as they finish.")
    elif sync["status"] == "error":
        st.error(sync["error"])
    elif sync["status"] == "done":
        for f in sync["files"]:
            if "error" in f:
                st.error(f"{f['file']}: {f['error']}")
        if not sync["files"]:
            st.info(sync["message"])

    st.metric("Chunks in vector DB", stats["chunks"])
    st.metric("Entities in graph", stats["entities"])
    st.metric("Relations in graph", stats["relations"])

    if docs:
        st.markdown("**Indexed files**")
        for d in docs:
            warn = f" ⚠ {d['graph_failures']} chunks missing graph data" if d["graph_failures"] else ""
            st.markdown(f"- {d['file']} ({d['chunks']} chunks){warn}")
    elif sync["status"] != "running":
        st.info("No documents indexed yet. Put PDFs in the `documents/` folder "
                "and click **Rescan documents**.")


# ---------- sidebar ----------
with st.sidebar:
    st.header("Documents")
    st.caption("PDFs are loaded by the backend from the project's `documents/` folder.")
    status_panel()

    if st.button("Rescan documents", type="primary"):
        try:
            api("POST", "/sync")
            st.toast("Scanning the documents folder...")
        except Exception as e:
            st.error(f"Could not start the scan: {e}")

    if st.button("Reset knowledge base"):
        try:
            api("DELETE", "/reset", timeout=60)
            st.session_state.messages.clear()
            st.rerun()
        except requests.HTTPError as e:
            st.warning(e.response.json().get("detail", str(e)))
        except Exception as e:
            st.error(str(e))

# ---------- chat ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("ctx"):
            show_sources(m["ctx"])

if prompt := st.chat_input("Ask something about your documents..."):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        holder: dict = {}
        try:
            answer = st.write_stream(stream_answer(prompt, history, holder))
        except Exception as e:
            answer = f"Sorry, something went wrong: {e}"
            st.error(answer)
        if holder.get("ctx"):
            show_sources(holder["ctx"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "ctx": holder.get("ctx")}
    )
