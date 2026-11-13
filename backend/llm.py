
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import config


# OpenAI chat model for generating answers.
_client = ChatOpenAI(
    model=config.OPENAI_MODEL,
    api_key=config.OPENAI_API_KEY,
    max_tokens=1500,
    temperature=0,
)


def complete(system: str, user: str, max_tokens: int = 1500) -> str:
    """Generate a complete response using OpenAI."""
    response = _client.bind(max_tokens=max_tokens).invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
    )
    return str(response.content)


def stream(system: str, messages: list[dict], max_tokens: int = 1500):
    """Stream an OpenAI response as text chunks."""
    model = _client.bind(max_tokens=max_tokens)

    # Convert the conversation history to LangChain messages.
    chat_messages = [SystemMessage(content=system)]

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            chat_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            chat_messages.append(AIMessage(content=content))

    # Yield each generated text chunk.
    for chunk in model.stream(chat_messages):
        if isinstance(chunk.content, str):
            if chunk.content:
                yield chunk.content
        elif isinstance(chunk.content, list):
            for block in chunk.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    yield block.get("text", "")


def parse_json(text: str) -> dict:
    """Parse JSON, including JSON wrapped in Markdown fences."""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        # Try extracting a JSON object from surrounding prose.
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            try:
                result = json.loads(match.group(0))
                return result if isinstance(result, dict) else {}
            except json.JSONDecodeError:
                pass

    return {}


ANSWER_SYSTEM = """You are a helpful assistant answering questions about the user's PDF documents.
You receive two kinds of context:
- DOCUMENT PASSAGES from a vector search (with file and page).
- KNOWLEDGE GRAPH FACTS as (subject) -[RELATION]-> (object) triples, plus passages linked to those entities.
Answer using ONLY this context. Mention page numbers when helpful. If the context
does not contain the answer, say so plainly instead of guessing."""


def _format_context(ctx: dict) -> str:
    """Format vector-search and graph-search results for the LLM."""
    parts = ["## DOCUMENT PASSAGES (vector search)"]

    for passage in ctx.get("vector", []):
        parts.append(
            f"[{passage.get('doc', 'Unknown document')}, "
            f"p.{passage.get('page', '?')}] {passage.get('text', '')}"
        )

    graph = ctx.get("graph", {})
    parts.append("\n## KNOWLEDGE GRAPH FACTS")
    parts.extend(graph.get("facts", []) or ["(none found)"])

    parts.append("\n## PASSAGES LINKED TO GRAPH ENTITIES")

    for passage in graph.get("chunks", []):
        parts.append(
            f"[{passage.get('doc', 'Unknown document')}, "
            f"p.{passage.get('page', '?')}] {passage.get('text', '')}"
        )

    return "\n".join(parts)


def answer_stream(question: str, ctx: dict, history: list[dict]):
    """Stream an answer grounded in the retrieved context."""
    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in history[-6:]
        if message.get("role") in {"user", "assistant"}
    ]

    messages.append({
        "role": "user",
        "content": f"{_format_context(ctx)}\n\n## QUESTION\n{question}",
    })

    yield from stream(ANSWER_SYSTEM, messages)
