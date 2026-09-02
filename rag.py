"""
Ties retrieval (vector_store) together with generation (Claude) to answer
questions grounded in the user's personal knowledge base.
"""

import anthropic
from config import ANTHROPIC_API_KEY
import vector_store

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL_NAME = "claude-sonnet-5"
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the \
provided context from the user's personal knowledge base.

Rules:
- Base your answer strictly on the provided context.
- If the context doesn't contain enough information to answer, say so clearly \
instead of guessing.
- Cite which source(s) you drew from using the bracketed labels given, e.g. [1], [2].
- Be concise and direct."""


def build_context_block(hits):
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
    return "\n\n".join(blocks)


def answer_question(question, top_k=TOP_K):
    hits = vector_store.query(question, top_k=top_k)

    if not hits:
        return {
            "answer": "Your knowledge base is empty (or nothing matched). Upload some documents first.",
            "sources": [],
        }

    context_block = build_context_block(hits)
    user_message = f"Context from knowledge base:\n\n{context_block}\n\nQuestion: {question}"

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = response.content[0].text

    return {
        "answer": answer_text,
        "sources": [hit["source"] for hit in hits],
    }


if __name__ == "__main__":
    result = answer_question("What programming language is good for AI?")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])