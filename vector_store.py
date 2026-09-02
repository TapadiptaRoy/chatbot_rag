"""
vector_store.py
----------------
Connects to Chroma Cloud and handles adding, querying, listing, and
deleting document chunks (as vectors).
"""

import uuid
import chromadb
from config import CHROMA_TENANT, CHROMA_DATABASE, CHROMA_API_KEY
from embeddings import embed_documents, embed_query

client = chromadb.CloudClient(
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
    api_key=CHROMA_API_KEY,
)

collection = client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(doc_id, source_name, chunks):
    """Embed and store a list of text chunks belonging to one document."""
    vectors = embed_documents(chunks)

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    metadatas = [
        {"source": source_name, "doc_id": doc_id}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


def query(question, top_k=5):
    """Return the top_k most similar chunks to the question."""
    question_vector = embed_query(question)

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=top_k,
    )

    hits = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "score": 1 - distance,
        })

    return hits


def list_documents():
    """Return a list of unique source documents currently indexed."""
    all_items = collection.get()
    metadatas = all_items["metadatas"]

    seen = {}
    for meta in metadatas:
        doc_id = meta["doc_id"]

        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "source": meta["source"], "chunks": 0}

        seen[doc_id]["chunks"] += 1

    return list(seen.values())


def delete_document(doc_id):
    """Remove every chunk belonging to one document."""
    collection.delete(where={"doc_id": doc_id})


if __name__ == "__main__":
    # --- ONE-TIME CLEANUP: remove the two duplicate test docs from earlier runs ---
    

    test_doc_id = uuid.uuid4().hex
    test_chunks = [
        "The Eiffel Tower is located in Paris, France.",
        "Python is a popular programming language for AI development.",
    ]

    count = add_chunks(test_doc_id, "test.txt", test_chunks)
    print(f"Added {count} chunks with doc_id {test_doc_id}")
    print(f"Collection now has {collection.count()} total chunks")

    results = query("What programming language is good for AI?")
    for hit in results:
        print(f"{hit['score']:.3f} — {hit['source']} — {hit['text']}")

    print(list_documents())