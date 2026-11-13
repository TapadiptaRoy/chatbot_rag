"""
vector_store.py
----------------
Connects to Chroma Cloud and handles adding, querying, listing, and
deleting document chunks (as vectors).
"""

import os
import uuid
import chromadb
from config import CHROMA_TENANT, CHROMA_DATABASE, CHROMA_API_KEY
from embeddings import embed_documents, embed_query

# Defaults to the live collection. Set CHROMA_COLLECTION in the environment
# (e.g. eval_knowledge_base) to point at a separate collection for testing.
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "knowledge_base")

# Chroma Cloud allows at most 300 records per write and returns at most
# 300 records per request, so writes and reads are done in pages of 300.
CHROMA_PAGE_SIZE = 300

client = chromadb.CloudClient(
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
    api_key=CHROMA_API_KEY,
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(doc_id, source_name, chunks):
    """Embed and store a list of text chunks belonging to one document."""
    vectors = embed_documents(chunks)

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    metadatas = [
        {"source": source_name, "doc_id": doc_id}
        for _ in chunks
    ]

    for start in range(0, len(chunks), CHROMA_PAGE_SIZE):
        end = start + CHROMA_PAGE_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
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
    seen = {}
    offset = 0

    while True:
        batch = collection.get(
            limit=CHROMA_PAGE_SIZE,
            offset=offset,
            include=["metadatas"],
        )
        metadatas = batch["metadatas"]
        if not metadatas:
            break

        for meta in metadatas:
            doc_id = meta["doc_id"]

            if doc_id not in seen:
                seen[doc_id] = {"doc_id": doc_id, "source": meta["source"], "chunks": 0}

            seen[doc_id]["chunks"] += 1

        if len(metadatas) < CHROMA_PAGE_SIZE:
            break
        offset += CHROMA_PAGE_SIZE

    return list(seen.values())


def delete_document(doc_id):
    """Remove every chunk belonging to one document."""
    collection.delete(where={"doc_id": doc_id})


if __name__ == "__main__":
    print(f"Using collection: {COLLECTION_NAME}")

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

    # Clean up so test data doesn't pile up in the collection
    delete_document(test_doc_id)
    print("Test document deleted.")