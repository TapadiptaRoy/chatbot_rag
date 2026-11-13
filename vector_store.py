"""
vector_store.py
----------------
Connects to Chroma Cloud and handles adding, querying, listing, and
deleting document chunks (as vectors). Every chunk carries an "owner" label
so each visitor only sees their own uploads plus the shared public samples.
"""

import os
import time
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

# Documents owned by PUBLIC_OWNER (your sample documents, the eval set) are
# visible to everyone and can't be deleted through the web app.
PUBLIC_OWNER = "public"

client = chromadb.CloudClient(
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
    api_key=CHROMA_API_KEY,
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _visible_to(owner):
    """Metadata filter: public documents plus the given owner's own documents."""
    owners = [PUBLIC_OWNER]
    if owner and owner != PUBLIC_OWNER:
        owners.append(owner)
    return {"owner": {"$in": owners}}


def add_chunks(doc_id, source_name, chunks, owner=PUBLIC_OWNER):
    """Embed and store a list of text chunks belonging to one document."""
    vectors = embed_documents(chunks)

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    uploaded_at = int(time.time())  # lets a later cleanup job delete old uploads
    metadatas = [
        {
            "source": source_name,
            "doc_id": doc_id,
            "owner": owner,
            "uploaded_at": uploaded_at,
        }
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


def query(question, top_k=5, owner=None):
    """Return the top_k most similar chunks among the public documents plus
    the given owner's own documents."""
    question_vector = embed_query(question)

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=top_k,
        where=_visible_to(owner),
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


def list_documents(owner=None):
    """Return the unique documents visible to this owner (public ones included)."""
    seen = {}
    offset = 0

    while True:
        batch = collection.get(
            where=_visible_to(owner),
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
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "source": meta["source"],
                    "chunks": 0,
                    "public": meta.get("owner") == PUBLIC_OWNER,
                }

            seen[doc_id]["chunks"] += 1

        if len(metadatas) < CHROMA_PAGE_SIZE:
            break
        offset += CHROMA_PAGE_SIZE

    return list(seen.values())


def delete_document(doc_id, owner=None):
    """
    Remove every chunk belonging to one document.

    With an owner (what the web app always passes), only that owner's copy is
    deleted, so a visitor can't delete someone else's or a public document.
    Without an owner it deletes by doc_id alone; that is for admin scripts only.
    """
    if owner == PUBLIC_OWNER:
        raise ValueError("Public documents can't be deleted through an owner.")

    if owner is None:
        collection.delete(where={"doc_id": doc_id})
    else:
        collection.delete(where={"$and": [{"doc_id": doc_id}, {"owner": owner}]})


if __name__ == "__main__":
    print(f"Using collection: {COLLECTION_NAME}")

    alice, bob = "test-owner-alice", "test-owner-bob"
    doc_a, doc_p = uuid.uuid4().hex, uuid.uuid4().hex

    add_chunks(doc_a, "alice-private.txt",
               ["Alice's secret plan is to learn about vector databases."], owner=alice)
    add_chunks(doc_p, "shared-sample.txt",
               ["Vector databases store embeddings for similarity search."])  # public

    print("Alice sees:", sorted(d["source"] for d in list_documents(alice)))
    print("Bob sees  :", sorted(d["source"] for d in list_documents(bob)))

    print("Bob's search finds:", sorted({h["source"] for h in query("vector databases", owner=bob)}))
    print("Alice's search finds:", sorted({h["source"] for h in query("vector databases", owner=alice)}))

    delete_document(doc_a, owner=bob)    # Bob tries to delete Alice's document
    print("After Bob's delete attempt, Alice still has:",
          sorted(d["source"] for d in list_documents(alice) if not d["public"]))

    delete_document(doc_a, owner=alice)  # Alice deletes her own
    delete_document(doc_p)               # admin cleanup of the public test doc
    print("After cleanup, Alice sees:", sorted(d["source"] for d in list_documents(alice)))