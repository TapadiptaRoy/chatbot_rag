"""
add_sample.py
-------------
Manage the PUBLIC sample documents: visible to every visitor, and not
deletable through the web app.

Usage (project root, venv active):
  python add_sample.py path/to/file.pdf     add a sample document
  python add_sample.py --list               list the public sample documents
  python add_sample.py --remove DOC_ID      remove a sample document

This writes to the collection named by CHROMA_COLLECTION, which is your LIVE
collection ("knowledge_base") unless you set it otherwise.
"""

import os
import sys
import uuid

import document_loader
import vector_store


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    print(f"Using collection: {vector_store.COLLECTION_NAME}")

    if args[0] == "--list":
        docs = vector_store.list_documents()  # no owner given -> public documents only
        for d in docs:
            print(f"{d['doc_id']}  {d['chunks']:>4} chunks  {d['source']}")
        print(f"{len(docs)} public document(s)")
        return

    if args[0] == "--remove" and len(args) == 2:
        vector_store.delete_document(args[1])  # admin delete by doc_id
        print(f"Removed {args[1]}")
        return

    path = args[0]
    text = document_loader.extract_text(path)
    chunks = document_loader.chunk_text(text)
    if not chunks:
        sys.exit("No extractable text found in that file.")

    doc_id = "sample_" + uuid.uuid4().hex[:8]
    count = vector_store.add_chunks(
        doc_id, os.path.basename(path), chunks, owner=vector_store.PUBLIC_OWNER
    )
    print(f"Added {os.path.basename(path)} as a public sample: {count} chunks (doc_id {doc_id})")


if __name__ == "__main__":
    main()