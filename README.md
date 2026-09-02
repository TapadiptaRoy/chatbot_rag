# Stacks — Personal Knowledge Base (RAG)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-black)
![Status](https://img.shields.io/badge/status-live-brightgreen)

A retrieval-augmented generation app that lets you upload your own documents (PDF, DOCX, TXT, MD) and ask questions about them in plain English — with answers grounded in what you've actually uploaded, not general internet knowledge.

**🔗 Live demo:** [stacks-chatbot-rag.onrender.com](https://stacks-chatbot-rag.onrender.com)

---

## How it works

1. **Extract** — uploaded files are parsed into plain text (`pypdf` for PDFs, `docx2txt` for Word docs).
2. **Chunk** — text is split into overlapping ~800-character pieces, so retrieval can find specific relevant passages rather than whole documents.
3. **Embed** — each chunk is converted into a 1024-dimension vector using Voyage AI's embedding model, capturing its semantic meaning.
4. **Store & search** — vectors are stored in Chroma Cloud. When you ask a question, it's embedded the same way, and cosine similarity search finds the most relevant chunks.
5. **Generate** — the retrieved chunks are passed to Claude (Anthropic) along with the question, with instructions to answer *only* using that context and cite sources — this is what grounds the answer and reduces hallucination.

---

## Architecture

├── app.py Flask routes (upload, query, list/delete documents)
├── config.py Loads API keys/secrets from environment variables
├── document_loader.py Text extraction (PDF/DOCX/TXT/MD) + chunking
├── embeddings.py Wraps the Voyage AI embeddings API
├── vector_store.py Wraps Chroma Cloud (add/query/list/delete)
├── rag.py Retrieval + prompt construction + Claude generation
├── templates/index.html Frontend UI
├── static/style.css, script.js Styling and browser-side logic
├── requirements.txt
└── Procfile Tells the host how to run the app (gunicorn)


Each backend module has a single, narrow responsibility and no knowledge of how the others work internally — `embeddings.py` doesn't know where text comes from, `vector_store.py` doesn't know how embeddings are computed, etc.

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python, Flask |
| Embeddings | Voyage AI (`voyage-4-large`) |
| Vector database | Chroma Cloud |
| Generation | Claude (Anthropic API) |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Render |

---

## Running it locally

```bash
git clone https://github.com/TapadiptaRoy/chatbot_rag.git
cd chatbot_rag
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example` for the required keys):

VOYAGE_API_KEY=
CHROMA_API_KEY=
CHROMA_TENANT=
CHROMA_DATABASE=
ANTHROPIC_API_KEY=


```bash
python app.py
```

Open `http://localhost:5000`.

---

## Design decisions worth knowing about

- **Uploaded files aren't stored permanently.** Files are processed in a temp location just long enough to extract text, then discarded — no dependency on persistent local disk.
- **Embeddings are API-based, not local.** Avoids the RAM overhead of running an embedding model on a free-tier host.
- **Chunk size (800 chars) / overlap (120 chars)** are tunable in `document_loader.py`.

## Known limitations

- PDF extraction has no OCR — scanned/image-only PDFs won't extract text.
- Single shared knowledge base — no user accounts or multi-tenancy.
- No conversation history — each question is answered independently
