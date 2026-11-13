"""
app.py
------
Flask web app tying together document_loader, embeddings, vector_store,
and rag into a browser-usable interface. Each visitor gets an anonymous,
private workspace (via a signed cookie); sample documents are shared.
"""

import os
import uuid
import tempfile
from datetime import timedelta
from flask import Flask, request, jsonify, render_template, session
from flask_limiter import Limiter
from werkzeug.utils import secure_filename

import document_loader
import vector_store
import rag

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_MB = 10
MAX_CHUNKS_PER_DOC = 400      # roughly 90 pages of text; caps embedding cost per upload
MAX_QUESTION_CHARS = 1000     # caps the size of each Claude request
MAX_DOCS_PER_VISITOR = 5      # caps how much one visitor can store

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# The secret key signs the visitor cookie. Set SECRET_KEY in your environment
# (locally in .env, and on Render). Without it a random key is generated on
# every start, which means every visitor loses access to their uploads
# whenever the app restarts or redeploys.
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32)
if not os.environ.get("SECRET_KEY"):
    app.logger.warning("SECRET_KEY is not set: visitor sessions will reset on every restart.")

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE") == "1",   # set COOKIE_SECURE=1 on Render (HTTPS)
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
)


def current_owner():
    """Anonymous per-visitor ID, created on first use and kept in a signed cookie."""
    owner = session.get("owner")
    if not owner:
        owner = uuid.uuid4().hex
        session["owner"] = owner
        session.permanent = True
    return owner


def client_ip():
    """
    Work out the real visitor IP for rate limiting.

    On Render, traffic goes Client -> Cloudflare -> Render proxy -> this app, so
    request.remote_addr is a private proxy address, and the last entry of
    X-Forwarded-For is Cloudflare's own IP. Preference order:
      1. CF-Connecting-IP / True-Client-IP (set by Cloudflare itself)
      2. the second-to-last X-Forwarded-For entry (the one Cloudflare added)
      3. remote_addr (this is what you get when running locally)
    """
    ip = request.headers.get("CF-Connecting-IP") or request.headers.get("True-Client-IP")
    if ip:
        return ip.strip()

    forwarded = [p.strip() for p in request.headers.get("X-Forwarded-For", "").split(",") if p.strip()]
    if len(forwarded) >= 2:
        return forwarded[-2]

    return request.remote_addr or "unknown"


# In-memory counters: fine for one gunicorn worker. They reset whenever the
# app restarts or redeploys, and are not shared if you ever add more workers.
limiter = Limiter(client_ip, app=app, storage_uri="memory://")


@app.errorhandler(429)
def too_many_requests(e):
    return jsonify({"error": "Too many requests. Please wait a bit and try again."}), 429


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": f"File too large (limit {MAX_UPLOAD_MB} MB)."}), 413


def _allowed(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
@limiter.limit("4 per hour;12 per day")
def upload():
    owner = current_owner()

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    mine = [d for d in vector_store.list_documents(owner) if not d["public"]]
    if len(mine) >= MAX_DOCS_PER_VISITOR:
        return jsonify({"error": f"You can keep up to {MAX_DOCS_PER_VISITOR} documents. Remove one to add another."}), 400

    filename = secure_filename(file.filename)
    doc_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename)[1].lower()

    # Save to a temp file just long enough to extract text, then delete it
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        text = document_loader.extract_text(tmp_path)
        chunks = document_loader.chunk_text(text)

        if not chunks:
            return jsonify({"error": "No extractable text found. Scanned or image-only files aren't supported (no OCR)."}), 400

        if len(chunks) > MAX_CHUNKS_PER_DOC:
            return jsonify({"error": f"This document is too long for the demo (limit: about {MAX_CHUNKS_PER_DOC} chunks, roughly 90 pages of text)."}), 400

        count = vector_store.add_chunks(doc_id, filename, chunks, owner=owner)
    except Exception:
        app.logger.exception("Failed to process upload %s", filename)
        return jsonify({"error": "Could not process this file."}), 500
    finally:
        os.remove(tmp_path)  # always clean up, even if something above failed

    return jsonify({
        "doc_id": doc_id,
        "filename": filename,
        "chunks_indexed": count,
    })


@app.route("/documents", methods=["GET"])
@limiter.limit("60 per minute")
def documents():
    return jsonify(vector_store.list_documents(owner=current_owner()))


@app.route("/documents/<doc_id>/delete", methods=["POST"])
@limiter.limit("20 per hour")
def delete_document(doc_id):
    vector_store.delete_document(doc_id, owner=current_owner())
    return jsonify({"status": "deleted", "doc_id": doc_id})


@app.route("/query", methods=["POST"])
@limiter.limit("8 per minute;60 per day")
def query():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if len(question) > MAX_QUESTION_CHARS:
        return jsonify({"error": f"Question too long (limit {MAX_QUESTION_CHARS} characters)."}), 400

    try:
        result = rag.answer_question(question, owner=current_owner())
    except Exception:
        app.logger.exception("Query failed")
        return jsonify({"error": "The answer service is temporarily unavailable. Please try again."}), 502

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)