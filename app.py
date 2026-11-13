"""
app.py
------
Flask web app tying together document_loader, embeddings, vector_store,
and rag into a browser-usable interface.
"""

import os
import uuid
import tempfile
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from werkzeug.utils import secure_filename

import document_loader
import vector_store
import rag

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_MB = 10
MAX_CHUNKS_PER_DOC = 400      # roughly 90 pages of text; caps embedding cost per upload
MAX_QUESTION_CHARS = 1000     # caps the size of each Claude request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


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
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

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

        count = vector_store.add_chunks(doc_id, filename, chunks)
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
    return jsonify(vector_store.list_documents())


@app.route("/documents/<doc_id>/delete", methods=["POST"])
@limiter.limit("20 per hour")
def delete_document(doc_id):
    vector_store.delete_document(doc_id)
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
        result = rag.answer_question(question)
    except Exception:
        app.logger.exception("Query failed")
        return jsonify({"error": "The answer service is temporarily unavailable. Please try again."}), 502

    return jsonify(result)


# TEMPORARY: delete this route after you've verified the IP detection on Render.
@app.route("/_ip")
def _ip():
    return jsonify({
        "seen_by_limiter": client_ip(),
        "cf_connecting_ip": request.headers.get("CF-Connecting-IP"),
        "true_client_ip": request.headers.get("True-Client-IP"),
        "x_forwarded_for": request.headers.get("X-Forwarded-For"),
        "remote_addr": request.remote_addr,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)