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
from werkzeug.utils import secure_filename

import document_loader
import vector_store
import rag

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB per upload


def _allowed(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
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
    ext = os.path.splitext(filename)[1]

    # Save to a temp file just long enough to extract text, then delete it
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        text = document_loader.extract_text(tmp_path)
        chunks = document_loader.chunk_text(text)
        count = vector_store.add_chunks(doc_id, filename, chunks)
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {e}"}), 500
    finally:
        os.remove(tmp_path)  # always clean up, even if something above failed

    if count == 0:
        return jsonify({"error": "No extractable text found in this file."}), 400

    return jsonify({
        "doc_id": doc_id,
        "filename": filename,
        "chunks_indexed": count,
    })


@app.route("/documents", methods=["GET"])
def documents():
    return jsonify(vector_store.list_documents())


@app.route("/documents/<doc_id>/delete", methods=["POST"])
def delete_document(doc_id):
    vector_store.delete_document(doc_id)
    return jsonify({"status": "deleted", "doc_id": doc_id})


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    question = (data or {}).get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    result = rag.answer_question(question)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)