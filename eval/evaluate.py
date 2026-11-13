"""
eval/evaluate.py
-----------------
Retrieval evaluation for the Stacks RAG app.

What it does
  1. Ingests the PDFs in eval/docs/ into a SEPARATE Chroma collection
     (eval_knowledge_base), never your live one.
  2. Runs every question in eval/questions.json through vector_store.query().
  3. Scores retrieval: hit@k, MRR, doc-level hit@5, per-category breakdown,
     top-1 similarity for answerable vs unanswerable questions, and latency.
  4. Saves a JSON results file in eval/results/ so runs can be compared.

Usage (from the project root, venv active)
  python eval/evaluate.py --reset          # first run, or after changing chunking
  python eval/evaluate.py                  # reuse already-ingested collection
  python eval/evaluate.py --reset --chunk-size 500 --overlap 80 --label cs500

A "hit" for an answerable question means: a retrieved chunk comes from one of
the question's source documents AND contains every answer keyword.
"""

import os
import sys

# MUST happen before vector_store is imported: it reads this at import time.
EVAL_COLLECTION = "eval_knowledge_base"
os.environ["CHROMA_COLLECTION"] = EVAL_COLLECTION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse
import glob
import json
import re
import statistics
import time
from datetime import datetime

import document_loader
import vector_store

if vector_store.COLLECTION_NAME == "knowledge_base":
    sys.exit("Refusing to run: vector_store is pointing at the LIVE collection.")

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(EVAL_DIR, "docs")
QUESTIONS_PATH = os.path.join(EVAL_DIR, "questions.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
KS = (1, 3, 5, 10)


# ---------- helpers ----------

def norm(text):
    """Lowercase, straighten curly quotes, collapse whitespace."""
    text = (text.replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))
    return re.sub(r"\s+", " ", text).lower()


def name_key(name):
    """Compare file names loosely (ignores case, spaces, underscores, symbols)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def retry(fn, *args, attempts=5, **kwargs):
    """Retry on transient errors (rate limits, network blips) with backoff."""
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i == attempts - 1:
                raise
            wait = 2 ** (i + 1)
            print(f"  ! {type(e).__name__}: {str(e)[:120]} -- retrying in {wait}s")
            time.sleep(wait)


def pct(x):
    return f"{100 * x:5.1f}%"


# ---------- ingestion ----------

def reset_collection():
    try:
        vector_store.client.delete_collection(name=vector_store.COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet
    vector_store.collection = vector_store.client.get_or_create_collection(
        name=vector_store.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest(chunk_size, overlap):
    pdfs = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    if not pdfs:
        sys.exit(f"No PDFs found in {DOCS_DIR}. Put the six guides there first.")

    print(f"Ingesting {len(pdfs)} PDFs (chunk_size={chunk_size}, overlap={overlap})")
    ingested = []
    for i, path in enumerate(pdfs):
        name = os.path.basename(path)
        text = document_loader.extract_text(path)
        chunks = document_loader.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            print(f"  - {name}: no extractable text, skipped")
            continue
        retry(vector_store.add_chunks, f"eval_{i}", name, chunks)
        print(f"  - {name}: {len(chunks)} chunks")
        ingested.append(name)
    return ingested


# ---------- evaluation ----------

def evaluate(questions, max_k):
    rows = []
    for n, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        hits = retry(vector_store.query, q["question"], top_k=max_k)
        latency = time.perf_counter() - t0

        row = {
            "id": q["id"],
            "category": q["category"],
            "answerable": q["answerable"],
            "question": q["question"],
            "latency_s": round(latency, 3),
            "top1_score": round(hits[0]["score"], 4) if hits else None,
            "top_sources": [h["source"] for h in hits[:5]],
            "top1_text": hits[0]["text"][:160] if hits else "",
            "first_doc_rank": None,
            "first_hit_rank": None,
        }

        if q["answerable"]:
            src_keys = {name_key(d) for d in q["source_docs"]}
            kws = [norm(k) for k in q["answer_keywords"]]
            for rank, h in enumerate(hits, 1):
                doc_ok = name_key(h["source"]) in src_keys
                if doc_ok and row["first_doc_rank"] is None:
                    row["first_doc_rank"] = rank
                if doc_ok and all(k in norm(h["text"]) for k in kws):
                    row["first_hit_rank"] = rank
                    break

        rows.append(row)
        if n % 10 == 0:
            print(f"  ...{n}/{len(questions)} questions done")
    return rows


def hit_at(rows, k):
    return sum(1 for r in rows if r["first_hit_rank"] and r["first_hit_rank"] <= k) / len(rows)


def doc_hit_at(rows, k):
    return sum(1 for r in rows if r["first_doc_rank"] and r["first_doc_rank"] <= k) / len(rows)


def mrr(rows):
    return sum(1 / r["first_hit_rank"] for r in rows if r["first_hit_rank"]) / len(rows)


def summarize(rows):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    lat = sorted(r["latency_s"] for r in rows)

    summary = {
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "hit_at": {str(k): round(hit_at(ans, k), 4) for k in KS},
        "doc_hit_at_5": round(doc_hit_at(ans, 5), 4),
        "mrr": round(mrr(ans), 4),
        "mean_top1_score_answerable": round(statistics.mean(r["top1_score"] for r in ans), 4),
        "mean_top1_score_unanswerable": (
            round(statistics.mean(r["top1_score"] for r in unans), 4) if unans else None
        ),
        "latency_mean_s": round(statistics.mean(lat), 3),
        "latency_median_s": round(statistics.median(lat), 3),
        "latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 3),
        "by_category": {},
    }
    for cat in sorted({r["category"] for r in ans}):
        sub = [r for r in ans if r["category"] == cat]
        summary["by_category"][cat] = {
            "n": len(sub),
            "hit_at_1": round(hit_at(sub, 1), 4),
            "hit_at_5": round(hit_at(sub, 5), 4),
            "mrr": round(mrr(sub), 4),
        }
    return summary


def print_report(summary, rows, n_chunks):
    print("\n" + "=" * 64)
    print(f"RETRIEVAL RESULTS  ({summary['n_answerable']} answerable questions, {n_chunks} chunks)")
    print("=" * 64)
    for k in KS:
        print(f"  hit@{k:<2}            {pct(summary['hit_at'][str(k)])}")
    print(f"  doc-level hit@5    {pct(summary['doc_hit_at_5'])}   (right document, ignoring keywords)")
    print(f"  MRR                {summary['mrr']:.3f}")

    print("\n  By category            n   hit@1   hit@5    MRR")
    for cat, s in summary["by_category"].items():
        print(f"  {cat:<20} {s['n']:>3}  {pct(s['hit_at_1'])}  {pct(s['hit_at_5'])}  {s['mrr']:.3f}")

    print("\n  Top-1 similarity (useful for choosing an abstention threshold)")
    print(f"    answerable questions    mean {summary['mean_top1_score_answerable']:.3f}")
    if summary["mean_top1_score_unanswerable"] is not None:
        print(f"    unanswerable questions  mean {summary['mean_top1_score_unanswerable']:.3f}")

    print(f"\n  Latency per query: mean {summary['latency_mean_s']}s, "
          f"median {summary['latency_median_s']}s, p95 {summary['latency_p95_s']}s")

    misses = [r for r in rows
              if r["answerable"] and (r["first_hit_rank"] is None or r["first_hit_rank"] > 5)]
    print(f"\n  Misses at k=5: {len(misses)}")
    for r in misses:
        where = ("doc found at rank %d" % r["first_doc_rank"]) if r["first_doc_rank"] else "doc not retrieved"
        print(f"    #{r['id']:<3} [{r['category']}] {r['question'][:68]}")
        print(f"          {where}; top sources: {[s[:12] for s in r['top_sources'][:3]]}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--reset", action="store_true",
                    help="wipe the eval collection and re-ingest (required after changing chunking)")
    ap.add_argument("--label", default="baseline")
    args = ap.parse_args()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Eval collection: {vector_store.COLLECTION_NAME}")
    ingested = None
    if args.reset:
        reset_collection()
        ingested = ingest(args.chunk_size, args.overlap)
    elif vector_store.collection.count() == 0:
        ingested = ingest(args.chunk_size, args.overlap)
    else:
        print(f"Reusing existing collection ({vector_store.collection.count()} chunks). "
              f"Use --reset if you changed chunking settings.")

    n_chunks = vector_store.collection.count()

    # Warn if a question points at a document that isn't in the collection.
    present = {name_key(d["source"]) for d in vector_store.list_documents()}
    missing = {d for q in questions for d in q["source_docs"] if name_key(d) not in present}
    if missing:
        print("WARNING: these source_docs from questions.json were not found in the collection:")
        for d in sorted(missing):
            print("   -", d)

    print(f"\nRunning {len(questions)} questions (top_k={args.max_k})...")
    rows = evaluate(questions, args.max_k)
    summary = summarize(rows)
    print_report(summary, rows, n_chunks)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"run_{stamp}_{args.label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "label": args.label,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "max_k": args.max_k,
                "n_chunks": n_chunks,
                "embedding_model": getattr(__import__("embeddings"), "MODEL_NAME", "unknown"),
                "timestamp": stamp,
            },
            "summary": summary,
            "rows": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    main()