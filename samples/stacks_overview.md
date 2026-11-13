# Stacks: how it was built and how it was evaluated

## What Stacks does
Stacks is a retrieval-augmented generation (RAG) app. A visitor uploads documents (PDF, DOCX, TXT or Markdown) and asks questions in plain English. The answer is written by Claude using only passages retrieved from the visitor's own documents, with numbered citations such as [1] and [2] that point to the source passages.

## How a question is answered
When a document is uploaded, its text is extracted, split into chunks, converted into embedding vectors, and stored in a vector database. When a question is asked, the question is embedded the same way, the five most similar chunks are retrieved by cosine similarity, and those chunks are sent to Claude together with the question. Claude answers from those chunks and cites them.

## Technology stack
Backend: Python and Flask. Embeddings: Voyage AI (voyage-4-large, 1,024-dimensional vectors). Vector database: Chroma Cloud with cosine similarity. Answer generation: Claude (claude-sonnet-5) through the Anthropic API. Frontend: plain HTML, CSS and JavaScript. Hosting: Render.

## How documents are chunked
Text is split into chunks of about 800 characters with 120 characters of overlap between neighbouring chunks. Chunk breaks prefer sentence endings, then word boundaries. Overlap reduces the chance that an answer is cut in half, but it does not eliminate it, as the error analysis below shows.

## How hallucination is reduced
Claude is instructed to answer only from the retrieved context, to say so when the context is not enough, and to cite its sources. Retrieval is limited to the visitor's own documents plus shared samples. This reduces made-up answers but does not remove them, and the "not in your documents" behaviour has not yet been formally measured.

## Privacy and usage limits
Uploaded files are not stored: each file is read from a temporary location and deleted once its text is extracted. The extracted text chunks are stored in Chroma Cloud under an anonymous visitor ID kept in a signed cookie, so each visitor sees only their own uploads plus the shared samples. Limits: 5 documents per visitor, 10 MB per file, about 90 pages per document, 8 questions per minute and 60 per day, 4 uploads per hour and 12 per day per IP address.

## Evaluation benchmark
The benchmark has 64 questions over six interview-preparation guides (Flask, HTML, Next.js, React, CSS and Tailwind, Node and Express) totalling 584 chunks. 56 questions are answerable and 8 are deliberately unanswerable. The author wrote the questions from the guides' text. A retrieval counts as a hit when a retrieved chunk from the correct guide contains all the answer keywords for that question.

## Baseline retrieval results
hit@1 was 78.6%, hit@3 was 98.2%, hit@5 was 100% and hit@10 was 100%. Mean reciprocal rank (MRR) was 0.879, and the correct guide was in the top five results for 100% of questions. Median retrieval latency was about 0.7 seconds, covering query embedding and vector search but not Claude's answer generation. By category, hit@1 was 88% for direct lookups, 90% for paraphrased questions, 62.5% for conceptual questions and 60% for cross-document questions (only 5 questions).

## What the error analysis found
Twelve questions did not have the correct chunk ranked first. For 9 of them, the top chunk was the neighbouring piece of the same passage, so the answer was split across a chunk boundary (inferred from chunk adjacency, not a full manual review). One had a second valid passage elsewhere in the same guide, one was in the same region of the guide, and one was a genuine miss where a code listing outranked the explanation. The strict keyword-in-one-chunk rule therefore understates retrieval quality. In the benchmark, the passage containing the answer was always among the five chunks sent to Claude.

## A bug found while building
Chroma Cloud returns at most 300 records per request and accepts at most 300 records per write. The catalog listing and the upload code originally ignored this, so with more than about 300 chunks, documents would have disappeared from the catalog and large uploads would have failed. Both now read and write in pages of 300.

## Known limitations and next steps
There is no OCR, so scanned or image-only PDFs are rejected. The benchmark is small (each question is worth 1.8 points) and was written by the author. Answer quality and abstention have not been measured yet. The mean top-1 similarity was 0.61 for answerable questions and 0.40 for unanswerable ones, but the ranges overlap, so a simple similarity threshold cannot cleanly separate them. Planned work: passage-level scoring that does not depend on chunk boundaries, an answer-quality evaluation, and chunking improvements such as avoiding mid-word chunk starts.

## Questions you can ask about Stacks
What embedding model and vector database does Stacks use? How does Stacks reduce hallucination? What was the hit@5 in the evaluation? Why did some questions miss at rank 1? What are the known limitations? How are visitors' uploads kept private?