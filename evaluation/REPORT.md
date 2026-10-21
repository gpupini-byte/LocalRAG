# LocalRAG Evaluation Report

- Dataset: LocalRAG synthetic smoke benchmark (v1); 28 questions, 3 synthetic text documents.
- Labels: `synthetic_fixture_labels`. These are fixture labels, not reviews of external documents.
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions).
- Runtime: Python 3.12.10; package versions `{'PyMuPDF': '1.28.2', 'sentence-transformers': '5.7.0', 'FAISS': '1.15.1', 'NumPy': '2.5.4'}`.
- Index: FAISS IndexFlatIP over normalized float32 vectors.
- Score threshold: None; score is cosine similarity, not a probability.
- Model initialization measured separately: 1.4778 seconds (may be warm or cached; this run does not claim a cold start).
- Ollama answer generation enabled: False; human review performed: no.

## Results

| Chunk size / overlap | Chunks | Dimension | Recall@1 | Recall@3 | Recall@5 | Precision@1 | Precision@3 | Precision@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200/40 | 6 | 384 | 0.6250 | 1.0000 | 1.0000 | 0.6250 | 0.3333 | 0.2000 | 0.7986 |
| 400/80 | 3 | 384 | 0.3333 | 0.8750 | 0.8750 | 0.3333 | 0.2917 | 0.1750 | 0.5903 |

- 200/40 query embedding: n=28, mean=0.0056s, p50=0.0056s, p95=0.0063s.
- 200/40 FAISS retrieval: n=28, mean=0.0001s, p50=0.0001s, p95=0.0001s.
- 200/40 Ollama generation: n=0, mean=n/a, p50=n/a, p95=n/a.
- 200/40 total measured query: n=28, mean=0.0057s, p50=0.0056s, p95=0.0064s.
- 200/40 ingestion: document load=0.0008s, chunking=0.0010s, batch document embedding=0.1227s.
- 400/80 query embedding: n=28, mean=0.0060s, p50=0.0055s, p95=0.0078s.
- 400/80 FAISS retrieval: n=28, mean=0.0001s, p50=0.0001s, p95=0.0001s.
- 400/80 Ollama generation: n=0, mean=n/a, p50=n/a, p95=n/a.
- 400/80 total measured query: n=28, mean=0.0060s, p50=0.0056s, p95=0.0079s.
- 400/80 ingestion: document load=0.0008s, chunking=0.0012s, batch document embedding=0.0579s.

Retrieval metrics are averaged over queries with non-empty gold passage annotations only. Unsupported and missing-label queries are counted separately. Precision@k divides by k, including when fewer than k passage results exist. MRR uses the first relevant passage. Latency p95 is emitted only for at least 20 samples.

Answerability predictions are based on `retrieval produced at least one chunk`.

## Answerability and citations

- 200/40: 4 unsupported non-abstentions, 0 supported abstentions, 0 correct abstentions. Citation validity indicates label membership only; it is not a grounding judgment.
- 400/80: 4 unsupported non-abstentions, 0 supported abstentions, 0 correct abstentions. Citation validity indicates label membership only; it is not a grounding judgment.

## Method and limits

The benchmark corpus is synthetic and included in this repository. Stable passage anchors are projected onto chunks by word-span overlap, so gold labels survive both chunk boundaries. Both chunk configurations use the same documents and embedding model. Query embedding, FAISS search, optional generation, and total measured query time are reported separately; model initialization is separate from query timings. Process startup and Python imports are excluded.

Similarity thresholds trade false positives (irrelevant chunks pass) against false negatives (relevant chunks are filtered). The default threshold is disabled because this small synthetic set does not justify a general cutoff. Optional generation requires a locally running Ollama model. No human answer review is implied by these automated metrics.
