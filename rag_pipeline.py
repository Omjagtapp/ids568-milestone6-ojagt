"""
RAG Pipeline: Document Ingestion → Chunking → Embedding → FAISS → Retrieval → Generation

Documents are loaded from docs/*.txt at runtime.
Queries and ground-truth are loaded from config/queries.json.

Usage:
    python3 rag_pipeline.py                              # all three strategies
    python3 rag_pipeline.py --strategy recursive         # single strategy
    python3 rag_pipeline.py --docs_dir my_docs/         # custom document folder
    python3 rag_pipeline.py --queries config/q.json     # custom query file
    python3 rag_pipeline.py --chunk_size 256 --overlap 32
    python3 rag_pipeline.py --llm_model llama3:8b-instruct
"""

import argparse
import json
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents(docs_dir: str) -> List[Dict]:
    """
    Load all .txt files from docs_dir.
    Expected format (optional):
        Title: <title on first line>
        <blank line>
        <body text>
    Falls back to using the filename as the title.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"docs_dir not found: {docs_path.resolve()}")

    documents = []
    for txt_file in sorted(docs_path.glob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()

        # Parse optional "Title: ..." first line
        if lines and lines[0].startswith("Title:"):
            title = lines[0].removeprefix("Title:").strip()
            body_lines = lines[1:]
        else:
            title = txt_file.stem.replace("_", " ").title()
            body_lines = lines

        text = "\n".join(body_lines).strip()
        doc_id = txt_file.stem.split("_")[0] + "_" + txt_file.stem.split("_")[1] \
            if "_" in txt_file.stem else txt_file.stem

        documents.append({"id": doc_id, "title": title, "text": text, "source": str(txt_file)})

    if not documents:
        raise ValueError(f"No .txt files found in {docs_path.resolve()}")

    print(f"Loaded {len(documents)} documents from {docs_path.resolve()}")
    return documents


# ---------------------------------------------------------------------------
# Query + ground-truth loading
# ---------------------------------------------------------------------------

def load_queries(queries_path: str) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """
    Load queries from a JSON file.
    Expected schema:
        {"queries": [{"id": "q01", "text": "...", "relevant_docs": ["doc_04", ...]}, ...]}
    Returns (query_list, ground_truth_map).
    """
    path = Path(queries_path)
    if not path.exists():
        raise FileNotFoundError(f"queries file not found: {path.resolve()}")

    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data.get("queries", [])
    ground_truth: Dict[str, List[str]] = {
        q["id"]: q.get("relevant_docs", []) for q in queries
    }
    print(f"Loaded {len(queries)} queries from {path.resolve()}")
    return queries, ground_truth


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int


def fixed_size_chunking(doc: Dict, chunk_size: int, overlap: int) -> List[Chunk]:
    text = doc["text"]
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(Chunk(
            chunk_id=f"{doc['id']}_fixed_{idx}",
            doc_id=doc["id"],
            text=text[start:end],
            start_char=start,
            end_char=end,
        ))
        if end == len(text):
            break
        start += chunk_size - overlap
        idx += 1
    return chunks


def recursive_chunking(doc: Dict, chunk_size: int, overlap: int) -> List[Chunk]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    text = doc["text"]

    def _split(t: str, sep_idx: int) -> List[str]:
        if len(t) <= chunk_size or sep_idx >= len(separators):
            return [t]
        sep = separators[sep_idx]
        parts = t.split(sep) if sep else list(t)
        results, current = [], ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    results.append(current)
                if len(part) > chunk_size:
                    results.extend(_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part
        if current:
            results.append(current)
        overlapped = []
        for i, c in enumerate(results):
            overlapped.append(c + results[i + 1][:overlap] if i + 1 < len(results) else c)
        return overlapped

    raw = _split(text, 0)
    chunks, pos = [], 0
    for idx, ct in enumerate(raw):
        start = text.find(ct[:min(20, len(ct))], pos)
        start = pos if start == -1 else start
        end = start + len(ct)
        chunks.append(Chunk(
            chunk_id=f"{doc['id']}_recursive_{idx}",
            doc_id=doc["id"],
            text=ct,
            start_char=start,
            end_char=end,
        ))
        pos = max(pos, end - overlap)
    return chunks


def sentence_chunking(doc: Dict, max_sentences: int = 4, overlap_sentences: int = 1) -> List[Chunk]:
    text = doc["text"]
    ends = [m.end() for m in re.finditer(r'(?<=[.!?])\s+', text)]
    bounds = [0] + ends + [len(text)]
    sentences = [text[bounds[i]:bounds[i + 1]].strip()
                 for i in range(len(bounds) - 1) if bounds[i] < bounds[i + 1]]
    chunks, i, idx = [], 0, 0
    while i < len(sentences):
        window = sentences[i: i + max_sentences]
        ct = " ".join(window)
        start = text.find(window[0][:min(20, len(window[0]))]) if window else 0
        chunks.append(Chunk(
            chunk_id=f"{doc['id']}_sentence_{idx}",
            doc_id=doc["id"],
            text=ct,
            start_char=max(0, start),
            end_char=min(len(text), start + len(ct)),
        ))
        i += max_sentences - overlap_sentences
        idx += 1
    return chunks


def chunk_documents(
    docs: List[Dict],
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> Tuple[List[Chunk], float]:
    t0 = time.perf_counter()
    all_chunks: List[Chunk] = []
    for doc in docs:
        if strategy == "fixed":
            all_chunks.extend(fixed_size_chunking(doc, chunk_size, overlap))
        elif strategy == "sentence":
            all_chunks.extend(sentence_chunking(doc))
        else:
            all_chunks.extend(recursive_chunking(doc, chunk_size, overlap))
    return all_chunks, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str], model_name: str) -> Tuple[np.ndarray, float]:
    from sentence_transformers import SentenceTransformer
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embs, dtype="float32"), (time.perf_counter() - t0) * 1000


def embed_query(query: str, model_name: str) -> Tuple[np.ndarray, float]:
    from sentence_transformers import SentenceTransformer
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    emb = model.encode([query], normalize_embeddings=True)
    return np.array(emb, dtype="float32"), (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_index(embeddings: np.ndarray) -> Tuple[Any, float]:
    import faiss
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, (time.perf_counter() - t0) * 1000


def retrieve(
    query_emb: np.ndarray,
    index: Any,
    chunks: List[Chunk],
    k: int,
) -> Tuple[List[Chunk], List[float], float]:
    t0 = time.perf_counter()
    scores, indices = index.search(query_emb, k)
    latency = (time.perf_counter() - t0) * 1000
    hits = [chunks[i] for i in indices[0] if i < len(chunks)]
    return hits, scores[0].tolist()[:len(hits)], latency


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    context_chunks: List[Chunk],
    llm_model: str,
) -> Tuple[str, float]:
    context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in context_chunks)
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    t0 = time.perf_counter()
    try:
        import ollama
        resp = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
        answer = resp["message"]["content"].strip()
    except Exception as exc:
        answer = f"[Ollama unavailable – {exc}] Context: {context[:300]}…"
    return answer, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Groundedness scoring (lexical overlap)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "and",
    "or", "it", "that", "this", "for", "with", "be", "as", "at", "by",
    "from", "on", "not", "but",
}


def groundedness_score(answer: str, context_chunks: List[Chunk]) -> float:
    ctx = " ".join(c.text for c in context_chunks).lower()
    tokens = [t for t in re.findall(r'\b\w+\b', answer.lower())
              if t not in _STOP_WORDS and len(t) > 2]
    if not tokens:
        return 0.0
    return round(sum(1 for t in tokens if t in ctx) / len(tokens), 3)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: List[Chunk], relevant: List[str], k: int) -> float:
    relevant_set = set(relevant)
    hits = sum(1 for c in retrieved[:k] if c.doc_id in relevant_set)
    return round(hits / k, 3)


def recall_at_k(retrieved: List[Chunk], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for c in retrieved[:k] if c.doc_id in relevant_set)
    return round(hits / len(relevant_set), 3)


def mrr(retrieved: List[Chunk], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.doc_id in relevant_set:
            return round(1.0 / rank, 3)
    return 0.0


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    query_id: str
    query: str
    strategy: str
    chunk_size: int
    retrieved_docs: List[str]
    scores: List[float]
    answer: str
    groundedness: float
    precision_at_3: float
    recall_at_3: float
    precision_at_5: float
    recall_at_5: float
    mrr_score: float
    latency_ingestion_ms: float
    latency_chunking_ms: float
    latency_embedding_ms: float
    latency_index_build_ms: float
    latency_query_embed_ms: float
    latency_retrieval_ms: float
    latency_generation_ms: float

    @property
    def total_latency_ms(self) -> float:
        return (
            self.latency_chunking_ms + self.latency_embedding_ms
            + self.latency_index_build_ms + self.latency_query_embed_ms
            + self.latency_retrieval_ms + self.latency_generation_ms
        )


class RAGPipeline:
    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 512,
        overlap: int = 64,
        model_name: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama3.1:8b",
        k: int = 5,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model_name = model_name
        self.llm_model = llm_model
        self.k = k
        self.chunks: List[Chunk] = []
        self.index = None
        self._lat_ingestion = 0.0
        self._lat_chunking = 0.0
        self._lat_embedding = 0.0
        self._lat_index = 0.0
        self._ground_truth: Dict[str, List[str]] = {}

    def ingest(self, docs: List[Dict]) -> None:
        t0 = time.perf_counter()
        _ = [d["text"] for d in docs]          # simulate ingestion work
        self._lat_ingestion = (time.perf_counter() - t0) * 1000

        self.chunks, self._lat_chunking = chunk_documents(
            docs, self.strategy, self.chunk_size, self.overlap
        )
        embeddings, self._lat_embedding = embed_texts(
            [c.text for c in self.chunks], self.model_name
        )
        self.index, self._lat_index = build_index(embeddings)

    def set_ground_truth(self, ground_truth: Dict[str, List[str]]) -> None:
        self._ground_truth = ground_truth

    def query(self, query_text: str, query_id: str = "q00") -> PipelineResult:
        relevant = self._ground_truth.get(query_id, [])
        q_emb, lat_qemb = embed_query(query_text, self.model_name)
        retrieved, scores, lat_ret = retrieve(q_emb, self.index, self.chunks, self.k)
        answer, lat_gen = generate_answer(query_text, retrieved[:3], self.llm_model)
        gs = groundedness_score(answer, retrieved[:3])
        return PipelineResult(
            query_id=query_id,
            query=query_text,
            strategy=self.strategy,
            chunk_size=self.chunk_size,
            retrieved_docs=[c.doc_id for c in retrieved],
            scores=scores,
            answer=answer,
            groundedness=gs,
            precision_at_3=precision_at_k(retrieved, relevant, 3),
            recall_at_3=recall_at_k(retrieved, relevant, 3),
            precision_at_5=precision_at_k(retrieved, relevant, 5),
            recall_at_5=recall_at_k(retrieved, relevant, 5),
            mrr_score=mrr(retrieved, relevant),
            latency_ingestion_ms=self._lat_ingestion,
            latency_chunking_ms=self._lat_chunking,
            latency_embedding_ms=self._lat_embedding,
            latency_index_build_ms=self._lat_index,
            latency_query_embed_ms=lat_qemb,
            latency_retrieval_ms=lat_ret,
            latency_generation_ms=lat_gen,
        )


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    docs: List[Dict],
    queries: List[Dict],
    ground_truth: Dict[str, List[str]],
    strategy: str,
    chunk_size: int,
    overlap: int,
    model_name: str,
    llm_model: str,
    k: int,
) -> List[PipelineResult]:
    print(f"\n=== RAG Evaluation | strategy={strategy} chunk_size={chunk_size} overlap={overlap} ===")
    pipeline = RAGPipeline(
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        llm_model=llm_model,
        k=k,
    )
    pipeline.set_ground_truth(ground_truth)

    print("Ingesting documents…")
    pipeline.ingest(docs)
    print(f"  Chunks : {len(pipeline.chunks)}")
    print(f"  Chunking latency : {pipeline._lat_chunking:.1f} ms")
    print(f"  Embedding latency: {pipeline._lat_embedding:.1f} ms")
    print(f"  Index build      : {pipeline._lat_index:.1f} ms")

    results = []
    for q in queries:
        r = pipeline.query(q["text"], q["id"])
        results.append(r)
        print(
            f"  [{r.query_id}] P@3={r.precision_at_3} R@3={r.recall_at_3} "
            f"MRR={r.mrr_score} GS={r.groundedness} gen={r.latency_generation_ms:.0f}ms"
        )

    n = len(results)
    print(
        f"\n  Avg P@3={sum(r.precision_at_3 for r in results)/n:.3f}  "
        f"R@3={sum(r.recall_at_3 for r in results)/n:.3f}  "
        f"MRR={sum(r.mrr_score for r in results)/n:.3f}  "
        f"GS={sum(r.groundedness for r in results)/n:.3f}  "
        f"total_lat={sum(r.total_latency_ms for r in results)/n:.0f}ms"
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG Pipeline Evaluation")
    p.add_argument("--docs_dir", default="docs", help="Directory of .txt documents")
    p.add_argument("--queries", default="config/queries.json", help="Path to queries JSON")
    p.add_argument(
        "--strategy", default=None,
        choices=["fixed", "recursive", "sentence"],
        help="Single chunking strategy (default: run all three)",
    )
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--model_name", default="all-MiniLM-L6-v2")
    p.add_argument("--llm_model", default="llama3.1:8b")
    p.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    p.add_argument("--output", default="rag_results.json", help="Path to write results JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    docs = load_documents(args.docs_dir)
    queries, ground_truth = load_queries(args.queries)

    strategies = [args.strategy] if args.strategy else ["fixed", "recursive", "sentence"]

    all_results: List[PipelineResult] = []
    for strat in strategies:
        results = run_evaluation(
            docs=docs,
            queries=queries,
            ground_truth=ground_truth,
            strategy=strat,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            model_name=args.model_name,
            llm_model=args.llm_model,
            k=args.k,
        )
        all_results.extend(results)

    output = [
        {
            "query_id": r.query_id,
            "query": r.query,
            "strategy": r.strategy,
            "chunk_size": r.chunk_size,
            "retrieved_docs": r.retrieved_docs,
            "answer_snippet": r.answer[:300],
            "groundedness": r.groundedness,
            "precision_at_3": r.precision_at_3,
            "recall_at_3": r.recall_at_3,
            "precision_at_5": r.precision_at_5,
            "recall_at_5": r.recall_at_5,
            "mrr": r.mrr_score,
            "latency_ingestion_ms": round(r.latency_ingestion_ms, 2),
            "latency_chunking_ms": round(r.latency_chunking_ms, 2),
            "latency_embedding_ms": round(r.latency_embedding_ms, 2),
            "latency_index_build_ms": round(r.latency_index_build_ms, 2),
            "latency_query_embed_ms": round(r.latency_query_embed_ms, 2),
            "latency_retrieval_ms": round(r.latency_retrieval_ms, 2),
            "latency_generation_ms": round(r.latency_generation_ms, 2),
            "total_latency_ms": round(r.total_latency_ms, 2),
        }
        for r in all_results
    ]

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
