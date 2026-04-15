"""
RAG Pipeline: Document Ingestion → Chunking → Embedding → FAISS → Retrieval → Generation
Supports three chunking strategies, sentence-transformers embeddings, and Ollama LLM.
"""

import time
import json
import re
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Knowledge base – 15 short "documents" covering NLP / ML topics
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "id": "doc_01",
        "title": "Introduction to Transformers",
        "text": (
            "Transformers are a type of deep learning model introduced in the paper "
            "'Attention Is All You Need' (Vaswani et al., 2017). They rely entirely on "
            "self-attention mechanisms, discarding recurrence and convolutions. The "
            "architecture consists of an encoder and decoder, each built from stacked "
            "multi-head attention layers and position-wise feed-forward networks. "
            "Positional encodings are added to token embeddings so the model can "
            "exploit sequence order."
        ),
    },
    {
        "id": "doc_02",
        "title": "BERT and Bidirectional Pre-training",
        "text": (
            "BERT (Bidirectional Encoder Representations from Transformers) was "
            "introduced by Devlin et al. in 2018. Unlike GPT-style models that predict "
            "the next token left-to-right, BERT is pre-trained with a Masked Language "
            "Model (MLM) objective, masking 15 % of tokens and predicting them from "
            "context on both sides. A second objective, Next Sentence Prediction (NSP), "
            "teaches the model relationships between sentence pairs. Fine-tuning BERT "
            "achieves state-of-the-art results on many NLP benchmarks."
        ),
    },
    {
        "id": "doc_03",
        "title": "GPT and Autoregressive Language Models",
        "text": (
            "GPT (Generative Pre-trained Transformer) models generate text "
            "autoregressively: each token is conditioned on all preceding tokens. GPT-3 "
            "has 175 billion parameters and demonstrates impressive few-shot learning "
            "with no gradient updates at inference time. GPT-4 further improves "
            "reasoning and instruction following. The key training objective is causal "
            "language modelling (next-token prediction)."
        ),
    },
    {
        "id": "doc_04",
        "title": "Retrieval-Augmented Generation (RAG)",
        "text": (
            "Retrieval-Augmented Generation (RAG) combines a retrieval component with a "
            "generative model. Given a query, a retriever fetches relevant passages from "
            "an external corpus; these passages are concatenated with the query as "
            "context for the generator. RAG reduces hallucination because the model can "
            "ground its answer in retrieved evidence. It also allows the knowledge base "
            "to be updated without retraining the LLM."
        ),
    },
    {
        "id": "doc_05",
        "title": "Vector Databases and Approximate Nearest Neighbour Search",
        "text": (
            "Vector databases store high-dimensional embeddings and support efficient "
            "similarity search. FAISS (Facebook AI Similarity Search) provides exact and "
            "approximate nearest-neighbour (ANN) methods including Flat (brute-force), "
            "IVF (inverted file), and HNSW. Indexing trade-offs balance memory, build "
            "time, and query latency. Cosine similarity and inner-product search are "
            "common metrics for dense retrieval."
        ),
    },
    {
        "id": "doc_06",
        "title": "Sentence Transformers and Dense Retrieval",
        "text": (
            "Sentence Transformers (SBERT) fine-tune transformer models using siamese "
            "and triplet networks so that semantically similar sentences map to nearby "
            "points in embedding space. The model 'all-MiniLM-L6-v2' produces 384-dim "
            "embeddings and runs efficiently on CPU. Dense retrieval with sentence "
            "embeddings outperforms BM25 on open-domain QA tasks, especially for "
            "paraphrase-style questions."
        ),
    },
    {
        "id": "doc_07",
        "title": "Chunking Strategies for RAG",
        "text": (
            "Document chunking splits long texts into smaller passages that fit within "
            "the context window of the retriever and generator. Fixed-size chunking "
            "splits by a fixed token or character count with optional overlap. Recursive "
            "chunking uses a hierarchy of separators (paragraphs → sentences → words). "
            "Sentence-based chunking preserves linguistic boundaries. Chunk size (256–"
            "1024 tokens) and overlap (10–20 %) are key hyperparameters."
        ),
    },
    {
        "id": "doc_08",
        "title": "Evaluation Metrics for Retrieval",
        "text": (
            "Common retrieval evaluation metrics include Precision@k (fraction of top-k "
            "retrieved documents that are relevant), Recall@k (fraction of relevant "
            "documents retrieved in top-k), and Mean Reciprocal Rank (MRR, the average "
            "reciprocal rank of the first relevant result). NDCG@k weights relevance by "
            "position. Latency (ms per query) and throughput (queries/sec) measure "
            "operational efficiency."
        ),
    },
    {
        "id": "doc_09",
        "title": "Hallucination in Large Language Models",
        "text": (
            "Hallucination refers to LLM outputs that are factually incorrect or "
            "unsupported by the input context. Intrinsic hallucination contradicts the "
            "source, while extrinsic hallucination introduces new unverifiable claims. "
            "RAG reduces hallucination by providing explicit grounding passages. "
            "Groundedness scoring checks whether each claim in the response can be "
            "attributed to retrieved text using NLI or lexical overlap methods."
        ),
    },
    {
        "id": "doc_10",
        "title": "Attention Mechanisms",
        "text": (
            "Scaled dot-product attention computes compatibility between queries Q and "
            "keys K, scaled by sqrt(d_k), then applies a softmax and multiplies by "
            "values V: Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V. Multi-head attention "
            "runs h attention heads in parallel on different linear projections and "
            "concatenates results. Self-attention allows each token to attend to all "
            "other tokens in the sequence."
        ),
    },
    {
        "id": "doc_11",
        "title": "Instruction Tuning and RLHF",
        "text": (
            "Instruction tuning fine-tunes a pre-trained LLM on (instruction, response) "
            "pairs so the model follows natural-language directions. InstructGPT applies "
            "Reinforcement Learning from Human Feedback (RLHF): a reward model is "
            "trained on human preference rankings, then PPO optimises the LLM to "
            "maximise reward while staying close to the SFT baseline via a KL penalty."
        ),
    },
    {
        "id": "doc_12",
        "title": "Mixture of Experts (MoE)",
        "text": (
            "Mixture of Experts (MoE) scales model capacity without proportionally "
            "increasing compute. A gating network routes each token to a subset of "
            "expert FFN layers (typically top-2). Only the selected experts are active "
            "per forward pass, reducing FLOPs per token. Mistral 8x7B and GPT-4 (as "
            "reported) use sparse MoE. Load balancing losses prevent expert collapse."
        ),
    },
    {
        "id": "doc_13",
        "title": "Prompt Engineering",
        "text": (
            "Prompt engineering designs inputs to elicit desired LLM behaviour without "
            "parameter updates. Few-shot prompting includes examples in the context. "
            "Chain-of-thought (CoT) prompting instructs the model to reason step-by-step "
            "before answering, improving arithmetic and multi-hop reasoning. "
            "ReAct interleaves reasoning (Thought) and action (tool calls) with "
            "observations, enabling agents to interact with external environments."
        ),
    },
    {
        "id": "doc_14",
        "title": "Knowledge Distillation",
        "text": (
            "Knowledge distillation trains a smaller student model to mimic the output "
            "distribution of a larger teacher model. The student minimises a combination "
            "of the cross-entropy loss on ground-truth labels and a KL-divergence loss "
            "against the teacher's soft probabilities (at a temperature T). DistilBERT "
            "achieves 97 % of BERT performance at 60 % of its size."
        ),
    },
    {
        "id": "doc_15",
        "title": "Parameter-Efficient Fine-Tuning (PEFT)",
        "text": (
            "PEFT methods fine-tune a small fraction of parameters while freezing the "
            "pre-trained backbone. LoRA (Low-Rank Adaptation) adds trainable rank-r "
            "matrices to weight updates: W' = W + BA where B ∈ R^(d×r), A ∈ R^(r×k). "
            "Prefix tuning prepends soft tokens to the input. Adapters insert small "
            "bottleneck modules between transformer layers. PEFT is critical for "
            "fine-tuning large models on consumer hardware."
        ),
    },
]

# Ground-truth relevant doc IDs per query (used for precision/recall)
QUERY_GROUND_TRUTH: Dict[str, List[str]] = {
    "q01": ["doc_04", "doc_05", "doc_06"],
    "q02": ["doc_01", "doc_10"],
    "q03": ["doc_02", "doc_03"],
    "q04": ["doc_07"],
    "q05": ["doc_08"],
    "q06": ["doc_09", "doc_04"],
    "q07": ["doc_13"],
    "q08": ["doc_11"],
    "q09": ["doc_14", "doc_15"],
    "q10": ["doc_12"],
}

EVALUATION_QUERIES = [
    {"id": "q01", "text": "How does RAG work and what vector database is used?"},
    {"id": "q02", "text": "What is self-attention and how are positional encodings used?"},
    {"id": "q03", "text": "What are the differences between BERT and GPT pre-training?"},
    {"id": "q04", "text": "What chunking strategies are used in document retrieval pipelines?"},
    {"id": "q05", "text": "How is retrieval quality measured with precision and recall?"},
    {"id": "q06", "text": "What is hallucination in LLMs and how does RAG mitigate it?"},
    {"id": "q07", "text": "Explain chain-of-thought prompting and the ReAct pattern."},
    {"id": "q08", "text": "How does RLHF train language models to follow instructions?"},
    {"id": "q09", "text": "What is knowledge distillation and how does LoRA differ from it?"},
    {"id": "q10", "text": "What is Mixture of Experts and how does it reduce compute?"},
]


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


def fixed_size_chunking(doc: Dict, chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    text = doc["text"]
    chunks = []
    start = 0
    idx = 0
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


def recursive_chunking(doc: Dict, chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    """Recursively splits on paragraph → sentence → word boundaries."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    text = doc["text"]

    def _split(text: str, sep_idx: int) -> List[str]:
        if len(text) <= chunk_size or sep_idx >= len(separators):
            return [text]
        sep = separators[sep_idx]
        parts = text.split(sep) if sep else list(text)
        results: List[str] = []
        current = ""
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
        # Add overlap by appending first `overlap` chars of next chunk to each
        overlapped: List[str] = []
        for i, chunk_text in enumerate(results):
            if i + 1 < len(results):
                suffix = results[i + 1][:overlap]
                overlapped.append(chunk_text + suffix)
            else:
                overlapped.append(chunk_text)
        return overlapped

    raw_chunks = _split(text, 0)
    chunks = []
    pos = 0
    for idx, chunk_text in enumerate(raw_chunks):
        start = text.find(chunk_text[:min(20, len(chunk_text))], pos)
        if start == -1:
            start = pos
        end = start + len(chunk_text)
        chunks.append(Chunk(
            chunk_id=f"{doc['id']}_recursive_{idx}",
            doc_id=doc["id"],
            text=chunk_text,
            start_char=start,
            end_char=end,
        ))
        pos = max(pos, end - overlap)
    return chunks


def sentence_chunking(doc: Dict, max_sentences: int = 4, overlap_sentences: int = 1) -> List[Chunk]:
    """Splits text into windows of sentences."""
    text = doc["text"]
    sentence_endings = [m.end() for m in re.finditer(r'(?<=[.!?])\s+', text)]
    boundaries = [0] + sentence_endings + [len(text)]
    sentences = [text[boundaries[i]:boundaries[i + 1]].strip()
                 for i in range(len(boundaries) - 1) if boundaries[i] < boundaries[i + 1]]
    chunks = []
    i = 0
    idx = 0
    while i < len(sentences):
        window = sentences[i: i + max_sentences]
        chunk_text = " ".join(window)
        start = text.find(window[0][:min(20, len(window[0]))]) if window else 0
        chunks.append(Chunk(
            chunk_id=f"{doc['id']}_sentence_{idx}",
            doc_id=doc["id"],
            text=chunk_text,
            start_char=max(0, start),
            end_char=min(len(text), start + len(chunk_text)),
        ))
        i += max_sentences - overlap_sentences
        idx += 1
    return chunks


def chunk_documents(
    docs: List[Dict],
    strategy: str = "recursive",
    chunk_size: int = 512,
    overlap: int = 64,
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
    latency_ms = (time.perf_counter() - t0) * 1000
    return all_chunks, latency_ms


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(chunks: List[Chunk], model_name: str = "all-MiniLM-L6-v2") -> Tuple[np.ndarray, float]:
    from sentence_transformers import SentenceTransformer
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    latency_ms = (time.perf_counter() - t0) * 1000
    return np.array(embeddings, dtype="float32"), latency_ms


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> Tuple[np.ndarray, float]:
    from sentence_transformers import SentenceTransformer
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    emb = model.encode([query], normalize_embeddings=True)
    latency_ms = (time.perf_counter() - t0) * 1000
    return np.array(emb, dtype="float32"), latency_ms


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_index(embeddings: np.ndarray) -> Tuple[Any, float]:
    import faiss
    t0 = time.perf_counter()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product == cosine for normalised vecs
    index.add(embeddings)
    latency_ms = (time.perf_counter() - t0) * 1000
    return index, latency_ms


def retrieve(
    query_emb: np.ndarray,
    index: Any,
    chunks: List[Chunk],
    k: int = 5,
) -> Tuple[List[Chunk], List[float], float]:
    import faiss  # noqa
    t0 = time.perf_counter()
    scores, indices = index.search(query_emb, k)
    latency_ms = (time.perf_counter() - t0) * 1000
    retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
    score_list = scores[0].tolist()[:len(retrieved)]
    return retrieved, score_list, latency_ms


# ---------------------------------------------------------------------------
# Generation via Ollama
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    context_chunks: List[Chunk],
    model: str = "mistral:7b-instruct",
) -> Tuple[str, float]:
    context = "\n\n".join([f"[{c.doc_id}] {c.text}" for c in context_chunks])
    prompt = (
        f"You are a helpful assistant. Answer the question using ONLY the context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    t0 = time.perf_counter()
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response["message"]["content"].strip()
    except Exception as exc:
        answer = f"[Ollama unavailable – {exc}] Context summary: {context[:300]}..."
    latency_ms = (time.perf_counter() - t0) * 1000
    return answer, latency_ms


# ---------------------------------------------------------------------------
# Groundedness scoring (lexical overlap heuristic)
# ---------------------------------------------------------------------------

def groundedness_score(answer: str, context_chunks: List[Chunk]) -> float:
    """Fraction of answer tokens that appear in the retrieved context."""
    context_text = " ".join(c.text for c in context_chunks).lower()
    answer_tokens = re.findall(r'\b\w+\b', answer.lower())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "of",
                  "to", "and", "or", "it", "that", "this", "for", "with", "be",
                  "as", "at", "by", "from", "on", "not", "but"}
    content_tokens = [t for t in answer_tokens if t not in stop_words and len(t) > 2]
    if not content_tokens:
        return 0.0
    matched = sum(1 for t in content_tokens if t in context_text)
    return round(matched / len(content_tokens), 3)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: List[Chunk], relevant_ids: List[str], k: int) -> float:
    top_k = retrieved[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for c in top_k if c.doc_id in relevant_set)
    return round(hits / k, 3)


def recall_at_k(retrieved: List[Chunk], relevant_ids: List[str], k: int) -> float:
    top_k = retrieved[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for c in top_k if c.doc_id in relevant_set)
    return round(hits / len(relevant_set), 3) if relevant_set else 0.0


def mrr(retrieved: List[Chunk], relevant_ids: List[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.doc_id in relevant_set:
            return round(1.0 / rank, 3)
    return 0.0


# ---------------------------------------------------------------------------
# Full pipeline run
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
            self.latency_chunking_ms
            + self.latency_embedding_ms
            + self.latency_index_build_ms
            + self.latency_query_embed_ms
            + self.latency_retrieval_ms
            + self.latency_generation_ms
        )


class RAGPipeline:
    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 512,
        overlap: int = 64,
        model_name: str = "all-MiniLM-L6-v2",
        llm_model: str = "mistral:7b-instruct",
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
        self._latency_ingestion_ms = 0.0
        self._latency_chunking_ms = 0.0
        self._latency_embedding_ms = 0.0
        self._latency_index_build_ms = 0.0

    def ingest(self, docs: List[Dict]) -> None:
        t0 = time.perf_counter()
        # Ingestion phase: just loading the docs into memory
        _ = [d["text"] for d in docs]
        self._latency_ingestion_ms = (time.perf_counter() - t0) * 1000

        self.chunks, self._latency_chunking_ms = chunk_documents(
            docs, self.strategy, self.chunk_size, self.overlap
        )
        embeddings, self._latency_embedding_ms = embed_chunks(self.chunks, self.model_name)
        self.index, self._latency_index_build_ms = build_index(embeddings)

    def query(self, query_text: str, query_id: str = "q00") -> PipelineResult:
        relevant_ids = QUERY_GROUND_TRUTH.get(query_id, [])
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
            precision_at_3=precision_at_k(retrieved, relevant_ids, 3),
            recall_at_3=recall_at_k(retrieved, relevant_ids, 3),
            precision_at_5=precision_at_k(retrieved, relevant_ids, 5),
            recall_at_5=recall_at_k(retrieved, relevant_ids, 5),
            mrr_score=mrr(retrieved, relevant_ids),
            latency_ingestion_ms=self._latency_ingestion_ms,
            latency_chunking_ms=self._latency_chunking_ms,
            latency_embedding_ms=self._latency_embedding_ms,
            latency_index_build_ms=self._latency_index_build_ms,
            latency_query_embed_ms=lat_qemb,
            latency_retrieval_ms=lat_ret,
            latency_generation_ms=lat_gen,
        )


# ---------------------------------------------------------------------------
# CLI / demo entry point
# ---------------------------------------------------------------------------

def run_evaluation(strategy: str = "recursive", chunk_size: int = 512) -> List[PipelineResult]:
    print(f"\n=== RAG Evaluation | strategy={strategy} chunk_size={chunk_size} ===")
    pipeline = RAGPipeline(strategy=strategy, chunk_size=chunk_size)
    print("Ingesting documents…")
    pipeline.ingest(DOCUMENTS)
    print(f"  Chunks created   : {len(pipeline.chunks)}")
    print(f"  Chunking latency : {pipeline._latency_chunking_ms:.1f} ms")
    print(f"  Embedding latency: {pipeline._latency_embedding_ms:.1f} ms")
    print(f"  Index build      : {pipeline._latency_index_build_ms:.1f} ms")

    results = []
    for q in EVALUATION_QUERIES:
        r = pipeline.query(q["text"], q["id"])
        results.append(r)
        print(
            f"  [{r.query_id}] P@3={r.precision_at_3} R@3={r.recall_at_3} "
            f"MRR={r.mrr_score} GS={r.groundedness} "
            f"gen={r.latency_generation_ms:.0f}ms"
        )

    avg_p3 = sum(r.precision_at_3 for r in results) / len(results)
    avg_r3 = sum(r.recall_at_3 for r in results) / len(results)
    avg_mrr = sum(r.mrr_score for r in results) / len(results)
    avg_gs = sum(r.groundedness for r in results) / len(results)
    avg_total = sum(r.total_latency_ms for r in results) / len(results)
    print(f"\n  Avg P@3={avg_p3:.3f}  Avg R@3={avg_r3:.3f}  Avg MRR={avg_mrr:.3f}  "
          f"Avg GS={avg_gs:.3f}  Avg total latency={avg_total:.0f}ms")
    return results


if __name__ == "__main__":
    all_results = []
    for strat in ["fixed", "recursive", "sentence"]:
        results = run_evaluation(strategy=strat, chunk_size=512)
        all_results.extend(results)

    # Dump results to JSON for the evaluation report
    output = []
    for r in all_results:
        output.append({
            "query_id": r.query_id,
            "query": r.query,
            "strategy": r.strategy,
            "chunk_size": r.chunk_size,
            "retrieved_docs": r.retrieved_docs,
            "answer_snippet": r.answer[:200],
            "groundedness": r.groundedness,
            "precision_at_3": r.precision_at_3,
            "recall_at_3": r.recall_at_3,
            "precision_at_5": r.precision_at_5,
            "recall_at_5": r.recall_at_5,
            "mrr": r.mrr_score,
            "latency_chunking_ms": round(r.latency_chunking_ms, 2),
            "latency_embedding_ms": round(r.latency_embedding_ms, 2),
            "latency_index_build_ms": round(r.latency_index_build_ms, 2),
            "latency_query_embed_ms": round(r.latency_query_embed_ms, 2),
            "latency_retrieval_ms": round(r.latency_retrieval_ms, 2),
            "latency_generation_ms": round(r.latency_generation_ms, 2),
            "total_latency_ms": round(r.total_latency_ms, 2),
        })
    with open("rag_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to rag_results.json")
