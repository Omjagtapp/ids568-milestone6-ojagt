# RAG Pipeline Architecture Diagram

## Full Data-Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE OVERVIEW                               │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌───────────────┐
  │  Raw Documents│  (15 NLP/ML topic docs; plain text with doc_id + title)
  └───────┬───────┘
          │  Ingestion (~0.01 ms)
          ▼
  ┌───────────────┐
  │   Document    │  Load text into memory, assign doc_id, validate structure
  │   Ingestion   │
  └───────┬───────┘
          │
          ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                         CHUNKING STAGE (~0.1–0.2 ms)                     │
  │                                                                           │
  │  Strategy A: Fixed-Size          Strategy B: Recursive         Strategy C │
  │  ┌──────────────────────┐        ┌────────────────────────┐   Sentence   │
  │  │chunk_size=512 chars  │        │Paragraph → Sentence    │   ┌─────────┐│
  │  │overlap=64 chars      │        │→ Word boundary split   │   │4 sents/ ││
  │  │15 chunks produced    │        │chunk_size=512, ovlp=64 │   │window   ││
  │  └──────────────────────┘        │15 chunks produced      │   │25 chunks││
  │                                  └────────────────────────┘   └─────────┘│
  └───────────────────────────────────────────────────────────────────────────┘
          │
          │  List[Chunk]  (chunk_id, doc_id, text, start_char, end_char)
          ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                        EMBEDDING STAGE (~875–1141 ms)                    │
  │                                                                           │
  │  Model: all-MiniLM-L6-v2 (sentence-transformers)                         │
  │  Output: 384-dimensional float32 vectors, L2-normalised                  │
  │  Batch size: 32                                                           │
  │                                                                           │
  │  [chunk₁ text] ──► [0.12, -0.05, …, 0.08]  (384-dim)                   │
  │  [chunk₂ text] ──► [0.03,  0.21, …, -0.14] (384-dim)                   │
  │         ⋮                    ⋮                                            │
  └───────────────────────────────────────────────────────────────────────────┘
          │
          │  np.ndarray  shape=(N_chunks, 384)
          ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                       VECTOR STORE (FAISS IndexFlatIP)                   │
  │                                                                           │
  │  Index type: Flat Inner-Product (exact, brute-force)                     │
  │  Similarity: cosine (via L2-normalised vectors → IP == cosine)           │
  │  Build time: < 1 ms for 15–25 chunks                                     │
  │                                                                           │
  │  ┌───────────────────────────────────────────────┐                        │
  │  │  Vector Index                                  │                        │
  │  │  [v₁, v₂, v₃, …, vN]  ← all chunk embeddings│                        │
  │  └───────────────────────────────────────────────┘                        │
  └───────────────────────────────────────────────────────────────────────────┘
          ▲                           │
          │  (index built once;       │
          │   queried per request)    │
          │                           ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                            QUERY PROCESSING                              │
  │                                                                           │
  │  User Query  ──►  Embed Query (all-MiniLM-L6-v2)  ──►  q_vec (384-dim) │
  │                   Latency: ~50–80 ms                                      │
  └───────────────────────────────────────────────────────────────────────────┘
          │
          │  q_vec
          ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                        RETRIEVAL STAGE (< 1 ms)                          │
  │                                                                           │
  │  index.search(q_vec, k=5)                                                │
  │  Returns top-k chunk indices + similarity scores                          │
  │                                                                           │
  │  Query: "How does RAG work?"                                             │
  │  ┌──────────────────────────────────────────────┐                        │
  │  │ Rank 1: doc_04 chunk  | score=0.912          │                        │
  │  │ Rank 2: doc_06 chunk  | score=0.871          │                        │
  │  │ Rank 3: doc_05 chunk  | score=0.843          │                        │
  │  │ Rank 4: doc_07 chunk  | score=0.801          │                        │
  │  │ Rank 5: doc_09 chunk  | score=0.776          │                        │
  │  └──────────────────────────────────────────────┘                        │
  └───────────────────────────────────────────────────────────────────────────┘
          │
          │  top-3 chunks concatenated as context
          ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                      GENERATION STAGE (Ollama LLM)                       │
  │                                                                           │
  │  Model: mistral:7b-instruct (via Ollama local server)                    │
  │  Prompt template:                                                         │
  │  ┌──────────────────────────────────────────────────────────────────────┐│
  │  │ "Answer using ONLY the context below.                                ││
  │  │  Context: [doc_04] … [doc_06] … [doc_05] …                          ││
  │  │  Question: <user query>                                              ││
  │  │  Answer:"                                                            ││
  │  └──────────────────────────────────────────────────────────────────────┘│
  │                                                                           │
  │  Latency: 0–155 ms (0 ms = Ollama not installed; 155 ms = first call)   │
  └───────────────────────────────────────────────────────────────────────────┘
          │
          │  Generated text
          ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                    GROUNDEDNESS SCORING (post-hoc)                       │
  │                                                                           │
  │  Lexical overlap: fraction of non-stopword answer tokens found in context│
  │  Score range: 0.0 (fully hallucinated) → 1.0 (fully grounded)           │
  │  Observed scores: 0.646 – 0.692                                          │
  └───────────────────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌───────────────┐
  │ Grounded      │  Final answer with citations (doc_id references embedded)
  │ Response      │
  └───────────────┘


## Stage Latency Summary (Recursive strategy, 15 chunks)

| Stage                  | Latency        | Notes                              |
|------------------------|----------------|------------------------------------|
| Document ingestion     | < 0.1 ms       | In-memory load                     |
| Chunking               | 0.0 – 0.2 ms   | Depends on strategy                |
| Embedding (batch)      | 875 – 11524 ms | First run downloads model weights  |
| FAISS index build      | < 1 ms         | Flat index, 15–25 vectors          |
| Query embedding        | 50 – 80 ms     | Single vector encode               |
| FAISS retrieval        | < 1 ms         | Exact brute-force search           |
| LLM generation         | 0 – 155 ms     | 0 ms when Ollama not available     |
| Groundedness scoring   | < 1 ms         | Regex token overlap                |
| **Total (per query)**  | **~130 ms**    | After index is built               |


## Component Interaction Diagram

```
                    ┌───────────────────┐
                    │   Knowledge Base  │
                    │   (15 documents)  │
                    └────────┬──────────┘
                             │ ingest()
                             ▼
                    ┌───────────────────┐      ┌──────────────────────────┐
                    │  ChunkingEngine   │      │  SentenceTransformer     │
                    │  ─ fixed          │─────►│  all-MiniLM-L6-v2        │
                    │  ─ recursive      │      │  384-dim embeddings      │
                    │  ─ sentence       │      └────────────┬─────────────┘
                    └───────────────────┘                   │
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │  FAISS IndexFlatIP   │
                                                 │  (vector store)      │
                                                 └──────────┬───────────┘
                                                            │
               ┌──────────────────┐                        │
               │   User Query     │──►  embed_query()  ────┤
               └──────────────────┘                        │ search(k=5)
                                                            │
                                                 ┌──────────▼───────────┐
                                                 │  Top-k Chunks        │
                                                 │  + Similarity Scores │
                                                 └──────────┬───────────┘
                                                            │
                                                 ┌──────────▼───────────┐
                                                 │  Ollama LLM          │
                                                 │  mistral:7b-instruct │
                                                 └──────────┬───────────┘
                                                            │
                                                 ┌──────────▼───────────┐
                                                 │  Grounded Response   │
                                                 │  + Groundedness Score│
                                                 └──────────────────────┘
```
