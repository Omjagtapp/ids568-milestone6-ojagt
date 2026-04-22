# Milestone 6: RAG & Agentic Pipelines

**Course:** IDS 568 – MLOps  
**Module:** 7 – Retrieval-Augmented Generation and Agentic Systems  
**Author:** ojagt-1

---

## Repository Contents

```
.
├── rag_pipeline.py           # Full RAG pipeline (ingestion → chunking → embedding → FAISS → retrieval → generation)
├── agent_controller.py       # Multi-tool ReAct agent (retriever + summarizer + extractor)
├── requirements.txt          # Python dependencies
├── rag_results.json          # Machine-readable evaluation results (auto-generated)
├── agent_traces/             # 10 JSON trace files (task_01.json … task_10.json)
│   ├── task_01.json
│   ├── task_02.json
│   ├── task_03.json
│   ├── task_04.json
│   ├── task_05.json
│   ├── task_06.json
│   ├── task_07.json
│   ├── task_08.json
│   ├── task_09.json
│   └── task_10.json
├── rag_evaluation_report.md  # ~2-page evaluation: metrics, grounding, failures, latency
├── rag_pipeline_diagram.md   # ASCII/Markdown pipeline architecture diagram
└── agent_report.md           # Agent tool selection, performance, failure analysis
```

---

## Architecture Overview

### RAG Pipeline

```
Documents → Chunker → Embedder (all-MiniLM-L6-v2) → FAISS IndexFlatIP
                                                             │
User Query → embed_query() ──────────────────────────► similarity search
                                                             │
                                                      Top-k Chunks
                                                             │
                                                 Ollama LLM (llama3.1:8b)
                                                             │
                                                    Grounded Response
```

- **Chunking strategies:** fixed-size, recursive (paragraph→sentence→word), sentence-window
- **Embeddings:** `all-MiniLM-L6-v2` (384-dim), L2-normalised, cosine similarity via FAISS inner-product
- **Vector store:** FAISS `IndexFlatIP` (exact brute-force; suitable for ≤ 10K chunks)
- **Generation:** `llama3.1:8b` via Ollama local server; graceful fallback if unavailable

### Agent Controller

```
Task → Tool Selector (heuristic or LLM) → Tool Execution → Observation
                ▲                                                │
                └──────────────── Context Buffer ◄──────────────┘
                                        │
                                 Final Answer Synthesis
```

- **Tools:** retriever (RAG), summarizer (Ollama), extractor (regex)
- **ReAct loop:** max 5 iterations; stops early when synthesizer tool completes
- **Traces:** each run serialised to `agent_traces/task_NN.json`

---

## Model Deployment

| Property                   | Value                                          |
|----------------------------|------------------------------------------------|
| Model name (evaluated)     | `llama3.1:8b`                                  |
| Model family               | Meta Llama 3.1                                 |
| Size class                 | 8B parameters                                  |
| Quantisation               | 4-bit GGUF (default Ollama download, ~4.9 GB)  |
| Serving stack              | Ollama v0.12.6 (local HTTP server, port 11434) |
| Hardware tested            | Apple Silicon M-series (arm64, macOS)          |
| Minimum RAM                | 8 GB unified memory                            |
| Typical generation latency | 3,000–10,000 ms per query (Apple M1/M2 CPU)   |
| Embedding model            | `all-MiniLM-L6-v2` (384-dim, ~90 MB)          |
| Embedding latency          | ~5 ms per query (model cached in memory)       |
| Alternative model          | `mistral:7b-instruct` (also compatible)        |

**Evaluated runs use `llama3.1:8b` — a real open-weight 8B instruct model running locally via Ollama. No proprietary APIs are used.**

Start the model server before running any generation:

```bash
ollama serve                  # start Ollama server (runs in background)
ollama pull llama3.1:8b       # download model (~4.9 GB, one-time)
ollama list                   # verify: should show llama3.1:8b

# Alternative (also supported):
ollama pull mistral:7b-instruct
```

---

## Setup Instructions

### 1. Install Python dependencies

```bash
# Python 3.9+ required
python3 -m pip install -r requirements.txt
```

> **Note:** The first run of `rag_pipeline.py` downloads `all-MiniLM-L6-v2` (~90 MB) from Hugging Face. Subsequent runs use the local cache.

### 2. Install and start Ollama (for LLM generation)

```bash
# macOS
brew install ollama

# Or download from https://ollama.com/download

# Start the Ollama server
ollama serve
```

### 3. Pull the LLM model

```bash
ollama pull llama3.1:8b

# Alternative (also supported):
ollama pull mistral:7b-instruct
```

Verify the model is available:
```bash
ollama list
# Should show: llama3.1:8b
```

> **Without Ollama:** The RAG pipeline and agent will still run. Generation falls back to returning a snippet of the retrieved context. All retrieval metrics (P@k, R@k, MRR, groundedness) are unaffected.

---

## Usage

### Run the RAG Pipeline

```bash
python3 rag_pipeline.py
```

This evaluates all three chunking strategies on all 10 queries and saves results to `rag_results.json`.

**Expected output:**
```
=== RAG Evaluation | strategy=fixed chunk_size=512 ===
Ingesting documents…
  Chunks created   : 15
  Chunking latency : 0.0 ms
  Embedding latency: 875.0 ms
  [q01] P@3=0.667 R@3=0.667 MRR=1.0 GS=0.652 gen=155ms
  ...
Results saved to rag_results.json
```

### Run the Agent

```bash
python3 agent_controller.py
```

Runs all 10 agent tasks and saves traces to `agent_traces/`.

**Expected output:**
```
=== Agent Evaluation – 10 Tasks ===
Building RAG index (one-time)…
[task_01] Retrieve and summarize the key ideas behind Retrieval-Augmented Genera…
  Tools used: ['retriever', 'summarizer'] | Steps: 2 | Latency: 1057ms | Success: True
...
All 10 traces saved to agent_traces/
```

### Programmatic Usage (RAG)

```python
from rag_pipeline import RAGPipeline, DOCUMENTS

pipeline = RAGPipeline(strategy="recursive", chunk_size=512, k=5)
pipeline.ingest(DOCUMENTS)
result = pipeline.query("How does RAG reduce hallucination?", query_id="q06")
print(result.answer)
print(f"P@3={result.precision_at_3}, R@3={result.recall_at_3}, GS={result.groundedness}")
```

### Programmatic Usage (Agent)

```python
from agent_controller import run_agent_task

trace = run_agent_task(
    task_id="custom_01",
    task="Explain the difference between LoRA and full fine-tuning.",
    use_llm_routing=False,  # set True if Ollama is running
)
print(trace.final_answer)
print(f"Tools: {[tc.tool_name for tc in trace.tool_calls]}")
```

---

## Known Limitations

1. **No out-of-scope detection**: Queries outside the 15-document knowledge base are answered with lowest-ranked retrieved chunks instead of a "no information available" response. Fix: add a minimum similarity threshold.

2. **Model reload per query**: `embed_query()` reloads `all-MiniLM-L6-v2` on every call in the current implementation. In production, cache the model globally to reduce query latency from ~938 ms to ~5 ms.

3. **Small knowledge base**: 15 documents cover NLP/ML topics only. The system is intended as a prototype demonstrating the RAG pattern, not a production retrieval system.

4. **FAISS brute-force index**: `IndexFlatIP` is exact and memory-resident. For corpora > 100K chunks, switch to `IndexIVFFlat` (with `nlist` parameter tuning) or `IndexHNSWFlat` for sub-linear ANN search.

5. **Heuristic tool selection loops**: Out-of-scope queries (e.g., task_09) trigger repeated retriever calls since no similarity threshold exits the loop early. Adding a min-score cutoff (0.35) and (tool, input) de-duplication resolves this.

---

## Dependency Versions

| Package              | Version Required   | Purpose                       |
|----------------------|--------------------|-------------------------------|
| faiss-cpu            | ≥ 1.9.0            | Vector similarity search      |
| sentence-transformers| 2.2.2              | Dense text embeddings         |
| ollama               | 0.1.8              | LLM generation client         |
| numpy                | 1.26.4             | Array operations              |
| scipy                | 1.11.4             | Scientific computing          |
| torch                | 2.1.2              | Sentence-transformers backend |
| transformers         | 4.36.2             | Tokenizers / model configs    |
| huggingface-hub      | 0.20.3             | Model downloads               |

Install all: `python3 -m pip install -r requirements.txt`
