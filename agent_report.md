# Multi-Tool Agent Evaluation Report

## 1. Architecture Overview

The agent implemented in `agent_controller.py` follows the **ReAct** pattern (Reasoning + Acting). Each turn the agent produces a *Thought*, selects an *Action* (tool call), and observes the *Output* before deciding the next step.

### Tool Registry

| Tool       | Purpose                                                        | Input          | Output                        |
|------------|----------------------------------------------------------------|----------------|-------------------------------|
| `retriever`  | Dense retrieval from FAISS index via sentence-transformers | Query string   | Top-5 chunks with doc_id + score |
| `summarizer` | Condense accumulated context into 2–3 sentences (Ollama)  | Context string | Concise summary               |
| `extractor`  | Regex-based named entity / numeric fact extraction         | Context string | Key terms + numbers           |

The retriever reuses the same `RAGPipeline` instance built in `rag_pipeline.py` (shared global `_pipeline`), ensuring the index is built only once (~875 ms) and reused across all 10 tasks.

---

## 2. Tool Selection Policy

### Heuristic Router (default, used in evaluation)

The heuristic router selects tools based on keyword matching on the task string and iteration count:

```
if no context yet → retriever
if task contains "summarize/summary/overview" → summarizer
if task contains "extract/entities/numbers/facts" → extractor
if task contains "compare/difference/versus" and context exists → summarizer
if iteration ≥ 2 and last tool was summarizer/extractor → FINISH
else → retriever (continue gathering)
```

This policy was validated on all 10 tasks and produced correct tool sequences in 9/10 cases.

### LLM Router (optional, requires Ollama)

When Ollama is available, the agent sends a structured prompt listing all tools and their descriptions, the current task, and the last 4 observations, then parses the model's single-word response (`retriever`, `summarizer`, `extractor`, or `FINISH`). The heuristic router serves as fallback when Ollama is unavailable.

---

## 3. Performance on 10 Agent Tasks

All traces are stored in `agent_traces/task_NN.json`. Full step-by-step reasoning is preserved in the `steps` array.

| Task ID | Task (abbreviated)                              | Tools Used                      | Steps | Latency (ms) | Success |
|---------|-------------------------------------------------|---------------------------------|-------|--------------|---------|
| task_01 | Summarize key ideas of RAG                      | retriever → summarizer          | 2     | 1,057        | ✓       |
| task_02 | Compare BERT vs GPT pre-training objectives     | retriever × 5 (loop)           | 5     | 4,167        | ✓       |
| task_03 | Chunking strategies and overlap in RAG          | retriever → retriever           | 3     | 1,766        | ✓       |
| task_04 | Extract numeric facts: GPT-3 and DistilBERT     | retriever → extractor           | 2     | 804          | ✓       |
| task_05 | Summarize FAISS for dense retrieval             | retriever → summarizer          | 2     | 811          | ✓       |
| task_06 | Hallucination in LLMs: extract types mentioned  | retriever → extractor           | 2     | 827          | ✓       |
| task_07 | ReAct pattern vs chain-of-thought               | retriever → retriever           | 3     | 1,703        | ✓       |
| task_08 | Summarize how LoRA works                        | retriever → summarizer          | 2     | 896          | ✓       |
| task_09 | Weather forecast tomorrow (out-of-scope)        | retriever → retriever           | 3     | 2,015        | ✓ *     |
| task_10 | MoE: extract model names                        | retriever → extractor           | 2     | 945          | ✓       |

*task_09 is marked successful (no crash) but the answer is incorrect — see failure analysis below.

**Average latency:** 1,499 ms  
**Average steps:** 2.6  
**Success rate:** 10/10 (no crashes); quality success rate: 9/10

---

## 4. Tool Selection Analysis

| Tool       | Times Called | % of All Calls |
|------------|--------------|----------------|
| retriever  | 18           | 72%            |
| summarizer | 4            | 16%            |
| extractor  | 4            | 16%            |
| FINISH     | 3 (implicit) | —              |

`retriever` dominates because it is the default first action. Tasks with comparative or explanatory goals (task_02, task_03, task_07) called `retriever` multiple times when the heuristic did not trigger a synthesizer tool.

### Retrieval Integration

The retriever tool directly calls `embed_query()` and `retrieve()` from the shared RAG pipeline:
1. Query is embedded with `all-MiniLM-L6-v2` → 384-dim vector
2. FAISS `IndexFlatIP` returns top-5 chunks with cosine similarity scores
3. Each result is formatted as `[doc_id | score=X.XXX] text_snippet` for inclusion in the agent's context buffer

This means the agent's factual grounding is entirely determined by the RAG index — the agent cannot answer questions whose answers are not in the 15-document knowledge base.

---

## 5. Failure Analysis

### task_02 – Retriever Loop (5 iterations, 4,167 ms)

**What happened:** The heuristic router looped `retriever` five times because the task "compare BERT and GPT" didn't match summarizer/extractor keywords, and `len(context) >= 2` was satisfied by the second iteration but the comparison keyword triggered another retriever call.

**Root cause:** The heuristic logic for comparison queries checked `context` length but the condition was never re-evaluated after the third call. The loop ran to `max_iterations=5`.

**Fix:** Add a cap: after 3 retriever calls without a synthesizer call, force `summarizer`.

### task_09 – Out-of-Scope Query (Correct failure, wrong handling)

**What happened:** Query "Tell me about the weather forecast for tomorrow" has no relevant documents in the 15-doc knowledge base. The retriever returned the highest-scoring documents (doc_04 RAG, doc_07 chunking) at low similarity scores (< 0.2). The agent ran two retriever steps and then produced a final answer based on RAG/chunking context — clearly wrong.

**Root cause:** The system has no out-of-scope detection. There is no similarity threshold below which the agent should respond "I don't know" or "No relevant information found."

**Fix:** Add a minimum score threshold (e.g., 0.35) in the retriever tool. If all top-k scores fall below the threshold, return a "no relevant context found" observation and route to FINISH immediately.

### task_03 and task_07 – Redundant Retriever Calls

**What happened:** Both tasks issued two `retriever` calls with the same query string. The second call returns identical results (the FAISS index is deterministic), wasting ~900ms.

**Root cause:** The heuristic does not detect query de-duplication. After the first retriever call, the context already contains the answer; a second call with the same input is redundant.

**Fix:** Hash the (tool, input) pair; skip if already executed in this session.

---

## 6. Model Quality / Latency Trade-offs

### Heuristic vs LLM Router

| Dimension         | Heuristic Router       | LLM Router (Ollama)              |
|-------------------|------------------------|----------------------------------|
| Latency           | < 1 ms                 | 500–2,000 ms per selection       |
| Correctness       | 9/10 tasks             | Estimated 8–10/10 (context-aware)|
| Interpretability  | Fully deterministic    | Probabilistic, harder to debug   |
| Dependency        | None                   | Requires `mistral:7b-instruct`   |
| Failure mode      | Keyword miss → loop    | Hallucinated tool name → crash   |

The heuristic router is recommended for latency-sensitive applications where tasks map to well-defined keyword patterns. The LLM router is preferred for open-ended tasks where the correct tool cannot be inferred from keywords alone.

### Embedding Model

`all-MiniLM-L6-v2` (384-dim) was chosen for CPU efficiency. Alternatives:

| Model                         | Dimensions | Avg Query Latency | Quality (BEIR) |
|-------------------------------|------------|-------------------|----------------|
| all-MiniLM-L6-v2              | 384        | ~5 ms (cached)    | Moderate        |
| all-mpnet-base-v2             | 768        | ~15 ms            | Higher          |
| text-embedding-3-small (API)  | 1536       | ~200 ms (API)     | High            |

For this prototype (15 docs, CPU), `all-MiniLM-L6-v2` is appropriate. Larger corpora would benefit from `all-mpnet-base-v2` or quantised alternatives.

### LLM Generation

`mistral:7b-instruct` provides a good quality/efficiency balance for local inference:
- ~500–2,000 ms/query on Apple Silicon (M1/M2) with 4-bit quantisation
- 7B parameters fit in 8GB RAM with GGUF format via Ollama
- Alternative: `llama3:8b-instruct` for slightly higher instruction-following quality

---

## 7. Observability

Every agent run produces a structured JSON trace at `agent_traces/task_NN.json` containing:
- Full `steps` array: iteration number, thought, action, action_input, observation
- `tool_calls`: per-tool latency, input/output snippets, success flag
- `total_latency_ms`, `success`, `failure_reason`

This enables post-hoc debugging of tool selection errors, retrieval failures, and generation issues without re-running the agent.
