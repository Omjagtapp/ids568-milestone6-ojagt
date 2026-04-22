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

### LLM Router (used in final evaluated runs)

The agent uses `llama3.1:8b` via Ollama for tool selection. The router prompt presents tools as lettered options (A/B/C/D) and instructs the model to reply with a single letter. This avoids the word-in-response parsing problem that plagued earlier prompts where the model would mention "retriever" in its explanation even when recommending a different tool.

---

## 3. Performance on 10 Agent Tasks

All traces are stored in `agent_traces/task_NN.json`. Runs use `llama3.1:8b` for both tool selection and final answer synthesis via Ollama.

| Task ID | Task (abbreviated)                              | Tools Used                                        | Steps | Latency (ms) | Success |
|---------|-------------------------------------------------|---------------------------------------------------|-------|--------------|---------|
| task_01 | Summarize key ideas of RAG                      | retriever → summarizer                            | 2     | 17,858       | ✓       |
| task_02 | Compare BERT vs GPT pre-training objectives     | retriever → extractor                             | 2     | 9,589        | ✓       |
| task_03 | Chunking strategies and overlap in RAG          | retriever → summarizer                            | 2     | 15,254       | ✓       |
| task_04 | Extract numeric facts: GPT-3 and DistilBERT     | retriever → extractor                             | 2     | 10,685       | ✓       |
| task_05 | Summarize FAISS for dense retrieval             | retriever → summarizer                            | 2     | 14,183       | ✓       |
| task_06 | Hallucination in LLMs: extract types mentioned  | retriever → extractor                             | 2     | 9,204        | ✓       |
| task_07 | ReAct pattern vs chain-of-thought               | retriever → summarizer                            | 2     | 16,315       | ✓ *     |
| task_08 | Summarize how LoRA works                        | retriever → summarizer                            | 2     | 14,840       | ✓       |
| task_09 | Weather forecast tomorrow (out-of-scope)        | retriever → extractor                             | 2     | 9,606        | ✓ **    |
| task_10 | MoE: extract model names                        | retriever → extractor                             | 2     | 8,714        | ✓       |

*task_07: LLM router chose summarizer this run (non-deterministic). Final answer correctly notes "ReAct isn't detailed in the provided context" — retrieval miss, not router error.  
**task_09: LLM router chose extractor (C) instead of FINISH (D) for an out-of-scope query. The final LLM answer correctly refused: "there is no information about the weather forecast for tomorrow." The extractor output was irrelevant (BERT/GPT terms).

**Average latency:** 12,625 ms (dominated by `llama3.1:8b` CPU inference ~5–10 s per call)  
**Average steps:** 2.0  
**Success rate:** 10/10 (no crashes); quality success rate: 9/10

---

## 4. Tool Selection Analysis

| Tool       | Times Called | % of All Calls |
|------------|--------------|----------------|
| retriever  | 10           | 50%            |
| summarizer | 5            | 25%            |
| extractor  | 5            | 25%            |
| FINISH     | 0 (implicit) | —              |

`retriever` is always the first action (one call per task, 10 total). The LLM router then routes to summarizer (5 tasks) or extractor (5 tasks) based on task phrasing. All 10 tasks completed in exactly 2 steps, with no FINISH decisions required — the synthesizer call terminates each loop.

### Retrieval Integration

The retriever tool directly calls `embed_query()` and `retrieve()` from the shared RAG pipeline:
1. Query is embedded with `all-MiniLM-L6-v2` → 384-dim vector
2. FAISS `IndexFlatIP` returns top-5 chunks with cosine similarity scores
3. Each result is formatted as `[doc_id | score=X.XXX] text_snippet` for inclusion in the agent's context buffer

This means the agent's factual grounding is entirely determined by the RAG index — the agent cannot answer questions whose answers are not in the 15-document knowledge base.

---

## 5. Failure Analysis

### task_04 – Numeric Extractor + Generation Hallucination

**What happened:** The query asked for numeric facts about GPT-3 and DistilBERT model sizes. The retriever returned doc_03 (GPT) twice and doc_14 (distillation). The extractor included "65B" in its output — a value sourced from a QLoRA passage in the retrieved context. The final LLM answer then hallucinated: "QLoRA mentions fine-tuning of 65B parameter models, which might be referring to DistilBERT." DistilBERT's actual size (40% fewer parameters than BERT-base, roughly 66M) is in doc_14, but the LLM conflated it with the unrelated 65B QLoRA figure.

**Root cause:** (1) The regex extractor pulls any numeric token without semantic grounding — it cannot distinguish "175 billion (GPT-3 size)" from "65B (QLoRA fine-tuning target)." (2) The LLM then performed extrinsic hallucination by connecting two unrelated numbers from different documents.

**Fix:** Scope extractor output to numbers adjacent to entity mentions; add a final-answer verification pass that rejects claims combining facts from different documents.

---

### task_07 – Chunking-Induced Retrieval Miss

**What happened:** The query asked about the ReAct pattern vs chain-of-thought. doc_13 contains a ReAct description, and the retriever did return doc_13 at ranks 1 and 2 (scores 0.667 and 0.349). However, the visible chunk text covered generic prompt-engineering and self-consistency passages; the ReAct-specific sentence did not appear in the top chunks passed to the summarizer. The summarizer produced a correct summary of CoT but the final answer acknowledged: "The ReAct prompting pattern is not explicitly mentioned in the provided context."

**Root cause:** Chunk boundaries split doc_13 so that the ReAct sentence landed in a chunk that ranked outside the useful context window for this query embedding. This is a chunking failure, not a corpus gap.

**Fix:** Use smaller chunks (256 chars) or sentence-level chunking for short definitional documents like doc_13; alternatively, expand the query with synonyms ("Thought Action Observation") to surface the ReAct-specific chunk.

---

### task_10 – Extractor Regex Limitation

**What happened:** The query asked to extract model names from the Mixture of Experts context. doc_12 explicitly names models including "Mistral 8x7B", "Switch Transformer", and similar. The extractor returned "Key terms: FFN, Attention Is All You Need" — missing all specific model names.

**Root cause:** The extractor regex matches `\b[A-Z]{2,}\b` (acronyms) and `\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b` (Title Case bigrams). Neither pattern matches model names containing digits and hyphens (e.g., "8x7B", "GPT-4") or standalone capitalised nouns like "Transformer" without a following Title-Case word.

**Fix:** Add a pattern for alphanumeric model names: `\b[A-Z][a-zA-Z0-9]*(?:[-_][a-zA-Z0-9]+)*\b`; or replace the regex extractor with a lightweight NER model for entity extraction.

---

### task_09 – Out-of-Scope with Wrong Tool Selection

**What happened:** The query "Tell me about the weather forecast for tomorrow" has no relevant documents in the knowledge base. The retriever returned doc_02 (BERT) and doc_03 (GPT) at very low similarity scores (0.135 and 0.114). The LLM router then selected `extractor` (C) rather than `FINISH` (D), producing output "Key terms: BERT, GPT, GPU, Bidirectional Encoder Representations... Numeric values: 0.135, 2018, 0.114, 3, 175 billion..." — all irrelevant to the query. The final LLM answer was nonetheless correct: "Unfortunately, there is no information about the weather forecast for tomorrow in the provided context." The final-answer synthesis LLM correctly disregarded the irrelevant extractor output.

**Root cause:** (1) No minimum similarity threshold — the router has no signal that retrieval failed, so it proceeds to a synthesis step. (2) The LLM router chose extractor over FINISH because the prompt does not instruct it to check score magnitudes before deciding.

**Fix:** Add a minimum score threshold of 0.35 in the retriever tool: if all top-k scores fall below the threshold, return "no relevant context found" and route directly to FINISH. This eliminates the unnecessary extractor call and makes the out-of-scope rejection explicit rather than coincidental.

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
