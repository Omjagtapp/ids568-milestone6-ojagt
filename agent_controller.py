"""
Multi-Tool Agent Controller with ReAct-style reasoning traces.

Tools available:
  1. retriever  – RAG pipeline lookup (FAISS + sentence-transformers)
  2. summarizer – Distil a set of retrieved passages into a short summary
  3. extractor  – Pull specific facts / named entities from context

Tool selection is driven by a lightweight LLM-based router (Ollama); falls back
to keyword heuristics when Ollama is unavailable.
"""

import json
import re
import time
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag_pipeline import (
    DOCUMENTS,
    EVALUATION_QUERIES,
    RAGPipeline,
    Chunk,
)

# ---------------------------------------------------------------------------
# Agent state and trace data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolExecution:
    tool_name: str
    tool_input: str
    tool_output: str
    latency_ms: float
    success: bool


@dataclass
class AgentTrace:
    task_id: str
    task: str
    steps: List[Dict] = field(default_factory=list)  # each step: thought, action, observation
    final_answer: str = ""
    total_latency_ms: float = 0.0
    tool_calls: List[ToolExecution] = field(default_factory=list)
    success: bool = True
    failure_reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "tool_calls": [
                {
                    "tool": tc.tool_name,
                    "input": tc.tool_input,
                    "output_snippet": tc.tool_output[:300],
                    "latency_ms": round(tc.latency_ms, 2),
                    "success": tc.success,
                }
                for tc in self.tool_calls
            ],
            "success": self.success,
            "failure_reason": self.failure_reason,
        }


@dataclass
class AgentState:
    task_id: str
    task: str
    context: List[str] = field(default_factory=list)   # accumulates observations
    iteration: int = 0
    max_iterations: int = 5
    done: bool = False


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

# Shared pipeline instance (built once, reused across tasks)
_pipeline: Optional[RAGPipeline] = None

def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(strategy="recursive", chunk_size=512, k=5)
        _pipeline.ingest(DOCUMENTS)
    return _pipeline


def tool_retriever(query: str) -> Tuple[str, float]:
    """Retrieve relevant passages for a query using the RAG pipeline."""
    t0 = time.perf_counter()
    pipe = _get_pipeline()
    q_emb, _ = __import__("rag_pipeline").embed_query(query)
    retrieved, scores, _ = __import__("rag_pipeline").retrieve(q_emb, pipe.index, pipe.chunks, k=5)
    passages = [
        f"[{c.doc_id} | score={s:.3f}] {c.text[:200]}"
        for c, s in zip(retrieved, scores)
    ]
    output = "\n".join(passages)
    latency_ms = (time.perf_counter() - t0) * 1000
    return output, latency_ms


def tool_summarizer(text: str, llm_model: str = "mistral:7b-instruct") -> Tuple[str, float]:
    """Summarize the given text in 2-3 sentences."""
    prompt = (
        f"Summarize the following text in 2-3 concise sentences:\n\n{text[:1500]}\n\nSummary:"
    )
    t0 = time.perf_counter()
    try:
        import ollama
        resp = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = resp["message"]["content"].strip()
    except Exception as exc:
        # Fallback: first two sentences of the text
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        summary = " ".join(sentences[:2])
        summary += f"  [Note: Ollama unavailable – {exc}]"
    latency_ms = (time.perf_counter() - t0) * 1000
    return summary, latency_ms


def tool_extractor(text: str, entity_type: str = "concepts") -> Tuple[str, float]:
    """Extract named entities or key concepts from text using regex heuristics."""
    t0 = time.perf_counter()
    # Capitalised phrases (potential proper nouns / model names)
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    capitalised = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:billion|million|%|B|M|k)?\b', text)
    unique_terms = list(dict.fromkeys(acronyms + capitalised))[:15]
    output = (
        f"Key terms: {', '.join(unique_terms) if unique_terms else 'none found'}. "
        f"Numeric values: {', '.join(numbers[:10]) if numbers else 'none found'}."
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return output, latency_ms


TOOL_REGISTRY = {
    "retriever": tool_retriever,
    "summarizer": tool_summarizer,
    "extractor": tool_extractor,
}

TOOL_DESCRIPTIONS = {
    "retriever": "Search the knowledge base for passages relevant to a query.",
    "summarizer": "Produce a short summary of a block of text.",
    "extractor": "Extract key named entities and numeric facts from text.",
}


# ---------------------------------------------------------------------------
# Tool selection: LLM-based with heuristic fallback
# ---------------------------------------------------------------------------

SELECTION_PROMPT_TEMPLATE = """You are an AI assistant that selects the best tool.

Available tools:
{tool_list}

Task: {task}
Conversation so far:
{context}

Which tool should be called next? Reply with EXACTLY one of: retriever, summarizer, extractor, FINISH.
If the task is answered, reply FINISH."""


def select_tool_llm(
    task: str,
    context: List[str],
    llm_model: str = "mistral:7b-instruct",
) -> str:
    tool_list = "\n".join(f"  {k}: {v}" for k, v in TOOL_DESCRIPTIONS.items())
    prompt = SELECTION_PROMPT_TEMPLATE.format(
        tool_list=tool_list,
        task=task,
        context="\n".join(context[-4:]) if context else "(none)",
    )
    try:
        import ollama
        resp = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp["message"]["content"].strip().lower()
        for tool in list(TOOL_REGISTRY.keys()) + ["finish"]:
            if tool in raw:
                return tool
        return "retriever"
    except Exception:
        return _select_tool_heuristic(task, context)


def _select_tool_heuristic(task: str, context: List[str]) -> str:
    """Keyword-based fallback tool selector."""
    task_lower = task.lower()
    if not context:
        return "retriever"
    last_obs = context[-1].lower() if context else ""
    if any(w in task_lower for w in ["summarize", "summary", "briefly", "overview"]):
        return "summarizer"
    if any(w in task_lower for w in ["extract", "entities", "names", "numbers", "facts"]):
        return "extractor"
    if any(w in task_lower for w in ["compare", "difference", "versus", "vs"]):
        if "retriever" in last_obs or len(context) < 2:
            return "retriever"
        return "summarizer"
    if len(context) >= 2:
        return "FINISH"
    return "retriever"


# ---------------------------------------------------------------------------
# ReAct-style agent loop
# ---------------------------------------------------------------------------

def _build_thought(state: AgentState, selected_tool: str) -> str:
    if state.iteration == 0:
        return f"I need to answer: '{state.task}'. I'll start by retrieving relevant information."
    if selected_tool == "summarizer":
        return "I have retrieved context. Now I will summarize it to form a concise answer."
    if selected_tool == "extractor":
        return "I will extract key entities and facts from the gathered context."
    if selected_tool == "FINISH":
        return "I have enough information to provide a final answer."
    return f"I will use the {selected_tool} tool to gather more information."


def run_agent_task(
    task_id: str,
    task: str,
    llm_model: str = "mistral:7b-instruct",
    use_llm_routing: bool = True,
) -> AgentTrace:
    t_start = time.perf_counter()
    state = AgentState(task_id=task_id, task=task)
    trace = AgentTrace(task_id=task_id, task=task)

    while not state.done and state.iteration < state.max_iterations:
        state.iteration += 1

        # --- Tool selection ---
        if use_llm_routing:
            selected_tool = select_tool_llm(task, state.context, llm_model)
        else:
            selected_tool = _select_tool_heuristic(task, state.context)

        thought = _build_thought(state, selected_tool)

        if selected_tool == "FINISH":
            state.done = True
            trace.steps.append({
                "iteration": state.iteration,
                "thought": thought,
                "action": "FINISH",
                "observation": "Task complete.",
            })
            break

        # --- Execute tool ---
        tool_fn = TOOL_REGISTRY.get(selected_tool)
        if tool_fn is None:
            state.done = True
            trace.success = False
            trace.failure_reason = f"Unknown tool selected: {selected_tool}"
            break

        try:
            # Determine tool input
            if selected_tool == "retriever":
                tool_input = task
                output, lat = tool_fn(tool_input)
            elif selected_tool == "summarizer":
                tool_input = "\n".join(state.context)
                output, lat = tool_fn(tool_input, llm_model)
            elif selected_tool == "extractor":
                tool_input = "\n".join(state.context)
                output, lat = tool_fn(tool_input)
            else:
                tool_input = task
                output, lat = tool_fn(tool_input)

            success = True
        except Exception as exc:
            output = f"Tool error: {exc}"
            lat = 0.0
            success = False

        exec_record = ToolExecution(
            tool_name=selected_tool,
            tool_input=tool_input[:200],
            tool_output=output,
            latency_ms=lat,
            success=success,
        )
        trace.tool_calls.append(exec_record)
        state.context.append(f"[{selected_tool}] {output}")

        trace.steps.append({
            "iteration": state.iteration,
            "thought": thought,
            "action": selected_tool,
            "action_input": tool_input[:200],
            "observation": output[:400],
        })

        # Decide if we have enough after retrieval + one more step
        if state.iteration >= 2 and selected_tool in ("summarizer", "extractor"):
            state.done = True

    # --- Final answer synthesis ---
    if state.context:
        context_text = "\n".join(state.context)
        final_prompt = (
            f"Based on the following context, give a final concise answer to: '{task}'\n\n"
            f"Context:\n{context_text[:2000]}\n\nFinal Answer:"
        )
        try:
            import ollama
            resp = ollama.chat(
                model=llm_model,
                messages=[{"role": "user", "content": final_prompt}],
            )
            trace.final_answer = resp["message"]["content"].strip()
        except Exception:
            # Fallback: extract first meaningful observation
            trace.final_answer = state.context[-1][:500] if state.context else "No answer generated."
    else:
        trace.final_answer = "No context gathered."

    trace.total_latency_ms = (time.perf_counter() - t_start) * 1000
    return trace


# ---------------------------------------------------------------------------
# 10 evaluation tasks
# ---------------------------------------------------------------------------

AGENT_TASKS = [
    {
        "id": "task_01",
        "task": "Retrieve and summarize the key ideas behind Retrieval-Augmented Generation.",
    },
    {
        "id": "task_02",
        "task": "Find information about BERT and GPT, then compare their pre-training objectives.",
    },
    {
        "id": "task_03",
        "task": "Explain what chunking strategies are used in RAG pipelines and why overlap matters.",
    },
    {
        "id": "task_04",
        "task": "Extract key numeric facts about GPT-3 and DistilBERT model sizes.",
    },
    {
        "id": "task_05",
        "task": "Summarize how FAISS enables efficient similarity search for dense retrieval.",
    },
    {
        "id": "task_06",
        "task": "Retrieve information about hallucination in LLMs and extract the types mentioned.",
    },
    {
        "id": "task_07",
        "task": "What is the ReAct prompting pattern and how does it differ from chain-of-thought?",
    },
    {
        "id": "task_08",
        "task": "Retrieve and summarize how LoRA works for parameter-efficient fine-tuning.",
    },
    {
        "id": "task_09",
        "task": "Tell me about the weather forecast for tomorrow.",  # out-of-scope / edge case
    },
    {
        "id": "task_10",
        "task": "Find information about Mixture of Experts and extract the model names mentioned.",
    },
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_all_tasks(save_traces: bool = True) -> List[AgentTrace]:
    os.makedirs("agent_traces", exist_ok=True)
    print("\n=== Agent Evaluation – 10 Tasks ===")
    print("Building RAG index (one-time)…")
    _get_pipeline()
    print("Index ready.\n")

    traces: List[AgentTrace] = []
    for task_info in AGENT_TASKS:
        print(f"[{task_info['id']}] {task_info['task'][:70]}…")
        trace = run_agent_task(
            task_id=task_info["id"],
            task=task_info["task"],
            use_llm_routing=False,   # use heuristic by default; flip to True with Ollama
        )
        traces.append(trace)

        # Save individual trace JSON
        if save_traces:
            path = f"agent_traces/{task_info['id']}.json"
            with open(path, "w") as f:
                json.dump(trace.to_dict(), f, indent=2)

        tool_names = [tc.tool_name for tc in trace.tool_calls]
        print(
            f"  Tools used: {tool_names} | "
            f"Steps: {len(trace.steps)} | "
            f"Latency: {trace.total_latency_ms:.0f}ms | "
            f"Success: {trace.success}"
        )
        if not trace.success:
            print(f"  Failure: {trace.failure_reason}")

    print(f"\nAll {len(traces)} traces saved to agent_traces/")
    return traces


if __name__ == "__main__":
    run_all_tasks(save_traces=True)
