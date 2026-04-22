"""
Multi-Tool Agent Controller with ReAct-style reasoning traces.

Tasks are loaded from config/agent_tasks.json.
The retriever tool reuses the RAG pipeline backed by docs/*.txt.

Usage:
    python3 agent_controller.py
    python3 agent_controller.py --tasks config/agent_tasks.json
    python3 agent_controller.py --docs_dir my_docs/ --llm_model llama3:8b-instruct
    python3 agent_controller.py --llm_routing          # use Ollama for tool selection
    python3 agent_controller.py --traces_dir out/
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_pipeline import (
    RAGPipeline,
    Chunk,
    load_documents,
    embed_query,
    retrieve,
)


# ---------------------------------------------------------------------------
# Data structures
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
    steps: List[Dict] = field(default_factory=list)
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
    context: List[str] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 5
    done: bool = False


# ---------------------------------------------------------------------------
# Shared RAG pipeline (built once per process)
# ---------------------------------------------------------------------------

_pipeline: Optional[RAGPipeline] = None


def _get_pipeline(docs_dir: str, model_name: str) -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        docs = load_documents(docs_dir)
        _pipeline = RAGPipeline(strategy="recursive", chunk_size=512, model_name=model_name)
        _pipeline.ingest(docs)
    return _pipeline


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_retriever(query: str, docs_dir: str, model_name: str) -> Tuple[str, float]:
    t0 = time.perf_counter()
    pipe = _get_pipeline(docs_dir, model_name)
    q_emb, _ = embed_query(query, model_name)
    hits, scores, _ = retrieve(q_emb, pipe.index, pipe.chunks, k=5)
    passages = [
        f"[{c.doc_id} | score={s:.3f}] {c.text[:200]}"
        for c, s in zip(hits, scores)
    ]
    return "\n".join(passages), (time.perf_counter() - t0) * 1000


def tool_summarizer(text: str, llm_model: str) -> Tuple[str, float]:
    prompt = f"Summarize the following text in 2-3 concise sentences:\n\n{text[:1500]}\n\nSummary:"
    t0 = time.perf_counter()
    try:
        import ollama
        resp = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
        summary = resp["message"]["content"].strip()
    except Exception as exc:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        summary = " ".join(sentences[:2]) + f"  [Ollama unavailable – {exc}]"
    return summary, (time.perf_counter() - t0) * 1000


def tool_extractor(text: str) -> Tuple[str, float]:
    t0 = time.perf_counter()
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    capitalised = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:billion|million|%|B|M|k)?\b', text)
    terms = list(dict.fromkeys(acronyms + capitalised))[:15]
    output = (
        f"Key terms: {', '.join(terms) if terms else 'none'}. "
        f"Numeric values: {', '.join(numbers[:10]) if numbers else 'none'}."
    )
    return output, (time.perf_counter() - t0) * 1000


TOOL_DESCRIPTIONS = {
    "retriever": "Search the knowledge base for passages relevant to a query.",
    "summarizer": "Produce a short summary of a block of text.",
    "extractor": "Extract key named entities and numeric facts from text.",
}

# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

_SELECTION_PROMPT = """\
You are a tool-routing agent. Choose the next tool to call.

Tools (pick exactly one):
  A) retriever   – search knowledge base for relevant passages
  B) summarizer  – condense gathered context into a concise answer
  C) extractor   – pull named entities / numeric facts from context
  D) FINISH      – enough information has been gathered; stop

Task: {task}

Steps taken so far ({n_steps} steps):
{context}

Rules:
- If no information has been gathered yet, choose A (retriever).
- If the task asks to "summarize" or "compare" and you already have retrieved passages, choose B (summarizer).
- If the task asks to "extract", "find names", or "find numbers" and you have passages, choose C (extractor).
- If the task is an out-of-scope question or the context clearly answers the task, choose D (FINISH).
- Do NOT choose A (retriever) if you already retrieved information in a previous step.

Reply with ONLY the letter: A, B, C, or D."""


def select_tool_llm(task: str, context: List[str], llm_model: str) -> str:
    prompt = _SELECTION_PROMPT.format(
        task=task,
        n_steps=len(context),
        context="\n".join(context[-3:]) if context else "(none yet)",
    )
    try:
        import ollama
        resp = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"].strip().upper()
        # Parse letter answer first
        if raw.startswith("A"):
            return "retriever"
        if raw.startswith("B"):
            return "summarizer"
        if raw.startswith("C"):
            return "extractor"
        if raw.startswith("D"):
            return "FINISH"
        # Fall back to keyword scan on full response
        raw_lower = raw.lower()
        if "finish" in raw_lower or "done" in raw_lower or "enough" in raw_lower:
            return "FINISH"
        if "summar" in raw_lower:
            return "summarizer"
        if "extract" in raw_lower:
            return "extractor"
        if "retriev" in raw_lower:
            return "retriever"
        # Default: if we have context, synthesize; otherwise retrieve
        return "FINISH" if context else "retriever"
    except Exception:
        return _select_tool_heuristic(task, context)


def _select_tool_heuristic(task: str, context: List[str]) -> str:
    tl = task.lower()
    if not context:
        return "retriever"
    if any(w in tl for w in ["summarize", "summary", "briefly", "overview"]):
        return "summarizer"
    if any(w in tl for w in ["extract", "entities", "names", "numbers", "facts"]):
        return "extractor"
    if any(w in tl for w in ["compare", "difference", "versus", "vs"]):
        return "summarizer" if len(context) >= 2 else "retriever"
    if len(context) >= 2:
        return "FINISH"
    return "retriever"


# ---------------------------------------------------------------------------
# ReAct agent loop
# ---------------------------------------------------------------------------

def _thought(state: AgentState, tool: str) -> str:
    if state.iteration == 0:
        return f"I need to answer: '{state.task}'. I will start by retrieving relevant information."
    if tool == "summarizer":
        return "I have context. Now I will summarize it into a concise answer."
    if tool == "extractor":
        return "I will extract key entities and numeric facts from the gathered context."
    if tool == "FINISH":
        return "I have enough information to produce a final answer."
    return f"I will call {tool} to gather more information."


def run_agent_task(
    task_id: str,
    task: str,
    docs_dir: str,
    model_name: str,
    llm_model: str,
    use_llm_routing: bool,
    traces_dir: str,
) -> AgentTrace:
    t_start = time.perf_counter()
    state = AgentState(task_id=task_id, task=task)
    trace = AgentTrace(task_id=task_id, task=task)

    while not state.done and state.iteration < state.max_iterations:
        state.iteration += 1

        selected = (
            select_tool_llm(task, state.context, llm_model)
            if use_llm_routing
            else _select_tool_heuristic(task, state.context)
        )
        thought = _thought(state, selected)

        if selected == "FINISH":
            state.done = True
            trace.steps.append({
                "iteration": state.iteration,
                "thought": thought,
                "action": "FINISH",
                "observation": "Task complete.",
            })
            break

        try:
            if selected == "retriever":
                tool_input = task
                output, lat = tool_retriever(tool_input, docs_dir, model_name)
            elif selected == "summarizer":
                tool_input = "\n".join(state.context)
                output, lat = tool_summarizer(tool_input, llm_model)
            elif selected == "extractor":
                tool_input = "\n".join(state.context)
                output, lat = tool_extractor(tool_input)
            else:
                raise ValueError(f"Unknown tool: {selected}")
            success = True
        except Exception as exc:
            output, lat, success = f"Tool error: {exc}", 0.0, False

        trace.tool_calls.append(ToolExecution(
            tool_name=selected,
            tool_input=tool_input[:200],
            tool_output=output,
            latency_ms=lat,
            success=success,
        ))
        state.context.append(f"[{selected}] {output}")
        trace.steps.append({
            "iteration": state.iteration,
            "thought": thought,
            "action": selected,
            "action_input": tool_input[:200],
            "observation": output[:400],
        })

        # Stop after a synthesizer tool
        if state.iteration >= 2 and selected in ("summarizer", "extractor"):
            state.done = True

    # Final answer
    ctx = "\n".join(state.context)
    final_prompt = (
        f"Based on the following context, give a concise final answer to: '{task}'\n\n"
        f"Context:\n{ctx[:2000]}\n\nFinal Answer:"
    )
    try:
        import ollama
        resp = ollama.chat(model=llm_model, messages=[{"role": "user", "content": final_prompt}])
        trace.final_answer = resp["message"]["content"].strip()
    except Exception:
        trace.final_answer = state.context[-1][:500] if state.context else "No answer generated."

    trace.total_latency_ms = (time.perf_counter() - t_start) * 1000

    # Write trace to file
    out_path = Path(traces_dir) / f"{task_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")

    return trace


# ---------------------------------------------------------------------------
# Load tasks from file
# ---------------------------------------------------------------------------

def load_tasks(tasks_path: str) -> List[Dict]:
    path = Path(tasks_path)
    if not path.exists():
        raise FileNotFoundError(f"tasks file not found: {path.resolve()}")
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    print(f"Loaded {len(tasks)} agent tasks from {path.resolve()}")
    return tasks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Tool Agent Evaluation")
    p.add_argument("--docs_dir", default="docs", help="Directory of .txt documents")
    p.add_argument("--tasks", default="config/agent_tasks.json", help="Path to tasks JSON")
    p.add_argument("--llm_model", default="llama3.1:8b")
    p.add_argument("--model_name", default="all-MiniLM-L6-v2", help="Embedding model")
    p.add_argument("--traces_dir", default="agent_traces", help="Directory to write trace files")
    p.add_argument(
        "--llm_routing", action="store_true",
        help="Use Ollama LLM for tool selection (default: heuristic)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tasks = load_tasks(args.tasks)

    print("\n=== Agent Evaluation ===")
    print(f"Building RAG index from {args.docs_dir}…")
    _get_pipeline(args.docs_dir, args.model_name)
    print("Index ready.\n")

    traces: List[AgentTrace] = []
    for task_info in tasks:
        print(f"[{task_info['id']}] {task_info['task'][:70]}…")
        trace = run_agent_task(
            task_id=task_info["id"],
            task=task_info["task"],
            docs_dir=args.docs_dir,
            model_name=args.model_name,
            llm_model=args.llm_model,
            use_llm_routing=args.llm_routing,
            traces_dir=args.traces_dir,
        )
        traces.append(trace)
        tool_names = [tc.tool_name for tc in trace.tool_calls]
        print(
            f"  Tools: {tool_names} | Steps: {len(trace.steps)} | "
            f"Latency: {trace.total_latency_ms:.0f}ms | Success: {trace.success}"
        )
        if not trace.success:
            print(f"  Failure: {trace.failure_reason}")

    print(f"\nAll {len(traces)} traces written to {args.traces_dir}/")


if __name__ == "__main__":
    main()
