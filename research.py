"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphRecursionError

from agents import (FINALIZER_PATH, NOTES_DIR, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR,
                    build_lead_agent, source_problems)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
# step cap of the lead's LangGraph graph (~2 steps per model->tool turn); subagents are capped by middleware (agents.py)
RECURSION_LIMIT = 400


class ProgressLog(BaseCallbackHandler):
    """Prints one line per tool call (lead and subagents) to stderr, so a slow or stuck run can be diagnosed."""

    def __init__(self):
        self.start = time.monotonic()

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "?"
        detail = " ".join(str(input_str).split())[:100]
        print(f"[{time.monotonic() - self.start:6.0f}s] {name}: {detail}", file=sys.stderr, flush=True)


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^a-z0-9]+", "-", (topic or "").lower()).strip("-")[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Research topic: {topic}\n\n"
            "Produce the survey report following your workflow: plan with write_todos, delegate the sub-questions to "
            "researchers in parallel, check and merge their notes into sources.json (at least 3 source families), "
            "write the report body, run the finalizer, run the validator until it prints OK, then have the "
            "citation-checker spot-check a few claims.")


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.
    Lead messages only: subagent tokens are not included, so this undercounts the real cost."""
    calls, tokens = Counter(), {"input": 0, "output": 0}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0)
        tokens["output"] += usage.get("output_tokens", 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls.get("task", 0),
            "tool_calls": dict(calls), "tokens": tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.
    A failed run (missing/empty report, missing/invalid sources.json) raises RuntimeError and writes nothing."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report, sources_raw = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report or not report.strip():
        raise RuntimeError(f"the agent produced no report at {REPORT_PATH}")
    if not sources_raw:
        raise RuntimeError(f"the agent produced no {SOURCES_PATH}")
    try:
        sources = json.loads(sources_raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid sources.json: {exc}") from exc
    if not isinstance(sources, list) or not all(isinstance(entry, dict) for entry in sources):
        raise RuntimeError("sources.json is not a JSON list of objects")
    problems = source_problems(sources)
    if problems:  # e.g. a URL no tool returned (typed from memory), a mislabelled family: never publish it
        raise RuntimeError("sources.json failed verification:\n  " + "\n  ".join(problems))

    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({str(entry.get("source")) for entry in sources if entry.get("source")})}
    slug = slugify(topic)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    # sources.json and the report are written byte for byte as downloaded from the sandbox
    (reports_dir / f"{slug}.sources.json").write_bytes(sources_raw)
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_bytes(report)
    return report_path


def make_fast_model():
    """make_model() (provided, not edited), plus: LAB_REASONING_OFF=1 turns the model's reasoning off on OpenRouter.
    A reasoning model "thinks" before every tool call, which made a run take 25+ minutes instead of a few."""
    model = make_model()
    if os.getenv("LAB_REASONING_OFF", "").strip().lower() in {"1", "true", "yes"} and hasattr(model, "extra_body"):
        model.extra_body = {**(model.extra_body or {}), "reasoning": {"enabled": False}}
    return model


def model_name_of(model):
    return getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "unknown")


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    model = make_fast_model()
    start = time.monotonic()
    with open_sandbox() as backend:  # the sandbox is always stopped and removed, even on errors
        backend.execute(f"mkdir -p {NOTES_DIR} {WORKDIR}/report")
        upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
        agent = build_lead_agent(backend, model)
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": RECURSION_LIMIT, "callbacks": [ProgressLog()]})
        except GraphRecursionError as exc:
            print(f"FAILED: the lead agent hit the recursion limit ({exc})", file=sys.stderr)
            return 1
        try:
            report_path = save_outputs(backend, topic, result["messages"], time.monotonic() - start,
                                       model_name_of(model))
        except RuntimeError as exc:
            stats = summarize(result["messages"], time.monotonic() - start, model_name_of(model))
            last = str(result["messages"][-1].content)[-800:] if result["messages"] else ""
            print(f"FAILED: {exc}\nlead tool calls: {stats['tool_calls']}\nlead's last message: {last}",
                  file=sys.stderr)
            return 1
    print(f"report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
