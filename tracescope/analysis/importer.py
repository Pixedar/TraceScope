"""
Import data from various formats into TraceSession objects.

Supported formats:
  - OpenAI chat format: {"messages": [{"role": ..., "content": ...}, ...]}
  - Anthropic format: {"content": [...], "role": ...} messages
  - Plain list of texts: ["text1", "text2", ...] or List[str] in code
  - Plain text: newline-separated messages
  - Reasoning traces: <thinking>...</thinking> blocks

TraceScope is universal — it analyzes any ordered collection of texts:
chatbot conversations, news headlines, research abstracts, log entries, etc.
"""

from __future__ import annotations

import json
import logging
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from tracescope.models.trace import TraceEntry, TraceSession

logger = logging.getLogger(__name__)


def _make_session_id() -> str:
    return str(uuid.uuid4())[:8]


def import_openai(
    data: Union[dict, list],
    label: str = "OpenAI conversation",
    session_id: Optional[str] = None,
) -> TraceSession:
    """Import from OpenAI chat format.

    Accepts either:
      - {"messages": [{"role": "...", "content": "..."}, ...]}
      - [{"role": "...", "content": "..."}, ...]
    """
    sid = session_id or _make_session_id()

    if isinstance(data, dict):
        messages = data.get("messages", [])
    else:
        messages = data

    entries = []
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle content blocks (text + images)
            content = " ".join(
                block.get("text", "") for block in content if block.get("type") == "text"
            )
        if not content or not content.strip():
            continue
        entries.append(
            TraceEntry(
                text=content.strip(),
                role=msg.get("role", "unknown"),
                step_index=i,
                session_id=sid,
                metadata={k: v for k, v in msg.items() if k not in ("role", "content")},
            )
        )

    return TraceSession(
        session_id=sid,
        label=label,
        entries=entries,
        source_format="openai",
        llm_model=data.get("model") if isinstance(data, dict) else None,
    )


def import_anthropic(
    data: Union[dict, list],
    label: str = "Anthropic conversation",
    session_id: Optional[str] = None,
) -> TraceSession:
    """Import from Anthropic message format.

    Accepts either:
      - {"messages": [{"role": "...", "content": "..."}, ...]}
      - [{"role": "...", "content": [{"type": "text", "text": "..."}]}, ...]
    """
    sid = session_id or _make_session_id()

    if isinstance(data, dict):
        messages = data.get("messages", [])
        model = data.get("model")
    else:
        messages = data
        model = None

    entries = []
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if isinstance(content, str) and not content.strip():
            continue
        entries.append(
            TraceEntry(
                text=str(content).strip(),
                role=msg.get("role", "unknown"),
                step_index=i,
                session_id=sid,
            )
        )

    return TraceSession(
        session_id=sid,
        label=label,
        entries=entries,
        source_format="anthropic",
        llm_model=model,
    )


def import_plain_text(
    text: str,
    delimiter: str = "\n\n",
    label: str = "Plain text",
    session_id: Optional[str] = None,
) -> TraceSession:
    """Import from plain text split by delimiter."""
    sid = session_id or _make_session_id()
    chunks = [c.strip() for c in text.split(delimiter) if c.strip()]

    entries = []
    for i, chunk in enumerate(chunks):
        role = "user" if i % 2 == 0 else "assistant"
        entries.append(
            TraceEntry(
                text=chunk,
                role=role,
                step_index=i,
                session_id=sid,
            )
        )

    return TraceSession(
        session_id=sid,
        label=label,
        entries=entries,
        source_format="plain_text",
    )


def import_reasoning_trace(
    text: str,
    label: str = "Reasoning trace",
    session_id: Optional[str] = None,
) -> TraceSession:
    """Import reasoning traces with <thinking>...</thinking> blocks.

    Splits on <thinking> tags and regular text blocks.
    """
    sid = session_id or _make_session_id()
    import re

    entries = []
    idx = 0

    # Split into thinking and non-thinking blocks
    parts = re.split(r"(<thinking>.*?</thinking>)", text, flags=re.DOTALL)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("<thinking>"):
            inner = part[len("<thinking>"):-len("</thinking>")].strip()
            if inner:
                entries.append(
                    TraceEntry(
                        text=inner,
                        role="reasoning",
                        step_index=idx,
                        session_id=sid,
                    )
                )
                idx += 1
        else:
            entries.append(
                TraceEntry(
                    text=part,
                    role="assistant",
                    step_index=idx,
                    session_id=sid,
                )
            )
            idx += 1

    return TraceSession(
        session_id=sid,
        label=label,
        entries=entries,
        source_format="reasoning",
    )


def from_list(
    texts: List[str],
    label: str = "Text collection",
    session_id: Optional[str] = None,
    entry_scores: Optional[List[Dict[str, float]]] = None,
) -> TraceSession:
    """Import from a plain list of texts.

    Works with any collection: news headlines, research abstracts,
    article summaries, sentences, log entries, or any other text data.
    No role or format assumptions — each text becomes one entry.

    Args:
        texts: List of text strings to analyze.
        label: Human-readable label for this session.
        session_id: Optional session ID (auto-generated if omitted).
        entry_scores: Optional per-entry score dicts, same length as texts.
                      E.g. [{"emotion": 0.8}, {"emotion": -0.3}, ...].

    Returns:
        TraceSession ready for the analysis pipeline.
    """
    if not texts:
        raise ValueError("texts list is empty — provide at least 1 text")

    sid = session_id or _make_session_id()
    entries = []
    skipped = 0
    for i, text in enumerate(texts):
        text = text.strip() if isinstance(text, str) else str(text).strip()
        if not text:
            skipped += 1
            continue
        scores = entry_scores[i] if entry_scores and i < len(entry_scores) else {}
        entries.append(
            TraceEntry(
                text=text,
                role="entry",
                step_index=i,
                session_id=sid,
                scores=scores,
            )
        )

    if skipped > 0:
        logger.warning(f"Skipped {skipped} empty string(s) out of {len(texts)} inputs")

    return TraceSession(
        session_id=sid,
        label=label,
        entries=entries,
        source_format="list",
    )


def from_lists(
    paths: List[List[str]],
    labels: Optional[List[str]] = None,
    label: str = "Multi-path collection",
    session_id: Optional[str] = None,
    entry_scores: Optional[List[List[Dict[str, float]]]] = None,
    path_scores: Optional[Dict[int, Dict[str, float]]] = None,
) -> TraceSession:
    """Import multiple independent semantic paths into one unified session.

    Each path is a separate ordered sequence of texts. The analysis pipeline
    will embed, cluster, and reduce all points together, but the MDN flow
    model will correctly skip velocity computation at path boundaries.

    Args:
        paths: List of text-list paths, e.g. [["a","b","c"], ["d","e","f"]].
        labels: Optional per-path labels (for metadata only).
        label: Human-readable label for the combined session.
        session_id: Optional session ID (auto-generated if omitted).
        entry_scores: Optional per-entry scores, shape matches paths.
                      E.g. [[{"confidence": 0.9}, ...], [{"confidence": 0.5}, ...]].
        path_scores: Optional per-path aggregate scores.
                     E.g. {0: {"success": 1.0, "cost": 0.05}, 1: {"success": 0.0}}.

    Returns:
        TraceSession with path_id set on each entry.

    Example::

        session = from_lists([
            ["Fed holds rates", "Tech earnings surge", "Housing cools"],
            ["Climate summit", "Quantum computing", "Mars rover update"],
        ], path_scores={0: {"success": 1.0}, 1: {"success": 0.5}})
    """
    if not paths:
        raise ValueError("paths list is empty — provide at least 1 path")
    for i, p in enumerate(paths):
        if not p:
            raise ValueError(f"Path {i} is empty — every path must have at least 1 text")

    sid = session_id or _make_session_id()
    entries: List[TraceEntry] = []

    for path_idx, path_texts in enumerate(paths):
        for step, text in enumerate(path_texts):
            text = text.strip() if isinstance(text, str) else str(text).strip()
            if not text:
                continue
            scores = {}
            if entry_scores and path_idx < len(entry_scores):
                path_entry_scores = entry_scores[path_idx]
                if path_entry_scores and step < len(path_entry_scores):
                    scores = path_entry_scores[step]
            entries.append(
                TraceEntry(
                    text=text,
                    role="entry",
                    step_index=step,
                    session_id=sid,
                    path_id=path_idx,
                    scores=scores,
                    metadata={
                        "path_label": (labels[path_idx]
                                       if labels and path_idx < len(labels)
                                       else f"Path {path_idx}")
                    },
                )
            )

    return TraceSession(
        session_id=sid,
        label=label,
        entries=entries,
        source_format="multi_path",
        path_scores=path_scores or {},
    )


def auto_import(
    file_path: Union[str, Path],
    label: Optional[str] = None,
    session_id: Optional[str] = None,
) -> TraceSession:
    """Auto-detect format and import from file.

    Supports .json and .txt files. JSON files can be:
      - OpenAI/Anthropic chat format ({"messages": [...]})
      - List of message dicts ([{"role": ..., "content": ...}, ...])
      - Plain list of strings (["text1", "text2", ...])
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")
    default_label = label or path.stem

    if path.suffix == ".json":
        data = json.loads(text)

        # Multi-path format: {"paths": [["a","b"], ["c","d"]]}
        if isinstance(data, dict) and "paths" in data:
            p = data["paths"]
            if isinstance(p, list) and p and isinstance(p[0], list):
                path_labels = data.get("labels", None)
                return from_lists(p, labels=path_labels,
                                  label=default_label, session_id=session_id)

        # Check for OpenAI format
        if isinstance(data, dict) and "messages" in data:
            messages = data["messages"]
            if messages and isinstance(messages[0].get("content", ""), str):
                return import_openai(data, label=default_label, session_id=session_id)
            else:
                # Could be Anthropic with content blocks
                return import_anthropic(data, label=default_label, session_id=session_id)

        # List of message dicts
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "role" in data[0]:
                return import_openai(data, label=default_label, session_id=session_id)

        # Plain list of strings (news headlines, abstracts, sentences, etc.)
        if isinstance(data, list) and data and isinstance(data[0], str):
            return from_list(data, label=default_label, session_id=session_id)

        raise ValueError(f"Unrecognized JSON format in {file_path}")

    # Plain text
    if "<thinking>" in text:
        return import_reasoning_trace(text, label=default_label, session_id=session_id)

    return import_plain_text(text, label=default_label, session_id=session_id)
