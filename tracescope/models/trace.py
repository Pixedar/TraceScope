"""
Data models for LLM conversation traces.

TraceEntry  – a single message or reasoning step.
TraceSession – an ordered collection of entries from one conversation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy as np


@dataclass
class TraceEntry:
    """A single message or reasoning step in a conversation.

    Attributes:
        text: The raw text content.
        role: One of "user", "assistant", "system", "reasoning", "tool", "entry".
        step_index: Position in the conversation (0-based, path-local for multi-path).
        session_id: ID of the owning TraceSession.
        embedding: High-dim embedding vector (set after embedding step).
        model_name: Which embedding model produced the embedding.
        timestamp: Unix timestamp (optional).
        path_id: Which path this entry belongs to (None = single-path mode).
        metadata: Arbitrary extra data (token counts, tool calls, etc.).
        scores: Named numeric scores for this entry (e.g. {"emotion_valence": 0.8,
                "error_rate": 0.1}). Fully optional — used for score-based coloring
                in visualization when present.
    """

    text: str
    role: str
    step_index: int
    session_id: str
    embedding: Optional[np.ndarray] = None
    model_name: Optional[str] = None
    timestamp: Optional[float] = None
    path_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)

    def has_embedding(self) -> bool:
        return self.embedding is not None

    def to_dict(self) -> dict:
        d = {
            "text": self.text,
            "role": self.role,
            "step_index": self.step_index,
            "session_id": self.session_id,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "path_id": self.path_id,
            "metadata": self.metadata,
        }
        if self.scores:
            d["scores"] = self.scores
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TraceEntry":
        return cls(
            text=d["text"],
            role=d["role"],
            step_index=d["step_index"],
            session_id=d["session_id"],
            model_name=d.get("model_name"),
            timestamp=d.get("timestamp"),
            path_id=d.get("path_id"),
            metadata=d.get("metadata", {}),
            scores=d.get("scores", {}),
        )


@dataclass
class TraceSession:
    """An ordered collection of TraceEntry objects from one conversation.

    Attributes:
        session_id: Unique identifier for this session.
        label: Human-readable label (e.g. "GPT-4o coding chat").
        entries: Ordered list of conversation entries.
        source_format: How the data was imported ("openai", "anthropic",
                       "plain_text", "reasoning").
        llm_model: The LLM that generated this conversation (optional).
        created_at: Unix timestamp of import time.
        path_scores: Per-path aggregate scores, keyed by path_id.
                     E.g. {0: {"success": 1.0, "cost": 0.05}, 1: {"success": 0.0}}.
                     Fully optional — used for score-based path coloring.
    """

    session_id: str
    label: str
    entries: List[TraceEntry]
    source_format: str
    llm_model: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    path_scores: Dict[int, Dict[str, float]] = field(default_factory=dict)

    @property
    def texts(self) -> List[str]:
        return [e.text for e in self.entries]

    @property
    def roles(self) -> List[str]:
        return [e.role for e in self.entries]

    def embeddings_matrix(self) -> Optional[np.ndarray]:
        """Return (N, D) array of embeddings, or None if not all entries are embedded."""
        if not all(e.has_embedding() for e in self.entries):
            return None
        return np.array([e.embedding for e in self.entries])

    @property
    def score_channels(self) -> List[str]:
        """Return sorted list of all score channel names found across entries and paths."""
        channels = set()
        for e in self.entries:
            channels.update(e.scores.keys())
        for ps in self.path_scores.values():
            channels.update(ps.keys())
        return sorted(channels)

    def to_dict(self) -> dict:
        d = {
            "session_id": self.session_id,
            "label": self.label,
            "entries": [e.to_dict() for e in self.entries],
            "source_format": self.source_format,
            "llm_model": self.llm_model,
            "created_at": self.created_at,
        }
        if self.path_scores:
            # JSON keys must be strings
            d["path_scores"] = {str(k): v for k, v in self.path_scores.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TraceSession":
        entries = [TraceEntry.from_dict(e) for e in d["entries"]]
        raw_ps = d.get("path_scores", {})
        path_scores = {int(k): v for k, v in raw_ps.items()} if raw_ps else {}
        return cls(
            session_id=d["session_id"],
            label=d["label"],
            entries=entries,
            source_format=d["source_format"],
            llm_model=d.get("llm_model"),
            created_at=d.get("created_at", 0.0),
            path_scores=path_scores,
        )

    def __len__(self) -> int:
        return len(self.entries)
