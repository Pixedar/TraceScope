"""
ChromaDB vector store for persistent embedding storage.

Collections are named {session_id}__{model_name} so the same conversation
can be embedded with different models and compared.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import chromadb


class VectorStore:
    """ChromaDB wrapper for storing and retrieving embeddings.

    Args:
        persist_dir: Directory for ChromaDB persistent storage.
    """

    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)

    @staticmethod
    def _collection_name(session_id: str, model_name: str) -> str:
        """Build a collection name from session ID and model name.

        ChromaDB collection names must be 3-63 chars, start/end with
        alphanumeric, and contain only alphanumerics, underscores, hyphens.
        """
        raw = f"{session_id}__{model_name}"
        sanitized = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw)
        if len(sanitized) < 3:
            sanitized = sanitized + "___"
        return sanitized[:63]

    def store_embeddings(
        self,
        session_id: str,
        model_name: str,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[dict]] = None,
    ):
        """Store embeddings for a session+model pair.

        Args:
            session_id: Session identifier.
            model_name: Embedding model name.
            texts: List of text strings (N).
            embeddings: (N, D) array of embedding vectors.
            metadatas: Optional per-entry metadata dicts.
        """
        col_name = self._collection_name(session_id, model_name)
        collection = self._client.get_or_create_collection(
            name=col_name,
            metadata={"session_id": session_id, "model_name": model_name},
        )

        ids = [f"{session_id}_{i}" for i in range(len(texts))]
        if metadatas is None:
            metadatas = [{"index": i} for i in range(len(texts))]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    def load_embeddings(
        self, session_id: str, model_name: str
    ) -> Optional[np.ndarray]:
        """Load stored embeddings for a session+model pair.

        Returns (N, D) array or None if not found.
        """
        col_name = self._collection_name(session_id, model_name)
        try:
            collection = self._client.get_collection(name=col_name)
        except Exception:
            return None

        result = collection.get(include=["embeddings"])
        if not result["embeddings"]:
            return None
        return np.array(result["embeddings"], dtype=np.float32)

    def list_sessions(self) -> List[dict]:
        """List all stored session/model combinations."""
        collections = self._client.list_collections()
        sessions = []
        for col in collections:
            meta = col.metadata or {}
            sessions.append({
                "collection_name": col.name,
                "session_id": meta.get("session_id", "unknown"),
                "model_name": meta.get("model_name", "unknown"),
                "count": col.count(),
            })
        return sessions

    def delete_session(self, session_id: str, model_name: str):
        """Delete a stored session+model collection."""
        col_name = self._collection_name(session_id, model_name)
        try:
            self._client.delete_collection(name=col_name)
        except Exception:
            pass

    def clear_all(self):
        """Delete all stored embedding collections."""
        for col in self._client.list_collections():
            try:
                self._client.delete_collection(name=col.name)
            except Exception:
                pass
