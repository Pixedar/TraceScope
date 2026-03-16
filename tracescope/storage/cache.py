"""
LLM response cache backed by SQLite.

Avoids redundant API calls by caching responses keyed on
SHA-256(model_name || system_prompt || user_prompt).
"""

from __future__ import annotations

import sqlite3
import time
from typing import Optional

from tracescope.utils.hashing import sha256_key


class LLMResponseCache:
    """SQLite-backed LLM response cache.

    Args:
        db_path: Path to the SQLite database file.
        enabled: If False, all lookups miss and stores are skipped.
    """

    def __init__(self, db_path: str, enabled: bool = True):
        self.db_path = db_path
        self.enabled = enabled
        if enabled:
            self._conn = sqlite3.connect(db_path)
            self._init_table()
        else:
            self._conn = None

    def _init_table(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                hash        TEXT PRIMARY KEY,
                model_name  TEXT NOT NULL,
                response    TEXT NOT NULL,
                created_at  REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, model_name: str, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Look up a cached response. Returns None on miss."""
        if not self.enabled:
            return None
        key = sha256_key(model_name, system_prompt, user_prompt)
        row = self._conn.execute(
            "SELECT response FROM llm_cache WHERE hash = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(self, model_name: str, system_prompt: str, user_prompt: str, response: str):
        """Store a response in the cache."""
        if not self.enabled:
            return
        key = sha256_key(model_name, system_prompt, user_prompt)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache (hash, model_name, response, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, model_name, response, time.time()),
        )
        self._conn.commit()

    def clear(self):
        """Clear all cached responses."""
        if self._conn:
            self._conn.execute("DELETE FROM llm_cache")
            self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()


class ResultCache:
    """SQLite-backed cache for expensive ML computation results.

    Caches clustering, dimension reduction, and velocity grid results
    keyed on embeddings fingerprint (matching Android's DimensionReducerAdapter).

    Uses the same SQLite database as LLMResponseCache but a separate table.

    Args:
        db_path: Path to the SQLite database file.
        enabled: If False, all lookups miss and stores are skipped.
    """

    def __init__(self, db_path: str, enabled: bool = True):
        self.db_path = db_path
        self.enabled = enabled
        if enabled:
            self._conn = sqlite3.connect(db_path)
            self._init_table()
        else:
            self._conn = None

    def _init_table(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS result_cache (
                key         TEXT PRIMARY KEY,
                step        TEXT NOT NULL,
                data        BLOB NOT NULL,
                created_at  REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get_result(self, step: str, fingerprint: str) -> Optional[bytes]:
        """Look up a cached computation result.

        Args:
            step: Pipeline step name (e.g. "clustering", "dim_reduction", "velocity_grid").
            fingerprint: Embeddings fingerprint (SHA-256 hex digest).

        Returns:
            Cached data as bytes, or None on miss.
        """
        if not self.enabled:
            return None
        key = sha256_key(step, fingerprint)
        row = self._conn.execute(
            "SELECT data FROM result_cache WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put_result(self, step: str, fingerprint: str, data: bytes):
        """Store a computation result in the cache.

        Args:
            step: Pipeline step name.
            fingerprint: Embeddings fingerprint.
            data: Result data as bytes (JSON-encoded or raw numpy).
        """
        if not self.enabled:
            return
        key = sha256_key(step, fingerprint)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO result_cache (key, step, data, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, step, data, time.time()),
        )
        self._conn.commit()

    def clear(self):
        """Clear all cached results."""
        if self._conn:
            self._conn.execute("DELETE FROM result_cache")
            self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
