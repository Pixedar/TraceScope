"""
TraceScopeConfig - Central configuration for the library.

Holds API keys, model selections, and storage paths.
Use the factory methods to create providers.

API keys are loaded in this priority order:
  1. Passed directly to TraceScopeConfig(openai_api_key="...")
  2. Loaded from .env file (searched: cwd, project root, ~/.tracescope/.env)
  3. Read from environment variable OPENAI_API_KEY
"""

from __future__ import annotations
import os
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _fix_ssl_certificates():
    """Fix broken SSL_CERT_FILE on Windows (common with Git for Windows).

    If SSL_CERT_FILE points to a nonexistent file, replace it with
    the certifi bundle so httpx/OpenAI SDK can create SSL contexts.
    """
    cert_file = os.environ.get("SSL_CERT_FILE")
    if cert_file and not os.path.isfile(cert_file):
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            # No certifi → just unset the broken path so Python uses defaults
            del os.environ["SSL_CERT_FILE"]


def _load_dotenv():
    """Load .env file from common locations."""
    try:
        from dotenv import load_dotenv

        # Search: cwd → project root → home/.tracescope/
        for candidate in [
            Path.cwd() / ".env",
            Path.cwd().parent / ".env",
            Path.home() / ".tracescope" / ".env",
        ]:
            if candidate.is_file():
                load_dotenv(candidate)
                return
    except ImportError:
        pass  # python-dotenv not installed, skip


# Fix SSL before any OpenAI client is created
_fix_ssl_certificates()


@dataclass
class TraceScopeConfig:
    """Configuration object for TraceScope.

    Args:
        openai_api_key: OpenAI API key (for embeddings and/or LLM interpretation).
                        Falls back to .env file, then OPENAI_API_KEY env var.
        anthropic_api_key: Anthropic API key (optional, for Claude-based interpretation).
                           Falls back to .env file, then ANTHROPIC_API_KEY env var.
        embedding_model: Name of the embedding model to use.
                         Default: "text-embedding-3-large" (OpenAI).
        embedding_provider_type: Provider type: "openai", "cohere", "huggingface".
        llm_model: Name of the LLM model for simpler tasks (axis/cluster labeling).
                   Default: "gpt-5-mini" (OpenAI).
        llm_model_complex: Name of the LLM model for complex tasks (explanations).
                   Default: "gpt-5" (OpenAI).
        llm_provider_type: Provider type: "openai", "anthropic".
        storage_dir: Directory for persistent storage (ChromaDB, cache).
                     Default: ~/.tracescope
        cache_enabled: Whether to cache LLM responses. Default: True.
        flow_mode: Flow model type: "mdn" (neural, default), "rbf" (kernel),
                   or "gpvf" (Gaussian process).
        mdn_hidden: MDN hidden layer size (50-300, default 100).
        mdn_iters: MDN training iterations (2000-20000, default 8000).
        velocity_grid_size: 3D velocity grid resolution (20-60, default 40).
        rbf_kernel: RBF kernel type (default "thin_plate_spline").
        rbf_smoothing: RBF regularization (default 0.1).
        deterministic: Whether to seed all RNGs for reproducible results.
                       Default: True. Set to False to allow non-deterministic
                       training (may explore different flow topologies).
        seed: Global integer seed used by every randomized stage of the
              pipeline when `deterministic=True` — MDN flow training
              (torch + numpy), KMeans clustering, and UMAP/t-SNE dimension
              reduction. Default: 42. Change this to explore alternative
              flow topologies / cluster layouts / axis alignments without
              disabling determinism entirely. Ignored when
              `deterministic=False` (in that case every stage picks its
              own random seed). The seed is part of the full-result cache
              fingerprint, so changing it triggers a fresh pipeline run.
    """

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-large"
    embedding_provider_type: str = "openai"
    llm_model: str = "gpt-5-mini"
    llm_model_complex: str = "gpt-5"
    llm_provider_type: str = "openai"
    storage_dir: str = field(default_factory=lambda: str(Path.home() / ".tracescope"))
    cache_enabled: bool = True
    flow_mode: str = "mdn"
    mdn_hidden: int = 100
    mdn_iters: int = 8000
    velocity_grid_size: int = 40
    rbf_kernel: str = "thin_plate_spline"
    rbf_smoothing: float = 0.1
    deterministic: bool = True
    seed: int = 42

    def __post_init__(self):
        # Load .env before checking env vars
        _load_dotenv()

        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

        os.makedirs(self.storage_dir, exist_ok=True)

    def create_embedding_provider(self):
        """Factory: create an EmbeddingProvider based on config."""
        from tracescope.providers.embedding import OpenAIEmbedding

        if self.embedding_provider_type == "openai":
            if not self.openai_api_key:
                raise ValueError("openai_api_key required for OpenAI embeddings")
            return OpenAIEmbedding(api_key=self.openai_api_key, model=self.embedding_model)
        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider_type}")

    def create_llm_provider(self, model_override: Optional[str] = None):
        """Factory: create an LLMProvider based on config.

        Args:
            model_override: If given, use this model instead of llm_model.
        """
        from tracescope.providers.llm import OpenAILLM, AnthropicLLM

        model = model_override or self.llm_model
        if self.llm_provider_type == "openai":
            if not self.openai_api_key:
                raise ValueError("openai_api_key required for OpenAI LLM")
            return OpenAILLM(api_key=self.openai_api_key, model=model)
        elif self.llm_provider_type == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("anthropic_api_key required for Anthropic LLM")
            return AnthropicLLM(api_key=self.anthropic_api_key, model=model)
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider_type}")

    def create_llm_provider_complex(self):
        """Factory: create an LLMProvider using the complex model (for explanations)."""
        return self.create_llm_provider(model_override=self.llm_model_complex)

    @property
    def chromadb_dir(self) -> str:
        return os.path.join(self.storage_dir, "chromadb")

    @property
    def cache_db_path(self) -> str:
        return os.path.join(self.storage_dir, "cache.db")
