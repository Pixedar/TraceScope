"""
TraceScope - Semantic Space Analysis Tool

Embed, cluster, and visualize any collection of texts in 3D semantic space.
Works with chatbot conversations, news articles, research papers, or any text data.

Quick start:
    from tracescope import TraceScopeConfig, AnalysisPipeline, auto_import, from_list, from_lists

    config = TraceScopeConfig(openai_api_key="sk-...")
    session = auto_import("conversation.json")   # or from_list(["text1", "text2", ...])
    # Multi-path: session = from_lists([["path1a", "path1b"], ["path2a", "path2b"]])
    pipeline = AnalysisPipeline(config)
    result = pipeline.analyze(session)

    # Visualization
    from tracescope import launch_renderer
    launch_renderer(result)

    # Programmatic API
    from tracescope import TraceQuery
    query = TraceQuery(result, pipeline.embedding_provider)
    query.get_lookup()
    query.explain_path(["hello", "goodbye"])
    query.query_flow_at("some text")
"""

from tracescope.config import TraceScopeConfig
from tracescope.providers.embedding import EmbeddingProvider, OpenAIEmbedding
from tracescope.providers.llm import LLMProvider, OpenAILLM
from tracescope.models.trace import TraceEntry, TraceSession
from tracescope.models.analysis import AnalysisResult, ClusterResult, AxisInfo
from tracescope.analysis.importer import auto_import, from_list, from_lists
from tracescope.analysis.pipeline import AnalysisPipeline
from tracescope.query import TraceQuery


def launch_dashboard(*args, **kwargs):
    """Deprecated — use launch_renderer() instead.

    The GPU renderer now includes all dashboard functionality (LLM explain,
    probe sliders, cluster legend, etc.) and falls back to software rendering
    when no GPU is available.
    """
    import warnings
    warnings.warn(
        "launch_dashboard() is deprecated. Use launch_renderer() instead — "
        "it now includes all dashboard features and works without a GPU.",
        DeprecationWarning, stacklevel=2,
    )
    from tracescope.visualization.dashboard import launch_dashboard as _launch
    return _launch(*args, **kwargs)


def launch_renderer(*args, **kwargs):
    """Lazy-loaded GPU renderer launcher (imports vispy only when called)."""
    from tracescope.visualization.gl_renderer import launch_renderer as _launch
    return _launch(*args, **kwargs)


__version__ = "0.1.0"

__all__ = [
    "TraceScopeConfig",
    "EmbeddingProvider", "OpenAIEmbedding",
    "LLMProvider", "OpenAILLM",
    "TraceEntry", "TraceSession",
    "AnalysisResult", "ClusterResult", "AxisInfo",
    "auto_import",
    "from_list",
    "from_lists",
    "AnalysisPipeline",
    "TraceQuery",
    "launch_dashboard",
    "launch_renderer",
]
