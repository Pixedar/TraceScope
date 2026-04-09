"""
TraceScope – Compare Embedding Models

Demonstrates analyzing the same data with two different
embedding models and comparing their vector spaces.

Before running:
  1. pip install tracescope    (or pip install -e . from repo root)
  2. Put your OpenAI API key in .env at project root:
     OPENAI_API_KEY=sk-...
"""

from pathlib import Path

from tracescope import TraceScopeConfig, AnalysisPipeline, from_list
from tracescope.providers.embedding import OpenAIEmbedding

_ROOT = Path(__file__).resolve().parent.parent


def main():
    config = TraceScopeConfig()

    session = from_list([
        "How do I read a file in Python?",
        "You can use open() with a context manager.",
        "What about writing to files?",
        "Use open() with 'w' mode.",
        "How do I handle binary files?",
        "Use 'rb' or 'wb' mode for binary.",
    ])

    pipeline = AnalysisPipeline(config)

    # Analyze with model A
    print("Analyzing with text-embedding-3-large...")
    pipeline.embedding_provider = OpenAIEmbedding(
        api_key=config.openai_api_key,
        model="text-embedding-3-large",
    )
    result_a = pipeline.analyze(session, train_flow=False, cache_path=str(_ROOT / "cache" / "cmp_large"))
    print(f"  Clusters: {result_a.cluster_labels}")
    print(f"  Axes: {result_a.axis_info.labels}")

    # Analyze with model B
    print("\nAnalyzing with text-embedding-3-small...")
    pipeline.embedding_provider = OpenAIEmbedding(
        api_key=config.openai_api_key,
        model="text-embedding-3-small",
    )
    result_b = pipeline.analyze(session, train_flow=False, cache_path=str(_ROOT / "cache" / "cmp_small"))
    print(f"  Clusters: {result_b.cluster_labels}")
    print(f"  Axes: {result_b.axis_info.labels}")

    # Compare
    print("\nComparing vector spaces...")
    comparison = pipeline.compare_models(
        session.session_id,
        "text-embedding-3-large",
        "text-embedding-3-small",
    )
    print(f"  Dimensions: {comparison['dim_a']} vs {comparison['dim_b']}")
    print(f"  Mean cosine similarity: {comparison['mean_cosine_similarity']:.4f}")
    print(f"  Min cosine similarity: {comparison['min_cosine_similarity']:.4f}")
    print(f"  Max cosine similarity: {comparison['max_cosine_similarity']:.4f}")


if __name__ == "__main__":
    main()
