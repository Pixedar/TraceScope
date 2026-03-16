"""
TraceScope – Compare Embedding Models

Demonstrates analyzing the same conversation with two different
embedding models and comparing their vector spaces.
"""

from tracescope import TraceScopeConfig, AnalysisPipeline, auto_import
from tracescope.providers.embedding import OpenAIEmbedding


def main():
    config = TraceScopeConfig()
    session = auto_import("sample_data/sample_conversation.json")
    pipeline = AnalysisPipeline(config)

    # Analyze with model A
    print("Analyzing with text-embedding-3-large...")
    pipeline.embedding_provider = OpenAIEmbedding(
        api_key=config.openai_api_key,
        model="text-embedding-3-large",
    )
    result_a = pipeline.analyze(session, train_flow=False)
    print(f"  Clusters: {result_a.cluster_labels}")
    print(f"  Axes: {result_a.axis_info.labels}")

    # Analyze with model B
    print("\nAnalyzing with text-embedding-3-small...")
    pipeline.embedding_provider = OpenAIEmbedding(
        api_key=config.openai_api_key,
        model="text-embedding-3-small",
    )
    result_b = pipeline.analyze(session, train_flow=False)
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

    # Compare side by side
    from tracescope.visualization.scatter3d import plot_multi_paths
    fig = plot_multi_paths(
        [result_a, result_b],
        labels=["text-embedding-3-large", "text-embedding-3-small"],
    )
    fig.show()


if __name__ == "__main__":
    main()
