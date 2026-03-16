"""
TraceScope – Basic Usage Example

Before running:
  1. pip install -e .
  2. Put your OpenAI API key in .env at project root:
     OPENAI_API_KEY=sk-...

This example:
  1. Imports a sample conversation
  2. Runs the full analysis pipeline
  3. Launches the interactive 3D renderer
"""

from tracescope import (
    TraceScopeConfig, AnalysisPipeline, auto_import, from_list, from_lists,
    launch_renderer,
)


def single_path_example():
    """Analyze a single list of texts."""
    print("=== Single Path Example ===\n")

    config = TraceScopeConfig(
        embedding_model="text-embedding-3-large",
        llm_model="gpt-5-mini",
    )

    # Any list of texts works: news headlines, research abstracts, chat messages...
    session = from_list([
        "Fed holds rates steady amid inflation concerns",
        "Tech earnings surge on AI demand",
        "Climate summit reaches carbon emissions deal",
        "Housing market cools as mortgage rates rise",
        "Quantum computing startup hits milestone",
    ], label="Tech & Finance News")

    print(f"Imported {len(session)} entries")

    pipeline = AnalysisPipeline(config)
    result = pipeline.analyze(
        session,
        progress_callback=lambda stage, pct: print(f"  [{stage}] {pct*100:.0f}%"),
        train_flow=True,
        cache_path="cache/single_path",  # Second run loads instantly
    )

    print(f"\nAnalysis complete:")
    print(f"  Clusters: {result.cluster_labels}")
    print(f"  Axes: {result.axis_info.labels}")
    print(f"  Flow model trained: {result.flow_model_trained}")

    # Launch 3D renderer with LLM explain button
    launch_renderer(result, explainer=pipeline.explainer)


def multi_path_example():
    """Analyze multiple independent paths together."""
    print("=== Multi-Path Example ===\n")

    config = TraceScopeConfig(
        embedding_model="text-embedding-3-large",
        llm_model="gpt-5-mini",
    )

    # Multiple independent semantic paths → unified space with correct flow field
    session = from_lists([
        ["Fed holds rates steady", "Tech earnings surge on AI", "Housing market cools"],
        ["Climate summit reaches deal", "Quantum computing milestone", "Mars rover update"],
        ["New vaccine approved", "Hospital staffing crisis", "Mental health funding"],
    ], labels=["Finance", "Science", "Health"])

    print(f"Imported {len(session)} entries across 3 paths")

    pipeline = AnalysisPipeline(config)
    result = pipeline.analyze(session, train_flow=True, cache_path="cache/multi_path")

    print(f"  Clusters: {result.cluster_labels}")
    print(f"  Axes: {result.axis_info.labels}")

    launch_renderer(result, explainer=pipeline.explainer)


def conversation_example():
    """Analyze a chatbot conversation from file."""
    print("=== Conversation Example ===\n")

    config = TraceScopeConfig(
        embedding_model="text-embedding-3-large",
        llm_model="gpt-5-mini",
    )

    session = auto_import("sample_data/sample_conversation.json")
    print(f"Imported {len(session)} messages from '{session.label}'")

    pipeline = AnalysisPipeline(config)
    result = pipeline.analyze(
        session,
        progress_callback=lambda stage, pct: print(f"  [{stage}] {pct*100:.0f}%"),
        train_flow=True,
        cache_path="cache/conversation",  # Second run loads instantly
    )

    print(f"\nAnalysis complete:")
    print(f"  Embedding model: {result.embedding_model}")
    print(f"  Clusters found: {result.clusters.n_clusters}")
    print(f"  Cluster labels: {result.cluster_labels}")
    print(f"  Axis labels: {result.axis_info.labels}")
    print(f"  Flow model trained: {result.flow_model_trained}")

    # Launch renderer with LLM explanations enabled
    # GUI controls: sliders, mark/clear, Explain button, cluster legend
    # Double-click a point to jump sliders there
    launch_renderer(result, explainer=pipeline.explainer)


if __name__ == "__main__":
    # Pick one:
    conversation_example()
    # single_path_example()
    # multi_path_example()
