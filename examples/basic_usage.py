"""
TraceScope – Basic Usage Example

Before running:
  1. pip install tracescope    (or pip install -e . from repo root)
  2. Put your OpenAI API key in .env at project root:
     OPENAI_API_KEY=sk-...

This example:
  1. Imports sample data
  2. Runs the full analysis pipeline (with caching)
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
    )

    # Any list of texts works: news headlines, research abstracts, chat messages...
    # Tip: 20+ entries gives much richer flow fields
    session = from_list([
        "Fed holds rates steady amid inflation concerns",
        "Tech earnings surge on AI demand",
        "Climate summit reaches carbon emissions deal",
        "Housing market cools as mortgage rates rise",
        "Quantum computing startup hits milestone",
        # ... add more entries for better flow field quality
    ])

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
    )

    # Multiple independent semantic paths → unified space with correct flow field
    # labels is optional — names each path for display
    session = from_lists([
        ["Fed holds rates steady", "Tech earnings surge on AI", "Housing market cools"],
        ["Climate summit reaches deal", "Quantum computing milestone", "Mars rover update"],
        ["New vaccine approved", "Hospital staffing crisis", "Mental health funding"],
        # ... more paths improve flow field quality
    ], labels=["Finance", "Science", "Health"])

    print(f"Imported {len(session)} entries across 3 paths")

    pipeline = AnalysisPipeline(config)
    result = pipeline.analyze(session, train_flow=True, cache_path="cache/multi_path")

    print(f"  Clusters: {result.cluster_labels}")
    print(f"  Axes: {result.axis_info.labels}")

    launch_renderer(result, explainer=pipeline.explainer)


def prm_demo_example():
    """Visualize math reasoning chains from the PRM800K dataset.

    Uses the included sample_data/prm_demo_40paths.json — 55 multi-step
    math reasoning chains showing how problem-solving flows through
    semantic space.
    """
    print("=== PRM800K Demo Example ===\n")

    config = TraceScopeConfig(
        embedding_model="text-embedding-3-large",
    )

    session = auto_import("sample_data/prm_demo_40paths.json")
    print(f"Imported {len(session)} entries from '{session.label}'")

    pipeline = AnalysisPipeline(config)
    result = pipeline.analyze(
        session,
        progress_callback=lambda stage, pct: print(f"  [{stage}] {pct*100:.0f}%"),
        train_flow=True,
        cache_path="cache/prm_demo",
    )

    print(f"\nAnalysis complete:")
    print(f"  Embedding model: {result.embedding_model}")
    print(f"  Clusters found: {result.clusters.n_clusters}")
    print(f"  Cluster labels: {result.cluster_labels}")
    print(f"  Axis labels: {result.axis_info.labels}")
    print(f"  Flow model trained: {result.flow_model_trained}")

    launch_renderer(result, explainer=pipeline.explainer)


if __name__ == "__main__":
    # Pick one:
    prm_demo_example()
    # single_path_example()
    # multi_path_example()
