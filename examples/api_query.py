"""
TraceScope – Programmatic API Example

Demonstrates the TraceQuery API for querying the semantic space
without the visualization GUI. Shows both single-path and multi-path usage.

Before running:
  1. pip install tracescope    (or pip install -e . from repo root)
  2. Put your OpenAI API key in .env at project root:
     OPENAI_API_KEY=sk-...
"""

from pathlib import Path

from tracescope import (
    TraceScopeConfig, AnalysisPipeline, from_list, from_lists, TraceQuery,
)

_ROOT = Path(__file__).resolve().parent.parent


def single_path_api():
    """Query API with a single-path session."""
    print("=== Single-Path API Example ===\n")

    config = TraceScopeConfig(
        embedding_model="text-embedding-3-large",
    )

    session = from_list([
        "How do I read a file in Python?",
        "You can use open() with a context manager.",
        "What about writing to files?",
        "Use open() with 'w' mode: with open('file.txt', 'w') as f: f.write(data)",
        "How do I handle binary files?",
        "Use 'rb' or 'wb' mode for binary: open('image.png', 'rb')",
        "Can I read files asynchronously?",
        "Yes, use aiofiles: async with aiofiles.open('file.txt') as f: data = await f.read()",
        "What about reading from URLs?",
        "Use the requests library: response = requests.get(url); data = response.text",
    ])

    print(f"Imported {len(session)} entries")

    pipeline = AnalysisPipeline(config)
    result = pipeline.analyze(
        session,
        progress_callback=lambda stage, pct: print(f"  [{stage}] {pct*100:.0f}%"),
        train_flow=True,
        cache_path=str(_ROOT / "cache" / "api_single"),
    )

    query = TraceQuery(result, pipeline.embedding_provider, pipeline.explainer)
    _run_queries(query)


def multi_path_api():
    """Query API with a multi-path session — multiple independent sequences
    analyzed together with shared embeddings, clusters, and a unified flow field."""
    print("=== Multi-Path API Example ===\n")

    config = TraceScopeConfig(
        embedding_model="text-embedding-3-large",
    )

    # Three independent paths → unified semantic space
    # labels is optional — names each path for display
    session = from_lists([
        [
            "How do I read a file in Python?",
            "Use open() with a context manager",
            "What about binary files?",
            "Use 'rb' mode for binary data",
        ],
        [
            "How do I make HTTP requests?",
            "Use the requests library",
            "How about async HTTP?",
            "Use aiohttp or httpx for async",
        ],
        [
            "How do I connect to a database?",
            "Use SQLAlchemy or psycopg2",
            "How do I write queries?",
            "Use parameterized queries to prevent SQL injection",
        ],
    ], labels=["File I/O", "HTTP", "Databases"])

    print(f"Imported {len(session)} entries across 3 paths")

    pipeline = AnalysisPipeline(config)
    result = pipeline.analyze(session, train_flow=True, cache_path=str(_ROOT / "cache" / "api_multi"))

    query = TraceQuery(result, pipeline.embedding_provider, pipeline.explainer)
    _run_queries(query)


def _run_queries(query: "TraceQuery"):
    """Run all four query methods on a TraceQuery instance."""

    # ── Get the lookup table (cached metadata about the space) ────────
    lookup = query.get_lookup()
    print("\n=== Lookup Table ===")
    print(f"  Axis labels: {lookup['axis_labels']}")
    print(f"  Clusters: {len(lookup['clusters'])}")
    for c in lookup["clusters"]:
        print(f"    [{c['id']}] {c['label']} ({c['size']} entries)")
    print(f"  Has flow field: {lookup['has_flow']}")

    # ── Method 1: Explain a path of new texts ─────────────────────────
    path_result = query.explain_path([
        "How do I open a file?",
        "How do I make HTTP requests?",
        "How do I build a REST API?",
    ])
    print("\n=== Path Explanation ===")
    for i, pt in enumerate(path_result["points"]):
        print(f"  Point {i}: {pt['text'][:50]}")
        print(f"    Axis %: {pt['axis_percentages']}")
    if path_result.get("explanation"):
        print(f"  Explanation: {path_result['explanation']}")

    # ── Method 2: Query flow field at a point ─────────────────────────
    if lookup["has_flow"]:
        flow_result = query.query_flow_at("How do I deploy a web application?")
        print("\n=== Flow at Point ===")
        print(f"  Speed: {flow_result['speed']}")
        for ax in flow_result["axis_decomposition"]:
            print(f"  {ax['direction']}{ax['magnitude']:.4f} along '{ax['axis_label']}'")
        for cp in flow_result["cluster_pull"][:3]:
            print(f"  {cp['interpretation']} '{cp['cluster_label']}' (alignment: {cp['alignment']:.3f})")

    # ── Method 3: Estimate direction from a sequence ──────────────────
    dir_result = query.query_direction_at([
        "What is a variable?",
        "How do functions work?",
        "Explain object-oriented programming",
    ])
    print("\n=== Estimated Direction ===")
    print(f"  Magnitude: {dir_result['estimated_magnitude']:.4f}")
    for ax in dir_result["axis_decomposition"]:
        print(f"  {ax['direction']}{ax['magnitude']:.4f} along '{ax['axis_label']}'")

    # ── Method 4: Compare two paths ───────────────────────────────────
    sim = query.path_similarity(
        ["How do I read files?", "How do I write files?", "How do I delete files?"],
        ["How do I open a database?", "How do I query tables?", "How do I close connections?"],
    )
    print("\n=== Path Similarity ===")
    print(f"  Overall score: {sim['overall_score']:.4f}")
    print(f"  Direction similarity: {sim['direction_similarity']:.4f}")
    print(f"  Mean cosine similarity: {sim['mean_cosine_similarity']:.4f}")


if __name__ == "__main__":
    # Pick one:
    single_path_api()
    # multi_path_api()
