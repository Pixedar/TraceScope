"""
Example: AI & Tech News Headlines
==================================
Analyze ~300 diverse headlines spanning AI, chips, climate, energy, biotech,
space, markets, geopolitics, labor, and media — interleaved to produce a
rich flow field with multiple attractor basins and branching rivers.

Before running:
  1. pip install tracescope    (or pip install -e . from repo root)
  2. Put your OpenAI API key in .env at project root:
     OPENAI_API_KEY=sk-...
"""

import json
from pathlib import Path

from tracescope import (
    TraceScopeConfig, AnalysisPipeline, from_list, launch_renderer,
)

_SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"

print("=== AI & Tech News Headlines ===\n")

config = TraceScopeConfig(
    embedding_model="text-embedding-3-large",
)

with open(_SAMPLE_DIR / "news_headlines.json", "r", encoding="utf-8") as f:
    headlines = json.load(f)

session = from_list(headlines)

print(f"Imported {len(session)} entries")

pipeline = AnalysisPipeline(config)
result = pipeline.analyze(
    session,
    progress_callback=lambda stage, pct: print(f"  [{stage}] {pct*100:.0f}%"),
    train_flow=True,
    cache_path="cache/single_path",
)

print(f"\nAnalysis complete:")
print(f"  Clusters: {result.cluster_labels}")
print(f"  Axes: {result.axis_info.labels}")
print(f"  Flow model trained: {result.flow_model_trained}")

launch_renderer(
    result,
    explainer=pipeline.explainer,
    dataset_name="AI & Tech News Headlines",
    initial_state={
        "speed": 1.1,
        "show_attractors": True,
        "flow_color": "cluster",
    },
)
