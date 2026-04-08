"""
Example: PRM800K Math Reasoning Chains
=======================================
Visualize math reasoning chains from the PRM800K dataset.

Uses the included sample_data/prm_demo_paths.json and loads key score
channels: prm_rating (per-step), solved (per-path outcome),
mean_prm_rating (path-level quality), and found_error (per-path).

Dataset: MIT license
Source:  https://github.com/openai/prm800k

Before running:
  1. pip install tracescope    (or pip install -e . from repo root)
  2. Put your OpenAI API key in .env at project root:
     OPENAI_API_KEY=sk-...
"""

import json
from pathlib import Path

from tracescope import (
    TraceScopeConfig, AnalysisPipeline, from_lists, launch_renderer,
)

_ROOT = Path(__file__).resolve().parent.parent

print("=== PRM800K Math Reasoning Chains ===\n")

config = TraceScopeConfig(
    embedding_model="text-embedding-3-large",
)

payload_path = _ROOT / "sample_data" / "prm_demo_paths.json"
with open(payload_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

raw_path_scores = payload.get("path_scores") or {}
if isinstance(raw_path_scores, list):
    path_scores = {i: score_map for i, score_map in enumerate(raw_path_scores)}
else:
    path_scores = {int(k): v for k, v in raw_path_scores.items()}

# Keep the most important score channels
_KEEP_CHANNELS = {"prm_rating", "solved", "mean_prm_rating", "found_error"}

# Filter entry_scores to keep only relevant channels
filtered_entry_scores = []
for per_path in payload.get("entry_scores", []):
    filtered_path = []
    for step_scores in per_path:
        filtered_path.append({
            k: v for k, v in step_scores.items() if k in _KEEP_CHANNELS
        })
    filtered_entry_scores.append(filtered_path)

# Filter path_scores to keep only relevant channels
filtered_path_scores = {}
for pid, score_map in path_scores.items():
    filtered_path_scores[pid] = {
        k: v for k, v in score_map.items() if k in _KEEP_CHANNELS
    }

session = from_lists(
    payload["paths"],
    labels=payload.get("labels"),
    entry_scores=filtered_entry_scores,
    path_scores=filtered_path_scores,
)

n_paths = len(payload["paths"])
score_channels = sorted({
    key
    for per_path in filtered_entry_scores
    for step_scores in per_path
    for key in step_scores.keys()
} | {
    key
    for score_map in filtered_path_scores.values()
    for key in score_map.keys()
})

print(f"Imported {len(session)} entries across {n_paths} paths from '{payload_path}'")
if score_channels:
    print(f"  Score channels: {score_channels}")

pipeline = AnalysisPipeline(config)
result = pipeline.analyze(
    session,
    progress_callback=lambda stage, pct: print(f"  [{stage}] {pct*100:.0f}%"),
    train_flow=True,
    cache_path="cache/prm_demo",
    param_range=[186, 194, 202],
)

print(f"\nAnalysis complete:")
print(f"  Embedding model: {result.embedding_model}")
print(f"  Clusters found: {result.clusters.n_clusters}")
print(f"  Cluster labels: {result.cluster_labels}")
print(f"  Axis labels: {result.axis_info.labels}")
print(f"  Flow model trained: {result.flow_model_trained}")

launch_renderer(
    result,
    explainer=pipeline.explainer,
    dataset_name="PRM800K Math Reasoning",
    initial_state={
        "speed": 2.5,
        "show_attractors": True,
        "flow_color": "score:found_error",
        "sensitivity": 0.01,
    },
)
