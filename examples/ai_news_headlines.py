"""
Example: AI & Tech News Headlines
==================================
Analyze ~300 diverse headlines spanning AI, chips, climate, energy, biotech,
space, markets, geopolitics, labor, and media — interleaved to produce a
rich flow field with multiple attractor basins and branching rivers.

Seed experimentation
--------------------
Every randomized stage of the pipeline (MDN flow training, KMeans
clustering, UMAP/t-SNE dim-reduction) is controlled by a single global
integer seed.  If the default seed produces an ugly flow for your data,
pass a different one with ``--seed 7`` (or 1, 17, 123, 999, …) to try a
different flow topology / cluster layout / axis alignment WITHOUT
losing reproducibility.  The seed is baked into the cache fingerprint
so each seed gets its own cached result.

To let every stage pick its own entropy (fully non-deterministic),
pass ``--nondet`` instead.

Before running:
  1. pip install tracescope    (or pip install -e . from repo root)
  2. Put your OpenAI API key in .env at project root:
     OPENAI_API_KEY=sk-...
"""

import argparse
import json
from pathlib import Path

from tracescope import (
    TraceScopeConfig, AnalysisPipeline, from_list, launch_renderer,
)

parser = argparse.ArgumentParser(description="AI & Tech News Headlines example")
parser.add_argument("--rbf", action="store_true", help="Use RBF flow model instead of MDN")
parser.add_argument(
    "--seed", type=int, default=42,
    help="Global seed for MDN, clustering, and UMAP (try 1/7/17/123/… to "
         "explore alternative flow topologies). Default: 42.",
)
parser.add_argument(
    "--nondet", action="store_true",
    help="Disable determinism — every stage picks its own random seed "
         "(different flow shape each run).",
)
args = parser.parse_args()

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_SAMPLE_DIR = _SCRIPT_DIR / "sample_data"

print("=== AI & Tech News Headlines ===\n")

config = TraceScopeConfig(
    embedding_model="text-embedding-3-large",
    seed=args.seed,
    deterministic=not args.nondet,
)

with open(_SAMPLE_DIR / "news_headlines.json", "r", encoding="utf-8") as f:
    headlines = json.load(f)

session = from_list(headlines)

print(f"Imported {len(session)} entries")

_flow_mode = "rbf" if args.rbf else "mdn"
_cache_suffix = "_rbf" if args.rbf else ""
# Bake the seed into the cache filename so alternate seeds don't stomp
# on each other's caches on disk.
_seed_suffix = "" if (args.seed == 42 and not args.nondet) else (
    "_nondet" if args.nondet else f"_seed{args.seed}"
)

pipeline = AnalysisPipeline(config)
result = pipeline.analyze(
    session,
    progress_callback=lambda stage, pct: print(f"  [{stage}] {pct*100:.0f}%"),
    train_flow=True,
    flow_mode=_flow_mode,
    cache_path=str(_ROOT / "cache" / f"single_path{_cache_suffix}{_seed_suffix}"),
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
        "flow_color": "cluster",
    },
)
