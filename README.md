# TraceScope

<p align="center">
  <img src="docs/assets/demo_v_3.gif" alt="TraceScope demo animation" width="950">
</p>

**TraceScope maps the flow of meaning**  
Embed, cluster, and visualize any collection of texts in 3D semantic space — then learn a continuous semantic flow field over that space, so you can see not just where texts are, but how meaning tends to move between them.

TraceScope builds a rich semantic map from your data — with labeled axes, named clusters, trajectories, and a trained flow model that reveals how themes, intent, style, or reasoning evolve across time.

Works with **anything**: chatbot conversations, agent traces, news headlines, research papers, product reviews, diary entries, support logs, or any ordered collection of text.

Use it in two ways:
- **Interactive GUI** for visual exploration, interpretability, and presentation
- **Lightweight API** for integration into LLM agents, observability pipelines, research tools, and semantic monitoring systems

## Why TraceScope

Most embedding tools show a static cloud of points. TraceScope goes further:

- **Semantic structure** — discover clusters, labeled axes, and nearest neighbors
- **Semantic dynamics** — model trajectories and learn a continuous flow field over sparse text sequences
- **Interpretability** — inspect how a conversation, system, or dataset drifts, stabilizes, loops, or transitions
- **Integration** — use the same semantic space programmatically through a lightweight query API
## Installation

```bash
# Full install — GPU renderer, MDN flow models, all LLM providers
pip install tracescope
```

**Lighter variants** (use `--no-deps` to skip the full dependency tree):

```bash
# CPU-only — renderer + all features, no PyTorch (RBF flow still works)
pip install --no-deps tracescope && pip install -r https://raw.githubusercontent.com/Pixedar/TraceScope/master/requirements-cpu.txt

# API-only — analysis pipeline, no GUI, no PyTorch
pip install --no-deps tracescope && pip install -r https://raw.githubusercontent.com/Pixedar/TraceScope/master/requirements-api.txt
```

You'll need an OpenAI API key for embeddings and LLM explanations. Set it in a `.env` file or pass it directly:

```
OPENAI_API_KEY=sk-...
```

## Quick Start

### Analyze a chatbot conversation
Useful for real-world agent debugging: reveal hidden conversational attractors, looping failure modes, unstable transitions, and recovery trajectories in multi-turn chats
```python
from tracescope import TraceScopeConfig, AnalysisPipeline, auto_import

config = TraceScopeConfig(embedding_model="text-embedding-3-large")
session = auto_import("conversation.json")
pipeline = AnalysisPipeline(config)
result = pipeline.analyze(session, train_flow=True)

print(f"Axes: {result.axis_info.labels}")
print(f"Clusters: {result.cluster_labels}")

# Save result for instant re-use (skips entire pipeline on next run)
result.save_result("my_analysis")

# Next time: loads instantly if data hasn't changed
result = pipeline.analyze(session, cache_path="my_analysis")
```

### Analyze any list of texts
Turn any ordered text collection into a semantic trajectory to reveal recurring human states, emotional patterns, behavioral loops, and emerging trends over time
```python
from tracescope import TraceScopeConfig, AnalysisPipeline, from_list

config = TraceScopeConfig()

# News headlines
session = from_list([
    "Fed holds rates steady amid inflation concerns",
    "Tech earnings surge on AI demand",
    "Climate summit reaches carbon emissions deal",
    "Housing market cools as mortgage rates rise",
    "Quantum computing startup hits milestone",
], label="Tech & Finance News")

# Research abstracts, product reviews, log entries — anything works
pipeline = AnalysisPipeline(config)
result = pipeline.analyze(session, train_flow=True)
```

### Visualize

```python
from tracescope import launch_renderer

# Interactive 3D renderer with flow field animation
# Controls: Space=flow, B=ball, P=points, A=auto-rotate, +/-=size
launch_renderer(result)

# With LLM explanations (Explain button in the GUI)
launch_renderer(result, explainer=pipeline.explainer)
```

## Input Formats

TraceScope accepts data in multiple formats:

### From code — single path (list of strings)

```python
from tracescope import from_list

session = from_list(["text one", "text two", "text three"], label="My texts")
```

### From code — multiple independent paths

Analyze several independent sequences together with shared embeddings, clusters, and axes, but a unified MDN flow field that correctly learns from each path independently (no spurious boundary velocities):

```python
from tracescope import TraceScopeConfig, AnalysisPipeline, from_lists

config = TraceScopeConfig()
pipeline = AnalysisPipeline(config)

session = from_lists([
    ["Fed holds rates steady", "Tech earnings surge on AI", "Housing market cools"],
    ["Climate summit reaches deal", "Quantum computing milestone", "Mars rover update"],
    ["New vaccine approved", "Hospital staffing crisis", "Mental health funding"],
], labels=["Finance", "Science", "Health"])

result = pipeline.analyze(session, train_flow=True)
```

### From file — auto-detected format

```python
from tracescope import auto_import

session = auto_import("data.json")
```

Supported JSON formats:

**Plain string array** — simplest, works for any text collection:
```json
["First text", "Second text", "Third text"]
```

**Multi-path** — multiple independent sequences analyzed together:
```json
{
  "paths": [
    ["Path 1 text A", "Path 1 text B", "Path 1 text C"],
    ["Path 2 text A", "Path 2 text B"]
  ],
  "labels": ["First path", "Second path"]
}
```

**OpenAI chat format**:
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

**Anthropic format**:
```json
{
  "model": "claude-sonnet-4-20250514",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]}
  ]
}
```

**Plain text** (`.txt` files, split on blank lines):
```
First message

Second message

Third message
```

## Programmatic API — TraceQuery

After running the pipeline once, use `TraceQuery` for fast programmatic access to the semantic space. No re-computation needed — everything is served from the pre-computed lookup table and velocity grid.

```python
from tracescope import TraceQuery

query = TraceQuery(result, pipeline.embedding_provider, pipeline.explainer)
```

### `get_lookup()` — Space metadata

Returns a dict with all computed information about the semantic space:

```python
lookup = query.get_lookup()

lookup["axis_labels"]    # ["topic depth", "technical level", "abstraction"]
lookup["clusters"]       # [{id, label, centroid_3d, size, sample_texts}, ...]
lookup["n_points"]       # number of original data points
lookup["has_flow"]       # whether flow field is available
lookup["axis_ranges"]    # [{axis, min, max}, ...]
lookup["embedding_model"] # e.g. "text-embedding-3-large"
```

### `explain_path(texts)` — Path through semantic space

Pass a list of new texts. They get embedded and projected into the existing 3D space using the same reducer. Returns where each point lands, which clusters it's near, and an LLM-generated explanation of the overall path.

```python
result = query.explain_path([
    "What is a variable?",
    "How do classes work?",
    "Explain distributed systems",
])

result["path_3d"]       # [[x,y,z], [x,y,z], [x,y,z]]
result["points"]        # per-point: axis_percentages, cluster_distances, nearest_texts
result["explanation"]   # LLM-generated path explanation
```

### `query_flow_at(text)` — Flow field snapshot

Embeds a single text and queries the flow field at that position. Returns the velocity vector decomposed into:
- **Axis components**: how strongly you're being pulled along each semantic axis
- **Cluster pull**: toward/away from each cluster with alignment score
- **Nearby points**: closest original texts and whether the flow would carry you through them

```python
result = query.query_flow_at("How do I deploy to production?")

result["velocity"]             # [vx, vy, vz]
result["speed"]                # magnitude
result["axis_decomposition"]   # [{axis_label, component, magnitude, direction}, ...]
result["cluster_pull"]         # [{cluster_label, alignment, distance, interpretation}, ...]
result["nearby_points"]        # [{text, distance, velocity_alignment, would_pass_through}, ...]
```

### `query_direction_at(texts)` — Direction estimate without flow field

Like `query_flow_at` but estimates direction from the path itself (no MDN needed). Pass 2+ texts — direction is computed from consecutive differences.

```python
result = query.query_direction_at([
    "What is Python?",
    "How do I use async/await?",
    "Building production microservices",
])

result["estimated_direction"]   # [dx, dy, dz]
result["estimated_magnitude"]   # float
result["axis_decomposition"]    # same format as query_flow_at
result["cluster_pull"]          # same format as query_flow_at
```

### `path_similarity(path_a, path_b)` — Compare semantic paths

Compares two text sequences using high-dimensional embeddings (no 3D projection). Uses Frechet distance (order-aware), DTW-aligned cosine similarity, and direction alignment.

```python
result = query.path_similarity(
    ["How to read files", "How to write files", "How to delete files"],
    ["How to open DB", "How to query tables", "How to close connections"],
)

result["overall_score"]          # 0-1, higher = more similar
result["direction_similarity"]   # are the paths going in the same direction?
result["frechet_distance"]       # order-aware distance (lower = closer)
result["mean_cosine_similarity"] # average point-to-point similarity
result["start_similarity"]       # how similar are the starting points
result["end_similarity"]         # how similar are the ending points
```

## Visualization

### Interactive 3D Renderer (`launch_renderer`)

All-in-one interactive 3D viewer with particle flow animation, probe controls, and LLM explanations. Uses GPU acceleration by default, falls back to software rendering automatically when no GPU is available.

```python
from tracescope import launch_renderer

# Basic — no LLM explain
launch_renderer(result)

# With LLM explanations (pass the pipeline's explainer)
launch_renderer(result, explainer=pipeline.explainer)
```

**GUI panels** (left sidebar):
- **Flow Controls** — flow animation and ball/probe toggle
- **Display** — data points, path visibility, simple lines toggle, info overlay
- **Probe** — X/Y/Z sliders, mark/clear control points, Explain button
- **Clusters** — color-coded legend with cluster descriptions
- **Flow Settings** — particle opacity, speed multiplier, particle count slider, entropy coloring

**Double-click** on a data point in the 3D view to see its text, cluster, and metadata.

## Configuration

```python
from tracescope import TraceScopeConfig

config = TraceScopeConfig(
    openai_api_key="sk-...",          # or set OPENAI_API_KEY env var
    anthropic_api_key="sk-ant-...",   # optional, for Anthropic LLM provider
    embedding_model="text-embedding-3-large",  # or "text-embedding-3-small"
    embedding_provider_type="openai",
    llm_model="gpt-5-mini",          # for axis/cluster labeling
    llm_provider_type="openai",      # or "anthropic"
    storage_dir="~/.tracescope",     # where embeddings and caches are stored
    cache_enabled=True,              # cache LLM responses and ML results

    # Flow model settings
    flow_mode="mdn",                 # "mdn" (default) or "rbf"
    mdn_hidden=100,                  # MDN hidden layer size (50-300)
    mdn_iters=8000,                  # MDN training iterations (2000-20000)
    velocity_grid_size=40,           # 3D velocity grid resolution (20-60)
    rbf_kernel="thin_plate_spline",  # RBF kernel (see below)
    rbf_smoothing=0.1,               # RBF regularization (0 = exact)
)
```

### Flow Models

TraceScope supports two flow field models for learning velocity fields from your semantic trajectories:

**MDN (Mixture Density Network)** — Default. A 2-component neural network that learns a probabilistic velocity field. Best for complex, multi-modal flow patterns. Requires PyTorch.

```python
result = pipeline.analyze(session,
    flow_mode="mdn",
    mdn_hidden=150,     # larger = more expressive (default 100)
    mdn_iters=12000,    # more iterations = more refined (default 8000)
    velocity_grid_size=50,  # higher res grid (default 40)
)
```

**RBF (Radial Basis Function)** — Lightweight alternative using scipy's `RBFInterpolator`. Produces smoother, more conservative flows. No PyTorch required — uses only scipy.

```python
result = pipeline.analyze(session,
    flow_mode="rbf",
    rbf_kernel="thin_plate_spline",  # or "multiquadric", "cubic", "linear", "gaussian"
    rbf_smoothing=0.1,               # 0 = exact interpolation, higher = smoother
)
```

Both models produce compatible velocity grids and work identically in the visualizer and `TraceQuery` API.

### Result Caching

Save and reload full pipeline results to skip re-computation:

```python
# First run — computes everything and saves
result = pipeline.analyze(session, cache_path="results/my_analysis")

# Second run — loads instantly if texts and embedding model match
result = pipeline.analyze(session, cache_path="results/my_analysis")

# Manual save/load
result.save_result("results/my_analysis")
loaded = AnalysisResult.load_result("results/my_analysis")
```

The cache uses a SHA-256 fingerprint of sorted texts + embedding model name. If your data changes, the cache is automatically invalidated and the pipeline re-runs.

## Pipeline Steps

The `analyze()` method runs these steps:

1. **Embed** — Convert texts to high-dimensional vectors (OpenAI text-embedding-3-large, 3072D)
2. **Cluster** — Auto-select k via silhouette scoring, KMeans with k-means++
3. **Reduce to 3D** — UMAP/tSNE grid search with cosine metric, pick best silhouette
4. **Compute axes** — PCA on projected coordinates
5. **Label axes** — LLM generates 2-word semantic labels using TF-IDF keyword evolution
6. **Label clusters** — LLM generates cluster descriptions with avoid mechanism + keyword differentiation
7. **Train flow model** — MDN (mixture density network) or RBF (radial basis function) learns velocity field from the trajectory
8. **Build velocity grid** — configurable grid (default 40³) of pre-computed velocities for fast trilinear interpolation

## Project Structure

```
tracescope/
  analysis/       # Pipeline, clustering, dim reduction, MDN, explainer
  models/         # TraceEntry, TraceSession, AnalysisResult, AxisInfo
  providers/      # Embedding (OpenAI) and LLM (OpenAI/Anthropic) providers
  storage/        # ChromaDB vector store + SQLite cache
  visualization/  # 3D renderer (vispy), flow field system, probe
  query.py        # TraceQuery programmatic API
  config.py       # TraceScopeConfig
  prompts.py      # All LLM prompt templates
```
