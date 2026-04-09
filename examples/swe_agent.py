"""
Example: SWE-Agent Trajectories
================================
Loads real software engineering agent trajectories from the
nebius/SWE-agent-trajectories dataset on HuggingFace.

Each trace follows an agent solving a GitHub issue through:
thought -> action -> observation cycles (file edits, test runs, debugging).

The MDN flow field reveals hidden attractors that pull agent behavior
toward success or failure patterns -- dynamics invisible to clustering alone.

Dataset: CC-BY-4.0 license
Source:  https://huggingface.co/datasets/nebius/SWE-agent-trajectories

Score channels:
  - "resolved" (per-path): 1.0 = issue resolved, 0.0 = failed

No extra dependencies required — fetches data directly from HuggingFace API.
"""

import json
import urllib.request
from pathlib import Path

from tracescope import TraceScopeConfig, AnalysisPipeline, from_lists, TraceQuery, launch_renderer

_ROOT = Path(__file__).resolve().parent.parent

# ── Load dataset from HuggingFace API ────────────────────────────

HF_API_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=nebius/SWE-agent-trajectories&config=default&split=train"
)

MAX_TRAJECTORIES = 30
MAX_STEPS_PER_TRACE = 25

print("Fetching SWE-agent-trajectories from HuggingFace API...")

paths = []
labels = []
path_scores = {}
entry_scores = []
offset = 0
batch_size = 100

while len(paths) < MAX_TRAJECTORIES:
    url = f"{HF_API_URL}&offset={offset}&length={batch_size}"
    req = urllib.request.Request(url, headers={"User-Agent": "TraceScope/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    rows = data.get("rows", [])
    if not rows:
        break

    for item in rows:
        if len(paths) >= MAX_TRAJECTORIES:
            break

        row = item.get("row", item)

        trajectory = None
        for key in ["trajectory", "steps", "messages", "history"]:
            val = row.get(key)
            if val is not None:
                if isinstance(val, str):
                    try:
                        trajectory = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        continue
                elif isinstance(val, list):
                    trajectory = val
                break

        if trajectory is None:
            steps_fallback = []
            for key in ["thought", "action", "observation", "content", "text"]:
                val = row.get(key)
                if val and isinstance(val, str) and len(val.strip()) > 10:
                    steps_fallback.append(val.strip()[:500])
            if steps_fallback:
                trajectory = steps_fallback

        if not trajectory or not isinstance(trajectory, list):
            continue

        steps = []
        step_scores = []
        for traj_item in trajectory[:MAX_STEPS_PER_TRACE]:
            if isinstance(traj_item, dict):
                for field in ["thought", "action", "observation", "content", "text", "message"]:
                    text = traj_item.get(field, "")
                    if text and len(str(text).strip()) > 10:
                        step_text = f"[{field}] {str(text).strip()[:500]}"
                        steps.append(step_text)
                        if field == "thought":
                            step_scores.append({"step_type": 0.8})
                        elif field == "action":
                            step_scores.append({"step_type": 0.5})
                        elif field == "observation":
                            step_scores.append({"step_type": 0.2})
                        else:
                            step_scores.append({"step_type": 0.5})
            elif isinstance(traj_item, str) and len(traj_item.strip()) > 10:
                steps.append(traj_item.strip()[:500])
                step_scores.append({"step_type": 0.5})

        if len(steps) < 4:
            continue

        paths.append(steps)
        entry_scores.append(step_scores)

        instance_id = row.get("instance_id") or row.get("task_id") or row.get("id") or f"traj_{len(paths)}"
        labels.append(str(instance_id)[:60])

        resolved = row.get("resolved") or row.get("success") or row.get("is_resolved")
        if resolved is not None:
            try:
                path_scores[len(paths) - 1] = {"resolved": float(bool(resolved))}
            except (ValueError, TypeError):
                pass

    offset += batch_size

total_entries = sum(len(p) for p in paths)
print(f"Extracted {len(paths)} trajectories with {total_entries} total entries")

if len(paths) < 3:
    print("WARNING: Too few valid trajectories extracted. The dataset format may have changed.")
    import sys
    sys.exit(1)

# ── Run analysis pipeline ────────────────────────────────────────

config = TraceScopeConfig()
session = from_lists(
    paths=paths,
    labels=labels,
    label="SWE-Agent Trajectories",
    entry_scores=entry_scores if entry_scores else None,
    path_scores=path_scores if path_scores else None,
)

pipeline = AnalysisPipeline(config)
result = pipeline.analyze(
    session,
    cache_path=str(_ROOT / "cache" / "swe_agent"),
    train_flow=True,
)

query = TraceQuery(result, pipeline.embedding_provider)

print("\n" + "=" * 60)
print("LOOKUP TABLE SUMMARY")
print("=" * 60)
lookup = query.get_lookup()
print(f"  Points: {lookup['n_points']}")
print(f"  Clusters: {len(lookup['clusters'])}")
print(f"  Axes: {lookup['axis_labels']}")
print(f"  Has flow: {lookup['has_flow']}")
if "score_channels" in lookup:
    print(f"  Score channels: {list(lookup['score_channels'].keys())}")

# ── GPU renderer ──────────────────────────────────────────────────

launch_renderer(
    result,
    explainer=pipeline.explainer,
    initial_state={"show_attractors": True},
)
