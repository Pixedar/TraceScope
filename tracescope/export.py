"""Export all flow analysis data (clusters, axes, attractors, paths, explanations)
to JSON and prompt-ready text files.

This does NOT run any LLM calls — it only dumps data that the user has already
generated (cached attractor explanations, saved paths, etc.).

Usage:
    from tracescope.export import export_flow_data
    export_flow_data("cache/prm_demo_rbf", output_dir="docs/export")
"""
import json
import os
import numpy as np
from tracescope.models.analysis import AnalysisResult
from tracescope.visualization.probe import probe_point


def export_flow_data(cache_path: str, output_dir: str = None,
                     score_channels: list = None,
                     sensitivity: float = 0.7) -> dict:
    """Export all cached flow data to a dict (and optionally to files).

    Args:
        cache_path: Base cache path (e.g. "cache/prm_demo_rbf").
        output_dir: If provided, writes JSON + prompt .md files here.
        score_channels: Score channels to include in attractor data.
        sensitivity: Attractor detection sensitivity (0.0-1.0).

    Returns:
        dict with all exported data.
    """
    result = AnalysisResult.load_result(cache_path)
    axis_labels = result.axis_info.labels
    cluster_labels = result.cluster_labels
    pts = result.projected_3d
    mins, maxs = pts.min(0), pts.max(0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    max_dist = float(np.linalg.norm(maxs - mins))

    # Cluster info
    clusters = []
    for c in range(result.clusters.n_clusters):
        labels_arr = np.array(result.clusters.labels)
        mask = labels_arr == c
        c_pts = pts[mask]
        name = cluster_labels[c] if c < len(cluster_labels) else f"Cluster {c}"
        clusters.append({
            'index': c,
            'name': name,
            'size': int(mask.sum()),
            'centroid': c_pts.mean(0).tolist() if len(c_pts) > 0 else [0, 0, 0],
        })

    # Attractors
    from tracescope.visualization.flow_field import FlowFieldSystem
    flow = FlowFieldSystem(result.velocity_grid, result.axis_min, result.axis_max)

    # Build score grids for each requested channel
    attractor_score_data = {}
    if score_channels:
        for ch in score_channels:
            raw_scores = result.get_entry_scores(ch)
            path_scores = result.get_path_scores(ch)
            vals = []
            for i, e in enumerate(result.session.entries):
                s = raw_scores[i]
                if s is None and e.path_id is not None:
                    s = path_scores.get(e.path_id)
                vals.append(s if s is not None else 0.5)
            attractor_score_data[ch] = np.array(vals, dtype=np.float32)

    attractors_list = []
    raw_attractors = flow.find_attractors(sensitivity=sensitivity, score_grid=None)
    centroids_3d = result.cluster_centroids_3d

    for i, att in enumerate(raw_attractors):
        pos = att['position']
        axis_pcts = [int(np.clip((pos[j] - mins[j]) / ranges[j] * 100, 0, 100))
                     for j in range(3)]

        cdists = []
        if centroids_3d is not None:
            for ci in range(len(centroids_3d)):
                d = float(np.linalg.norm(pos - centroids_3d[ci]))
                closeness = max(0, int((1 - d / max_dist) * 100))
                name = cluster_labels[ci] if ci < len(cluster_labels) else f"Cluster {ci}"
                cdists.append((name, closeness))

        # Nearest texts
        dists = np.linalg.norm(pts - pos, axis=1)
        nearest_idx = np.argsort(dists)[:5]
        nearest_texts = [result.session.entries[int(j)].text for j in nearest_idx]

        # Score means per channel
        score_means = {}
        for ch, vals in attractor_score_data.items():
            basin_vals = vals[np.linalg.norm(pts - pos, axis=1) < float(np.percentile(dists, 10))]
            if len(basin_vals) > 0:
                score_means[ch] = round(float(np.mean(basin_vals)), 3)

        att_data = {
            'index': i,
            'label': f'A{i + 1}',
            'position': [float(x) for x in pos],
            'strength': float(att['strength']),
            'basin_fraction': float(att['basin_fraction']),
            'divergence': float(att['divergence']),
            'axis_percentages': dict(zip(axis_labels, axis_pcts)),
            'cluster_distances': cdists,
            'nearest_texts': nearest_texts,
            'score_means': score_means,
        }
        attractors_list.append(att_data)

    # Load cached attractor explanations
    att_explain_path = f"{cache_path}_attractor_explanations.json"
    attractor_explanations = {}
    if os.path.exists(att_explain_path):
        with open(att_explain_path, 'r', encoding='utf-8') as f:
            attractor_explanations = json.load(f)

    # Attach explanations to attractors
    for att in attractors_list:
        key = str(att['index'])
        if key in attractor_explanations:
            att['explanation'] = attractor_explanations[key]

    # Load saved paths
    paths_file = f"{cache_path}_explained_paths.json"
    saved_paths = []
    if os.path.exists(paths_file):
        with open(paths_file, 'r', encoding='utf-8') as f:
            saved_paths = json.load(f)

    data = {
        'cache_path': cache_path,
        'flow_mode': result.flow_mode,
        'n_entries': len(result.session.entries),
        'axis_labels': axis_labels,
        'clusters': clusters,
        'attractors': attractors_list,
        'saved_paths': saved_paths,
        'score_channels': result.score_channels,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(cache_path)

        # JSON export
        json_path = os.path.join(output_dir, f"{basename}_export.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Exported JSON: {json_path}")

        # Prompt-ready markdown
        md_path = os.path.join(output_dir, f"{basename}_prompt.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(_build_prompt_md(data))
        print(f"Exported prompt: {md_path}")

    return data


def _build_prompt_md(data: dict) -> str:
    """Build a prompt-ready markdown from exported data."""
    lines = []
    lines.append(f"# Flow Analysis Export: {data['cache_path']}")
    lines.append(f"Flow mode: {data['flow_mode']} | Entries: {data['n_entries']}")
    lines.append("")

    # Axes
    lines.append("## Semantic Axes")
    for i, label in enumerate(data['axis_labels']):
        lines.append(f"- Axis {i + 1}: {label}")
    lines.append("")

    # Clusters
    lines.append("## Clusters")
    for c in data['clusters']:
        lines.append(f"- **{c['name']}** ({c['size']} entries)")
    lines.append("")

    # Attractors
    lines.append("## Attractors")
    for att in data['attractors']:
        lines.append(f"### {att['label']} (strength {att['strength']:.0%})")
        lines.append(f"Basin: {att['basin_fraction']:.2%} | Divergence: {att['divergence']:.4f}")
        lines.append("")
        lines.append("Location:")
        for name, pct in att['axis_percentages'].items():
            lines.append(f"  - {name}: {pct}%")
        lines.append("")
        if att.get('score_means'):
            lines.append("Score means:")
            for ch, val in att['score_means'].items():
                lines.append(f"  - {ch}: {val}")
            lines.append("")
        if att.get('cluster_distances'):
            lines.append("Cluster proximity:")
            for name, closeness in att['cluster_distances']:
                lines.append(f"  - {name}: {closeness}%")
            lines.append("")
        if att.get('explanation'):
            lines.append("**Explanation:**")
            lines.append(att['explanation'])
            lines.append("")

    # Paths
    if data['saved_paths']:
        lines.append("## Explained Paths")
        for i, sp in enumerate(data['saved_paths']):
            lines.append(f"### Path P{i + 1}")
            journey = sp.get('journey', {})

            # Journey metadata
            trend = journey.get('score_trend', 'unknown')
            visited = journey.get('attractors_visited', [])
            n_steps = journey.get('n_steps', 0)
            dwelling = journey.get('dwelling_fraction', 0)
            visited_str = ' → '.join(f'A{v+1}' for v in visited) if visited else 'none'
            lines.append(f"Trend: **{trend}** | Steps: {n_steps} | "
                         f"Basins visited: {visited_str} | "
                         f"Dwelling: {dwelling:.0%}")

            # Score journey (from dense samples)
            samples = journey.get('samples', [])
            if samples:
                lines.append("Score journey:")
                for s in samples:
                    frac = s.get('step_frac', 0)
                    scores_str = ', '.join(f'{k}={v:.2f}'
                                           for k, v in s.get('scores', {}).items())
                    att = s.get('nearest_attractor')
                    att_str = f'near A{att+1}' if att is not None else ''
                    pct = int(frac * 100)
                    lines.append(f"  {pct:3d}%: {scores_str} {att_str}")
                lines.append("")

            if sp.get('explanation'):
                lines.append("**Explanation:**")
                lines.append(sp['explanation'])
                lines.append("")

    return "\n".join(lines)
