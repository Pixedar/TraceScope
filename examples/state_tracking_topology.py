"""
Example: State-Tracking Topology
================================
Demonstrates the new analysis-layer topology API:

  query.topology_summary()        — attractors, basins, settling time,
                                     turbulence, Jacobian stability
  query.integrate_flow(text, ...) — ODE-style probe of the learned
                                     velocity field (scipy RK45)
  query.local_stability_at(text)  — numerical Jacobian / spectral radius
  query.stability_grid()          — per-cell stability over the velocity grid

The example covers four small "state-tracking" scenarios where the
relevant question is whether the model's semantic state is preserved,
flipped, or lost across turns:

  1. number-guessing state tracking
  2. "bank" ambiguity flip-flop (river vs financial)
  3. multi-turn agent debugging (state survives across turns?)
  4. math reasoning where the solution state is preserved vs lost

Run:
    python examples/state_tracking_topology.py
    python examples/state_tracking_topology.py --mdn       # use MDN flow

Notes:
  * Requires OPENAI_API_KEY in .env (or env) for embeddings.
  * Topology metrics are derived from the learned flow field — they are
    descriptions of the analysis-layer field TraceScope built, not
    guarantees about the underlying language model itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracescope import (
    TraceScopeConfig, AnalysisPipeline, from_lists, TraceQuery,
)


parser = argparse.ArgumentParser(description="State-tracking topology example")
parser.add_argument("--mdn", action="store_true",
                    help="Use MDN flow model instead of RBF")
parser.add_argument("--no-renderer", action="store_true",
                    help="Skip launching the GUI renderer at the end")
args = parser.parse_args()

_ROOT = Path(__file__).resolve().parent.parent

print("=== State-Tracking Topology ===\n")


# ─── Four state-tracking paths ─────────────────────────────────────

paths = [
    # 1. Number-guessing state tracking
    [
        "Pick a secret integer between 1 and 100, do not tell me yet.",
        "I guess 50. Is the secret higher or lower?",
        "You said higher. So the secret is between 51 and 100.",
        "I guess 75. Is the secret higher or lower?",
        "You said lower. So the secret is between 51 and 74.",
        "I guess 62. Is the secret higher or lower?",
        "You said higher. So the secret is between 63 and 74.",
        "I guess 68. Final answer.",
    ],
    # 2. "Bank" ambiguity flip-flop
    [
        "We sat on the bank and watched the slow river drift past us.",
        "Reeds and willows lined the muddy bank along the slow current.",
        "I went to the bank to deposit my paycheck before the weekend.",
        "The teller at the bank asked for my account number and ID.",
        "Back at the river, the bank was steep and slick with mud.",
        "The financial bank then approved my mortgage application.",
        "We hiked along the riverbank under the willows again.",
    ],
    # 3. Multi-turn agent debugging — does state survive across turns?
    [
        "User: my Python script crashes on startup with ImportError: numpy.",
        "Agent: which Python interpreter are you using? activate the venv first.",
        "User: I activated the venv but still get ImportError: numpy.",
        "Agent: please run `pip list | grep numpy` and paste the output.",
        "User: pip list shows numpy 1.26 installed in the venv already.",
        "Agent: try `python -c 'import sys; print(sys.executable)'` to confirm path.",
        "User: sys.executable points to the system python, not the venv python.",
        "Agent: that is the bug — your shell PATH still points to system python.",
    ],
    # 4. Math reasoning — solution state preserved vs lost
    [
        "Problem: solve 2x + 6 = 14 for x.",
        "Subtract 6 from both sides: 2x = 8.",
        "Divide both sides by 2: x = 4.",
        "Verify: 2 * 4 + 6 = 14. Correct, so x = 4.",
        "Wait, let me re-read — actually the problem said 2x + 6 = 18.",
        "Subtract 6: 2x = 12. Divide by 2: x = 6.",
        "Verify: 2 * 6 + 6 = 18. Correct, so x = 6.",
    ],
]
labels = [
    "number-guessing",
    "bank-ambiguity",
    "agent-debugging",
    "math-reasoning",
]

session = from_lists(paths, labels=labels)

print(f"Imported {len(session)} entries across {len(paths)} paths.\n")


# ─── Run the pipeline ──────────────────────────────────────────────

config = TraceScopeConfig(
    embedding_model="text-embedding-3-large",
)

_flow_mode = "mdn" if args.mdn else "rbf"
_cache_suffix = "" if args.mdn else "_rbf"

pipeline = AnalysisPipeline(config)
result = pipeline.analyze(
    session,
    progress_callback=lambda stage, pct: print(f"  [{stage}] {pct*100:.0f}%"),
    train_flow=True,
    flow_mode=_flow_mode,
    cache_path=str(_ROOT / "cache" / f"state_tracking_topology{_cache_suffix}"),
)

print(
    f"\nAnalysis complete: {result.clusters.n_clusters} clusters, "
    f"flow_trained={result.flow_model_trained}\n"
)


# ─── Topology summary ──────────────────────────────────────────────

query = TraceQuery(result, pipeline.embedding_provider)
summary = query.topology_summary(n_settling_samples=200)

print("--- topology_summary ---")
print(f"  has_flow             : {summary['has_flow']}")
print(f"  n_attractors         : {summary['n_attractors']}")
print(f"  basin_sizes          : {summary['basin_sizes']}")
print(f"  basin_fractions      : "
      f"{[round(x, 4) for x in summary['basin_fractions']]}")
print(f"  mean_settling_time   : {summary['mean_settling_time']}")
print(f"  median_settling_time : {summary['median_settling_time']}")
print(f"  fraction_converged   : {summary['fraction_converged']:.3f}")
print(f"  transition_turbulence: {summary['transition_turbulence']}")
js = summary["jacobian_stability"]
if js:
    print(
        f"  jacobian_stability   : spectral_radius "
        f"mean={js['spectral_radius_mean']:.4f} "
        f"p90={js['spectral_radius_p90']:.4f}; "
        f"divergence mean={js['divergence_mean']:.4f}"
    )


# ─── ODE-style probing for one prompt per path ─────────────────────

probe_texts = [
    "I guess 70. Is the secret higher or lower?",
    "We walked along the muddy riverbank at dusk.",
    "User: ImportError again, what should I check?",
    "Verify the answer: 2x + 6 = 18.",
]

print("\n--- integrate_flow (RK45) ---")
for text in probe_texts:
    info = query.integrate_flow(
        text, method="rk45", max_time=25.0, convergence_eps=1e-3,
    )
    print(
        f"  '{text[:46]}...'"
        f"  converged={info['converged']}"
        f"  settling_time={info['settling_time']}"
        f"  attractor_id={info['attractor_id']}"
        f"  final_speed={info['final_speed']:.5f}"
    )


# ─── Numerical Jacobian — local stability ──────────────────────────

print("\n--- local_stability_at ---")
for text in probe_texts:
    s = query.local_stability_at(text)
    print(
        f"  '{text[:46]}...'"
        f"  spectral_radius={s['spectral_radius']:.4f}"
        f"  divergence={s['divergence']:.4f}"
        f"  attracting={s['is_locally_attracting']}"
    )


# ─── Stability grid (just shape + summary stats) ───────────────────

print("\n--- stability_grid ---")
grid = query.stability_grid()
sr = grid["spectral_radius"]
dv = grid["divergence"]
print(f"  grid_size     : {grid['grid_size']}")
print(f"  spectral_radius shape={sr.shape}, "
      f"min={float(sr.min()):.4f}, mean={float(sr.mean()):.4f}, "
      f"max={float(sr.max()):.4f}")
print(f"  divergence      shape={dv.shape}, "
      f"min={float(dv.min()):.4f}, mean={float(dv.mean()):.4f}, "
      f"max={float(dv.max()):.4f}")


# ─── Optional: launch the renderer ─────────────────────────────────

if not args.no_renderer:
    try:
        from tracescope import launch_renderer
        print("\nLaunching renderer (close the window to exit)...")
        launch_renderer(result)
    except Exception as e:
        print(f"\nRenderer not launched: {e}")
