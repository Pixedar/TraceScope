# import json
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.neural_network import MLPRegressor
# from sklearn.ensemble import VotingRegressor
# from sklearn.compose import TransformedTargetRegressor
#
# _models, _mean, _scale, _min, _max = {}, None, None, None, None
# _axis_idx = {"x": 0, "y": 1, "z": 2}
#
# # def _fit_one(X, y):
# #     # 1) Scale inputs
# #     sc_in = StandardScaler()
# #     Xs = sc_in.fit_transform(X)
# #
# #     # 2) Create your two base learners
# #     kr = KernelRidge(alpha=0.05, kernel='rbf', gamma=0.2)
# #     mlp = MLPRegressor(
# #         hidden_layer_sizes=(16, 16),
# #         alpha=1e-2,
# #         activation='tanh',
# #         solver='lbfgs',             # more stable than 'adam'
# #         early_stopping=True,        # stops if no improvement
# #         n_iter_no_change=20,
# #         max_iter=2000,
# #         random_state=42
# #     )
# #
# #     # 3) Voting ensemble
# #     base = VotingRegressor([('krr', kr), ('mlp', mlp)],
# #                            weights=[0.7, 0.3])
# #
# #     # 4) Wrap in a TransformedTargetRegressor to scale y
# #     regr = TransformedTargetRegressor(
# #         regressor=base,
# #         transformer=StandardScaler()
# #     )
# #
# #     # 5) Fit on scaled X and y
# #     regr.fit(Xs, y)
# #     return dict(sc_in=sc_in, net=regr)
#
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import (
#     Matern, WhiteKernel, ConstantKernel as C
# )
#
# def _fit_one(X, y):
#     sc_in = StandardScaler()
#     Xs    = sc_in.fit_transform(X)
#
#     kernel = C(1.0, (1e-2, 1e2)) * Matern(
#                  length_scale=[1., 1. ,1.],      # ARD
#                  length_scale_bounds=(1e-2, 1e3),
#                  nu=1.5) + WhiteKernel(noise_level=1.,
#                                        noise_level_bounds=(1e-4, 1e2))
#
#     gpr = GaussianProcessRegressor(kernel=kernel,
#                                    n_restarts_optimizer=10,
#                                    normalize_y=True)
#
#     gpr.fit(Xs, y)
#     return dict(sc_in=sc_in, net=gpr)
#
#
# def train_models(cloud_json):
#     global _models, _mean, _scale, _min, _max
#
#     pts = np.asarray(json.loads(cloud_json), dtype=np.float32)
#     x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
#
#     _min, _max = pts.min(axis=0), pts.max(axis=0)
#     sc3 = StandardScaler().fit(pts)
#     _mean, _scale = sc3.mean_, sc3.scale_
#
#     _models = {
#         "x": _fit_one(np.c_[y, z], x),
#         "y": _fit_one(np.c_[x, z], y),
#         "z": _fit_one(np.c_[x, y], z)
#     }
#
#     return json.dumps({
#         "axis_min": _min.tolist(),
#         "axis_max": _max.tolist()
#     })
#
# def predict_axis(axis, a1, a2):
#     mdl = _models[axis]
#     xs  = mdl["sc_in"].transform([[a1, a2]])
#     # net.predict already returns de-scaled y thanks to TransformedTargetRegressor
#     raw = float(mdl["net"].predict(xs)[0])
#
#     # clamp to original data bounds
#     idx = _axis_idx[axis]
#     lo, hi = _min[idx], _max[idx]
#     return max(lo, min(hi, raw))

#Version 2
# import json
#
#
# _models, _min, _max = {}, None, None
# _axis_idx = {"x": 0, "y": 1, "z": 2}
#
# # ────────────────────────────────────────────────────────────────────────────────
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import (
#     Matern, WhiteKernel, ConstantKernel as C
# )
# from sklearn.preprocessing import StandardScaler
# import numpy as np
#
#
# def _fit_one(X, y):
#     """
#     GP = SHORT Matérn ν=2.5  +  LONG Matérn ν=2.5  +  White
#     ─────────────────────────────────────────────────────────
#     * SHORT keeps medium-scale bends (ℓ ≈ 0.25σ)
#     * LONG gives global shape (ℓ ≈ 1.2σ)
#     * Amplitude of SHORT can vary 0.25–0.65·σy
#     * Raised noise prior + looser SHORT bounds → less spike, more shape
#     """
#     sc_in = StandardScaler()
#     Xs = sc_in.fit_transform(X)
#     n_dims = Xs.shape[1]
#
#     std_y  = np.std(y)
#     var_y  = np.var(y)
#
#
#     # quick adjustment
#     short = C(0.45 * std_y, (0.3 * std_y, 0.8 * std_y)) * \
#             Matern(length_scale=np.full(n_dims, 0.3),
#                    length_scale_bounds=(0.15, 1.0), nu=2.5)
#
#     long = C(1.0 * std_y, (1e-2, 4.0 * std_y)) * \
#            Matern(length_scale=np.full(n_dims, 1.0),
#                   length_scale_bounds=(0.8, 2.0), nu=2.5)
#
#     # ── White noise prior: bigger → accepts more averaging → fewer wiggles
#     noise = WhiteKernel(
#         noise_level=max(var_y / 3.0, 2e-2),   # ↑ from /5 to /3
#         noise_level_bounds=(1e-3, 1e3)
#     )
#
#     kernel = short + long + noise
#
#     gpr = GaussianProcessRegressor(
#         kernel=kernel,
#         normalize_y=True,
#         alpha=1e-6,           # jitter
#         n_restarts_optimizer=6,
#         random_state=42
#     )
#     gpr.fit(Xs, y)
#
#     return {"sc_in": sc_in, "net": gpr}
#
# # ────────────────────────────────────────────────────────────────────────────────
# def train_models(cloud_json):
#     """
#     Fit three conditional GPs:
#         x̂(y,z),  ŷ(x,z),  ẑ(x,y)
#     Return the raw axis min/max for UI scaling.
#     """
#     global _models, _min, _max
#
#     pts = np.asarray(json.loads(cloud_json), dtype=np.float32)
#     x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
#
#     _min, _max = pts.min(axis=0), pts.max(axis=0)
#
#     _models = {
#         "x": _fit_one(np.c_[y, z], x),
#         "y": _fit_one(np.c_[x, z], y),
#         "z": _fit_one(np.c_[x, y], z)
#     }
#
#     return json.dumps({"axis_min": _min.tolist(),
#                        "axis_max": _max.tolist()})
#
# # ────────────────────────────────────────────────────────────────────────────────
# def predict_axis(axis, a1, a2):
#     """
#     Predict the passive axis given the other two coordinates.
#     """
#     mdl = _models[axis]
#     xs  = mdl["sc_in"].transform([[a1, a2]])
#     raw = float(mdl["net"].predict(xs)[0])   # prediction is already in raw units
#
#     # clamp to training bounds
#     idx = _axis_idx[axis]
#     lo, hi = _min[idx], _max[idx]
#     return max(lo, min(hi, raw))


"""
axes_dyn.py
===========

Three modes:
  • static  – your current per-axis conditional GP (kept as-is)
  • gpssm   – GP drift + Student-t process noise  (Pyro)
  • mdn     – 2-component mixture-density network (PyTorch)

Dependencies you probably already have:
  numpy, scikit-learn
Extra lightweight wheels (pip install once):
  torch==2.2.*, pyro-ppl==1.9.*, tqdm
"""
import json, io, math, random, contextlib, warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import itertools

# Optional: torch and matplotlib are only needed for MDN mode and visualization
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D
except ImportError:
    plt = None
    animation = None
# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────
def _fit_conditional_gp(X, y):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    n  = Xs.shape[1]
    std = np.std(y); var = np.var(y)
    k = (C(0.45*std,(0.3*std,0.8*std))*Matern(np.full(n,0.3),
         (0.15,1.0),nu=2.5) +
         C(1.0*std,(1e-2,4.*std))*Matern(np.full(n,1.0),
         (0.8,2.0),nu=2.5) +
         WhiteKernel(max(var/3.,2e-2),(1e-3,1e3)))
    gpr = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                   alpha=1e-6, n_restarts_optimizer=6,
                                   random_state=42)
    gpr.fit(Xs, y)
    return {"sc": sc, "net": gpr}

# ────────────────────────────────────────────────────────────────────────────
#  Mode registry  (populated further down)
# ────────────────────────────────────────────────────────────────────────────
_MODE_FNS = {}

def register(name):
    def _(fn): _MODE_FNS[name] = fn; return fn
    return _

# ────────────────────────────────────────────────────────────────────────────
#  1. static  (kept exactly as before)
# ────────────────────────────────────────────────────────────────────────────
@register("static")
def _train_static(points):
    x,y,z = points.T
    mdl = {"x": _fit_conditional_gp(np.c_[y,z], x),
           "y": _fit_conditional_gp(np.c_[x,z], y),
           "z": _fit_conditional_gp(np.c_[x,y], z)}
    axis_min, axis_max = points.min(0), points.max(0)
    def predict(axis,a1,a2):
        m  = mdl[axis]
        raw = float(m["net"].predict(m["sc"].transform([[a1,a2]]))[0])
        idx = {"x":0,"y":1,"z":2}[axis]
        return np.clip(raw, axis_min[idx], axis_max[idx])
    return {"kind":"static",
            "axis_min":axis_min, "axis_max":axis_max,
            "predict":predict}

# ────────────────────────────────────────────────────────────────────────────
# 2. Heavy-tail GP State-Space Model (Student-t process noise)
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
# 2. Heavy-tail GP State-Space Model  (Student-t innovations, sparse drift)
#     works on PyTorch 1.13  +  Pyro 1.7  (no contrib.ssgp needed)
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
# 2. Heavy-tail GP State-Space Model  (sparse drift + Student-t noise)
#     compatible with PyTorch 1.13.1  &  Pyro 1.7.0
# ────────────────────────────────────────────────────────────────────────────
@register("gpssm")
def _train_gpssm(points,
                 ν: float  = 2.5,      # Student-t df
                 M: int    = 20,       # inducing points per GP
                 iters: int = 2_000,
                 lr: float  = 1e-2,
                 kernel_l=0.5):
    """
    Fits three independent SparseGPRegression models to the velocity field
      ΔS = S[t+1] − S[t]
    and simulates trajectories with Euler integration and Student-t noise.
    """
    import math, random, torch, pyro
    from pyro.contrib.gp.kernels import Matern52
    from pyro.contrib.gp.models  import SparseGPRegression
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam

    torch.set_default_dtype(torch.float32)

    # ─── 1. build training data ───────────────────────────────────────────
    S   = torch.as_tensor(points, dtype=torch.float32)        # (N,3)
    dS  = S[1:] - S[:-1]                                      # (N-1,3)
    X   = S[:-1].contiguous()                                 # (N-1,3)
    N   = X.shape[0]

    M = min(M, N)
    Xu = X[random.sample(range(N), k=M)].clone()              # (M,3)

    # ─── 2. three sparse GPs (one per axis) ──────────────────────────────
    gps = []
    for k in range(3):                                        # x, y, z
        yk = dS[:, k].contiguous()            # 1-D tensor, shape (N-1,)
        # kern = Matern52(input_dim=3, lengthscale=torch.ones(3),kernel_l)
        kern = Matern52(input_dim=3,
                        lengthscale=torch.full((3,), kernel_l))
        gp   = SparseGPRegression(X, yk, kern, Xu=Xu, jitter=1e-5)

        # Variational training loop
        svi = SVI(gp.model, gp.guide,
                  Adam({"lr": lr}),
                  Trace_ELBO())
        for _ in range(iters):
            svi.step()

        gps.append(gp)

    axis_min, axis_max = points.min(0), points.max(0)
    a_min = torch.as_tensor(axis_min, dtype=torch.float32)
    a_max = torch.as_tensor(axis_max, dtype=torch.float32)
    studentT = torch.distributions.StudentT(df=ν)

    # helper:     μ(p) ∈ ℝ³
    def _drift(p: torch.Tensor) -> torch.Tensor:
        p = p.unsqueeze(0)                                    # (1,3)
        return torch.stack([gp(p)[0] for gp in gps])          # (3,)

    # public API: simulate a path
    def _drift(p: torch.Tensor) -> torch.Tensor:
        p2d = p.reshape(1, -1)  # (1,3)
        with torch.no_grad():
            # each gp(p2d)[0] gives shape (1,)
            mu = [gp(p2d)[0] for gp in gps]
            return torch.cat(mu, dim=0)  # (3,)   ← flat!

        # ▼ public API: generate a trajectory

    def simulate(start, steps=240, dt=1.0, sample=True,
                 noise_scale=1.0):
        with torch.no_grad():  # no autograd
            s = torch.as_tensor(start, dtype=torch.float32)
            path = [s.cpu().numpy().tolist()]
            for _ in range(steps):
                drift = _drift(s)
                noise = (studentT.sample((3,))* noise_scale
                         if sample else torch.zeros_like(drift))
                s = s + drift * dt + noise * math.sqrt(dt)
                s = torch.max(torch.min(s, a_max), a_min)  # clamp
                path.append(s.cpu().numpy().tolist())
            return path

    return {
        "kind":      "gpssm",
        "axis_min":  axis_min,
        "axis_max":  axis_max,
        "simulate":  simulate,
    }
# ─────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


# @register("gpssm")
# def _train_gpssm(points, ν=3.0):
#     """
#     Sparse variational GP drift  +  Student-t innovations.
#     Because we only have 100-300 steps, we use 20 inducing points
#     and 1-D Euler integration during simulation.
#     """
#     import torch, gpytorch, pyro, pyro.contrib.ssgp as ssgp
#     torch.set_default_dtype(torch.float32)
#     S  = torch.from_numpy(points.astype(np.float32))
#     ΔS = S[1:] - S[:-1]
#     X  = S[:-1]
#
#     # sparse GP drift
#     m = ssgp.StateSpaceGP(state_dim=3, inducing_points=20,
#                           noise_dist="studentT", nu=ν)
#     guide = ssgp.AutoNormal(m)
#     svi   = pyro.infer.SVI(m, guide,
#                            pyro.optim.Adam({"lr":1e-2}),
#                            loss=pyro.infer.Trace_ELBO())
#     for _ in range(2000):
#         svi.step(X, ΔS)
#
#     axis_min, axis_max = points.min(0), points.max(0)
#
#     def simulate(start, steps=240, dt=1.0, sample=True):
#         s = torch.tensor(start, dtype=torch.float32)
#         out = [s.numpy().tolist()]
#         for _ in range(steps):
#             mu, var = m(s.unsqueeze(0))
#             drift   = mu.squeeze(0)
#             if sample:
#                 noise = m.sample_noise().squeeze(0)
#             else:
#                 noise = torch.zeros_like(drift)
#             s = s + drift*dt + noise*math.sqrt(dt)
#             s = torch.max(torch.min(s, torch.from_numpy(axis_max)), torch.from_numpy(axis_min))
#             out.append(s.numpy().tolist())
#         return out
#     return {"kind":"gpssm",
#             "axis_min":axis_min,"axis_max":axis_max,
#             "simulate":simulate}

# ────────────────────────────────────────────────────────────────────────────
# 3. Mixture-Density Velocity Field (tiny 2-component MDN)
# ────────────────────────────────────────────────────────────────────────────
# @register("mdn")
# def _train_mdn(points, hidden=160, iters=8000, lr=3e-3, clip=8.0,noise_temp=1.0):
# # def _train_mdn(points, hidden=10, iters=3000, lr=3e-3, clip=8.0,noise_temp=1.0):
#     """
#     Much stabler training:
#       • σ is forced positive with softplus
#       • log-sum-exp trick for the mixture likelihood
#       • gradient clipping prevents runaway updates
#     """
#     import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
#     torch.manual_seed(42)
#
#     S   = torch.from_numpy(points.astype(np.float32))
#     X   = S[:-1]                     # shape (N-1, 3)
#     ΔS  = S[1:] - S[:-1]             # target velocities
#
#     class MDN(nn.Module):
#         def __init__(self, h):
#             super().__init__()
#             self.feat = nn.Sequential(nn.Linear(3, h), nn.Tanh(),
#                                       nn.Linear(h, h), nn.Tanh())
#             self.out  = nn.Linear(h, 3 + 3 + 2 + 1)   # μ1|μ2|logσ1|logσ2|logitπ
#         def forward(self, x):
#             t = self.out(self.feat(x))
#             μ1, μ2 = t[..., :3], t[..., 3:6]
#             # softplus → strictly positive σ, bottom-clamped at 1e-3
#             σ1 = F.softplus(t[..., 6:7]) + 1e-3
#             σ2 = F.softplus(t[..., 7:8]) + 1e-3
#             logitπ = t[..., 8:]
#             return μ1, μ2, σ1, σ2, logitπ.squeeze(-1)
#
#     net = MDN(hidden)
#     opt = optim.Adam(net.parameters(), lr=lr)
#
#
#     rng = torch.Generator().manual_seed(42)
#
#     λ = 1e-3                          # Jacobian weight
#     for epoch in range(iters):
#         μ1, μ2, σ1, σ2, logitπ = net(X)
#         def nll(μ, σ):
#             return 0.5 * ((ΔS-μ)/σ).pow(2).sum(-1) + 3*torch.log(σ.squeeze(-1))
#         logp1 = torch.log(torch.sigmoid(logitπ)+1e-9) - nll(μ1, σ1)
#         logp2 = torch.log(torch.sigmoid(-logitπ)+1e-9) - nll(μ2, σ2)
#         nll_all = -torch.logsumexp(torch.stack([logp1, logp2]), dim=0)
#
#         # ––– leave-one-out slice
#         j = torch.randint(0, X.size(0), (), generator=rng)
#         loo_nll = nll_all[j]
#
#         # ––– Hutchinson-trace Jacobian penalty (fast on CPU)
#         X_sub = X[torch.randperm(X.size(0), generator=rng)[:int(0.3*X.size(0))]]
#         v = torch.randn_like(X_sub)
#         def f(inp): return net(inp)[0]          # use μ1 only
#         Hv = torch.autograd.functional.jvp(f, X_sub, v, create_graph=True)[1]
#         jac_pen = (Hv.pow(2).sum(-1)).mean()    # ≈ ‖∂v/∂x‖² trace
#
#         loss = nll_all.mean() + loo_nll + λ * jac_pen
#         opt.zero_grad(); loss.backward()
#         nn.utils.clip_grad_norm_(net.parameters(), clip)
#         opt.step()
#
#
#     axis_min, axis_max = points.min(0), points.max(0)
#
#
#     net.metrics = {
#         'loo_nll': loo_nll.item(),
#         'jacobian': jac_pen.item()
#     }
#
#     @torch.no_grad()
#     def simulate(start, steps=2, dt=0.000001,
#                  sample=True,
#                  noise_temp=noise_temp):  # ← propagate default
#         # s = torch.tensor(start, dtype=torch.float32)
#         # cast dt to float32 so v * dt stays float32
#         dt_t = torch.tensor(dt, dtype=torch.float32)
#         s = torch.tensor(start, dtype=torch.float32)
#
#         out = [s.numpy().tolist()]
#         a_min = torch.tensor(axis_min + 0.1, dtype=torch.float32)
#         a_max = torch.tensor(axis_max - 0.1, dtype=torch.float32)
#         rng = torch.distributions.Normal(torch.zeros(3), torch.ones(3))
#         for _ in range(steps):
#             μ1, μ2, σ1, σ2, logitπ = net(s)
#             π = torch.sigmoid(logitπ)
#
#             if sample and torch.rand(1).item() < π.item():
#                 v = μ1 + rng.sample() * σ1 * noise_temp  # ← scaled
#             elif sample:
#                 v = μ2 + rng.sample() * σ2 * noise_temp  # ← scaled
#             else:
#                 v = μ1  # deterministic
#
#             # s = s + v * dt
#             # s = torch.max(torch.min(s, a_max), a_min)
#             # out.append(s.numpy().tolist())
#             s = s + v * dt_t
#             # cast back to float32 so net weights (float32) match
#             s = s.to(torch.float32)
#             s = torch.max(torch.min(s, a_max), a_min)
#             out.append(s.numpy().tolist())
#         return out
#
#     # return {"kind": "mdn",
#     #         "axis_min": axis_min, "axis_max": axis_max,
#     #         "simulate": simulate}
#     # include the trained network and hp in the returned dict
#     return {
#         "kind":      "mdn",
#         "net":       net,            # ← so evaluate_real can find it
#         "hidden":    hidden,         # ← for bookkeeping
#         "iters":     iters,          # ← for bookkeeping
#         "axis_min":  axis_min,
#         "axis_max":  axis_max,
#         "simulate":  simulate
#     }



# ────────────────────────────────────────────────────────────────────────────
#  Public API exposed to Chaquopy
# ────────────────────────────────────────────────────────────────────────────
_state = {}   # keeps whichever model was trained last

def train_models(cloud_json:str, mode:str="static", path_ids=None, **kwargs):
    """
    mode ∈ {"static","gpssm","mdn","gpvf","rbf"}
    path_ids: optional array of int path IDs (same length as points).
              When provided, MDN training skips velocity pairs that cross
              path boundaries so multiple independent paths can be analyzed
              together without spurious boundary velocities.
    kwargs: forwarded to the mode function (e.g. hidden, iters, lr for MDN).
    """
    pts = np.asarray(json.loads(cloud_json), dtype=np.float32)
    if pts.shape[0] < 4:
        raise ValueError("Need ≥4 points")

    if mode not in _MODE_FNS:
        raise ValueError(f"unknown mode {mode}")

    _state.clear()
    call_kwargs = dict(kwargs)
    if path_ids is not None and mode == "mdn":
        call_kwargs["path_ids"] = path_ids
    _state.update(_MODE_FNS[mode](pts, **call_kwargs))

    # Expand bounds with margin so particles have room around the data.
    # RBF already applies 10% internally; MDN/GPVF return exact data bounds.
    # Use a larger margin for small datasets where the span is tiny.
    a_min = np.asarray(_state["axis_min"], dtype=np.float32)
    a_max = np.asarray(_state["axis_max"], dtype=np.float32)
    span = a_max - a_min
    margin = 0.15 if len(pts) < 20 else 0.10
    a_min = a_min - margin * span
    a_max = a_max + margin * span
    _state["axis_min"] = a_min
    _state["axis_max"] = a_max

    return json.dumps({"axis_min": a_min.tolist(),
                       "axis_max": a_max.tolist()})

# for slider coupling (only “static” model supports it)
def predict_axis(axis:str, a1:float, a2:float):
    if _state.get("kind") != "static":
        raise RuntimeError("predict_axis only valid in static mode")
    return _state["predict"](axis, a1, a2)

# for the “Play” button animation  (gpssm / mdn)
def simulate_path(mode:str,
                  start_x:float, start_y:float, start_z:float,
                  steps:int=32, dt:float=0.1, sample_jumps:bool=True):
    if _state.get("kind") != mode:
        raise RuntimeError("model not trained for requested mode")
    path = _state["simulate"]([start_x,start_y,start_z],
                              steps,dt,sample_jumps)
    return json.dumps(path)


@register("gpvf")
def _train_gpvf(points, dt=1.0):
#     import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel as W, ConstantKernel as C

    X_raw = points[:-1]
    V     = (points[1:] - points[:-1]) / dt      # velocities

    sc = StandardScaler().fit(X_raw)
    Xs = sc.transform(X_raw)

    ℓ_init = np.ones(3)
    kern   = C(1.0, (1e-3, 1e3)) * \
             RBF(ℓ_init, length_scale_bounds=(1e-1, 1e2)) + W(1e-4)

    gpx = GaussianProcessRegressor(kern, alpha=1e-6, normalize_y=True,
                                   n_restarts_optimizer=4).fit(Xs, V[:, 0])
    gpy = GaussianProcessRegressor(kern, alpha=1e-6, normalize_y=True,
                                   n_restarts_optimizer=4).fit(Xs, V[:, 1])
    gpz = GaussianProcessRegressor(kern, alpha=1e-6, normalize_y=True,
                                   n_restarts_optimizer=4).fit(Xs, V[:, 2])

    def simulate(start, steps=240, dt=1.0,
                 sample_jumps=False, noise_std=0.0):
        p   = np.asarray(start, dtype=np.float32)
        out = [p.tolist()]
        for _ in range(steps):
            ps = sc.transform([p])           # scale query
            v  = np.array([gpx.predict(ps)[0],
                           gpy.predict(ps)[0],
                           gpz.predict(ps)[0]])
            if sample_jumps and noise_std > 0:
                v += np.random.normal(0, noise_std, 3)
            p  = p + v * dt
            out.append(p.tolist())
        return out

    return {"kind": "gpvf",
            "simulate": simulate,
            "axis_min": points.min(0),
            "axis_max": points.max(0)}



# def evaluate_real(model, pts, dt, seeds=10):
#     """Return REAL′ and its three components."""
#     import xarray as xr
#     import arviz as az, scipy.stats as ss
#     from sklearn.metrics import adjusted_rand_score
#     from sklearn.cluster import DBSCAN
#     import torch
#     S = torch.from_numpy(pts.astype(np.float32))
#     X, ΔS = S[:-1], S[1:] - S[:-1]
#
#     # 1) PSIS-LOO on velocity log-likelihood
#     μ1, μ2, σ1, σ2, logitπ = model['net'](X)
#     def logli(μ, σ):
#         # (ΔS-μ)/σ  broadcasts σ=[N,1] → [N,3] correctly
#         mse_term = -0.5 * ((ΔS - μ) / σ).pow(2).sum(-1)      # → [N]
#         logdet   = -3   * torch.log(σ.squeeze(-1))          # → [N]
#         return mse_term + logdet
#
#     lps = torch.logsumexp(torch.stack([
#             torch.log(torch.sigmoid(logitπ))  + logli(μ1, σ1),
#             torch.log(torch.sigmoid(-logitπ)) + logli(μ2, σ2)]), dim=0)
#     # wrap the pointwise log-likelihoods into xarray + InferenceData
#     import xarray as xr
#
#     ll_arr = xr.DataArray(lps.detach().numpy(), dims=("obs",))
#     idata = az.from_dict(log_likelihood={"y": ll_arr})
#     loo = model['net'].metrics['loo_nll']
#
#     # 2) long-run density ρ-W₁
#     sim = []
#     box = (pts.min(0), pts.max(0))
#     for s in np.random.uniform(box[0], box[1], size=(200,3)):
#         sim.extend(model['simulate'](s.tolist(), steps=int(10/dt), dt=dt, sample=True))
#     sim = np.asarray(sim)
#     rho_w1 = ss.wasserstein_distance(pts.flatten(), sim[:len(pts)].flatten())
#
#     # 3) attractor stability: bootstrap ARI + ensemble R
#     # --- bootstrap ARI
#     from sklearn.utils import resample
#     aris = []
#     for _ in range(30):
#         b_pts = resample(pts)
#         b_mdl = _train_mdn(b_pts, hidden=model['hidden'], iters=1000, lr=3e-3)
#         path = b_mdl['simulate'](b_pts[0].tolist(), steps=2000, dt=dt, sample=True)
#         Y = np.asarray(path)[-500:]
#         lab = DBSCAN(eps=0.04*np.ptp(pts,0).mean(), min_samples=20).fit(Y).labels_
#         aris.append(adjusted_rand_score(lab, lab))
#     ari = np.mean(aris)
#
#     # --- ensemble R
#     hits = 0
#     for sd in range(seeds):
#         torch.manual_seed(sd)
#         e_mdl = _train_mdn(pts, hidden=model['hidden'], iters=model['iters'])
#         path = e_mdl['simulate'](pts[0].tolist(), steps=2000, dt=dt, sample=True)
#         Y = np.asarray(path)[-500:]
#         lab = DBSCAN(eps=0.04*np.ptp(pts,0).mean(), min_samples=20).fit(Y).labels_
#         hits += (len(np.unique(lab)) > 1)   # crude: ≥1 cluster means attractor present
#     R = hits / seeds
#
#     real = 0.4*loo + 0.4*(1 - rho_w1) + 0.2*(0.5*ari + 0.5*R)
#     return real, loo, rho_w1, ari, R

# def evaluate_real(model, pts, dt, seeds=10):
#     """
#     Return REAL′ and its four normalised components.
#     Works even if the MDN was created with float64 weights elsewhere.
#     """
#     import numpy as np, torch, warnings, arviz as az, xarray as xr, scipy.stats as ss
#     from sklearn.cluster import DBSCAN
#     from sklearn.metrics import adjusted_rand_score
#     from sklearn.utils import resample
#
#     # ── 0)  ***force network to float32 so inputs match***
#     model["net"] = model["net"].float()
#
#     # ────────────────────────────────── 1) PSIS-LOO  (unchanged except for .elpd_loo)
#     S  = torch.from_numpy(pts.astype(np.float32))
#     X  = S[:-1]
#     dS = S[1:] - S[:-1]
#
#     μ1, μ2, σ1, σ2, logitπ = model["net"](X)
#     def logli(mu, sig):
#         return -0.5 * ((dS - mu) / sig).pow(2).sum(-1) - 3 * torch.log(sig.squeeze(-1))
#
#     lps = torch.logsumexp(torch.stack([
#         torch.log(torch.sigmoid(logitπ)) + logli(μ1, σ1),
#         torch.log(torch.sigmoid(-logitπ)) + logli(μ2, σ2)
#     ]), dim=0).detach().cpu().numpy()  # ← detach() added
#
#     ll_da  = xr.DataArray(np.tile(lps, (4, 1))[None], dims=("chain", "draw", "obs"))
#     post_da = xr.DataArray(np.zeros((1, 4)), dims=("chain", "draw"))
#     idata   = az.from_dict(posterior={"dummy": post_da},
#                            log_likelihood={"y": ll_da})
#
#     with warnings.catch_warnings():                    # silence Pareto-k warning
#         warnings.filterwarnings("ignore", category=UserWarning,
#                                 message="Estimated shape parameter")
#         psis_loo = az.loo(idata, pointwise=False).elpd_loo  # scalar
#
#     psis_s = np.clip((psis_loo / len(pts) + 3.0) / 2.5, 0, 1)
#
#     # ────────────────────────────────── 2) ρ-W₁ density drift  (unchanged)
#     box_lo, box_hi = pts.min(0), pts.max(0)
#     sims = []
#     for s in np.random.uniform(box_lo, box_hi, size=(200, 3)):
#         sims += model["simulate"](s.tolist(), steps=int(10 / dt), dt=dt, sample=True)
#     sims    = np.asarray(sims)[: len(pts)]
#     rho_w1  = ss.wasserstein_distance(pts.flatten(), sims.flatten())
#     # rho_s   = np.clip(1 - rho_w1 / (0.30 * np.ptp(pts)), 0, 1)
#     rho_scale = 0.25
#     # use the L2‐norm of the per‐axis ranges instead of just the max‐minus‐min along one axis
#     scale = rho_scale * np.linalg.norm(np.ptp(pts, axis=0))
#     rho_s = np.clip(1 - rho_w1 / scale, 0, 1)
#
#     # ────────────────────────────────── 3a) bootstrap ARI  (unchanged)
#     ref_Y   = np.asarray(model["simulate"](pts[0].tolist(), 2000, dt, True))[-500:]
#     ref_lab = DBSCAN(eps=0.04 * np.ptp(pts), min_samples=20).fit(ref_Y).labels_
#     aris = []
#     for _ in range(30):
#         b_pts = resample(pts)
#         b_mdl = _train_mdn(b_pts, hidden=model["hidden"], iters=1000, lr=3e-3)
#         b_mdl["net"] = b_mdl["net"].float()            # cast quick re-fit too
#         b_Y   = np.asarray(b_mdl["simulate"](b_pts[0].tolist(), 2000, dt, True))[-500:]
#         lab   = DBSCAN(eps=0.04 * np.ptp(pts), min_samples=20).fit(b_Y).labels_
#         aris.append(adjusted_rand_score(ref_lab, lab))
#     ari_s = float(np.mean(aris))
#
#     # ────────────────────────────────── 3b) ensemble consistency R  (unchanged)
#     hits = 0
#     # for sd in range(seeds):
#     #     torch.manual_seed(sd)
#     #     e_mdl = _train_mdn(pts, hidden=model["hidden"], iters=model["iters"])
#     #     e_mdl["net"] = e_mdl["net"].float()
#     #     Y = np.asarray(e_mdl["simulate"](pts[0].tolist(), 2000, dt, True))[-500:]
#     #     lab = DBSCAN(eps=0.04 * np.ptp(pts), min_samples=20).fit(Y).labels_
#     #     hits += (len(np.unique(lab)) > 1)
#     # R_s = hits / seeds
#
#     # compute reference dominant cluster ID once
#     ref_dom = np.argmax(np.bincount(ref_lab[ref_lab >= 0]))
#
#     matches = 0
#     for sd in range(seeds):
#         torch.manual_seed(sd)
#         e_mdl = _train_mdn(pts, hidden=model["hidden"], iters=model["iters"])
#         e_mdl["net"] = e_mdl["net"].float()
#         dt_min, dt_max = 0.1, 1.0
#         steps_min, steps_max = 820, 1000
#
#         # compute steps by linear interpolation (dt=0.1→1000, dt=1.0→800)
#         n_steps = int(
#             steps_max
#             + (steps_min - steps_max) * (dt - dt_min) / (dt_max - dt_min)
#         )
#
#         # run simulation with scaled step count
#         Y = np.asarray(
#             e_mdl["simulate"](pts[0].tolist(), n_steps, dt, True)
#         )[-500:]
#         lab = DBSCAN(eps=0.04 * np.ptp(pts), min_samples=20).fit(Y).labels_
#
#         # find this run’s dominant (most‐populated) cluster
#         dom = np.argmax(np.bincount(lab[lab >= 0]))
#         matches += (dom == ref_dom)
#
#     R_s = matches / seeds
#
#
#     # ────────────────────────────────── REAL′
#     REALp = 0.4 * psis_s + 0.4 * rho_s + 0.2 * (0.5 * ari_s + 0.5 * R_s)
#     return REALp, psis_s, rho_s, ari_s, R_s



def evaluate_real(model, pts, dt, seeds=15, window_len=10, mid_tol=6e-4):
    """
    Return REAL′ and its seven normalised components, including new diagnostics and multi-start evaluation:
      1) PSIS-LOO           4) Derivative hold-out
      2) ρ-W₁ density drift 5) Leave-sequence-out PSIS-LOO
      3) bootstrap ARI      6) Ensemble consistency R
                           7) Mid-edge start-ups
    Metrics 3 and 6 now average over three starting points: start, midpoint, and 75% point of the trajectory.
    """
    import numpy as np, torch, warnings, arviz as az, xarray as xr, scipy.stats as ss
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score
    from sklearn.utils import resample

    # Force network to float32
    model["net"] = model["net"].float()

    # Prepare data
    pts_np = np.asarray(pts, dtype=np.float32)
    N = len(pts_np)
    S  = torch.from_numpy(pts_np)
    X  = S[:-1]
    dS = S[1:] - S[:-1]

    # Forward pass for PSIS
    μ1, μ2, σ1, σ2, logitπ = model["net"](X)
    def logli(mu, sig):
        return -0.5 * ((dS - mu) / sig).pow(2).sum(-1) - 3 * torch.log(sig.squeeze(-1))
    lps = torch.logsumexp(torch.stack([
        torch.log(torch.sigmoid(logitπ)) + logli(μ1, σ1),
        torch.log(torch.sigmoid(-logitπ)) + logli(μ2, σ2)
    ]), dim=0).detach().cpu().numpy()

    # 1) Standard PSIS-LOO
    ll_da    = xr.DataArray(np.tile(lps, (4, 1))[None], dims=("chain", "draw", "obs"))
    post_da  = xr.DataArray(np.zeros((1, 4)), dims=("chain", "draw"))
    idata    = az.from_dict(posterior={"dummy": post_da}, log_likelihood={"y": ll_da})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Estimated shape parameter")
        psis_loo = az.loo(idata, pointwise=False).elpd_loo
    psis_s = np.clip((psis_loo / N + 1.0) / 5.0, 0, 1)

    # 2) ρ-W₁ density drift
    box_lo, box_hi = pts_np.min(0), pts_np.max(0)
    sims = []
    for s in np.random.uniform(box_lo, box_hi, size=(800, 3)):
        sims += model["simulate"](s.tolist(), steps=int(10 / dt), dt=dt, sample=True)
    sims = np.asarray(sims)[: N]
    rho_w1 = ss.wasserstein_distance(pts_np.flatten(), sims.flatten())
    scale  = 0.35 * np.linalg.norm(np.ptp(pts_np, axis=0))
    rho_s  = np.clip(1 - rho_w1 / scale, 0, 1)

    # Multi-start indices: start, midpoint, 75%
    idxs = [0, int(0.25 * N), N//2, int(0.75 * N)]

    # 3) Bootstrap ARI averaged over starts
    ari_list = []
    for idx in idxs:
        # reference clustering
        n_steps = int(700 + (600 - 700) * (dt - 0.1) / 0.9)
        ref_Y   = np.asarray(model["simulate"](pts_np[idx].tolist(), n_steps, dt, True))[-400:]
        ref_lab = DBSCAN(eps=0.07 * np.ptp(pts_np), min_samples=20).fit(ref_Y).labels_
        aris = []
        for _ in range(37):
            b_pts = resample(pts_np)
            b_mdl = _train_mdn(b_pts, hidden=model["hidden"], iters=1000, lr=2e-3)
            b_mdl["net"] = b_mdl["net"].float()
            n_steps = int(700 + (600 - 700) * (dt - 0.1) / 0.9)
            b_Y = np.asarray(b_mdl["simulate"](b_pts[idx].tolist(), n_steps, dt, True))[-400:]
            lab = DBSCAN(eps=0.07 * np.ptp(pts_np), min_samples=20).fit(b_Y).labels_
            aris.append(adjusted_rand_score(ref_lab, lab))
        ari_list.append(np.mean(aris))
    ari_s = float(np.mean(ari_list))

    # 4) Ensemble consistency R averaged over starts
    R_list = []
    for idx in idxs:
        matches = 0
        # reference dominant cluster
        n_steps = int(700 + (600 - 700) * (dt - 0.1) / 0.9)
        ref_Y   = np.asarray(model["simulate"](pts_np[idx].tolist(), n_steps, dt, True))[-400:]
        ref_lab = DBSCAN(eps=0.07 * np.ptp(pts_np), min_samples=20).fit(ref_Y).labels_
        ref_dom = np.argmax(np.bincount(ref_lab[ref_lab >= 0]))
        for sd in range(seeds):
            torch.manual_seed(sd)
            e_mdl = _train_mdn(pts_np, hidden=model["hidden"], iters=model["iters"])
            e_mdl["net"] = e_mdl["net"].float()
            n_steps = int(700 + (600 - 700) * (dt - 0.1) / 0.9)
            Y = np.asarray(e_mdl["simulate"](pts_np[idx].tolist(), n_steps, dt, True))[-400:]
            lab = DBSCAN(eps=0.07 * np.ptp(pts_np), min_samples=20).fit(Y).labels_
            dom = np.argmax(np.bincount(lab[lab >= 0]))
            matches += (dom == ref_dom)
        R_list.append(matches / seeds)
    R_s = float(np.mean(R_list))

    # 5) Derivative hold-out
    mu_all = (torch.sigmoid(logitπ).unsqueeze(-1) * μ1 + torch.sigmoid(-logitπ).unsqueeze(-1) * μ2).detach().cpu().numpy()
    idx_hold = np.random.choice(N-1, size=int(0.2 * (N-1)), replace=False)
    hold_rmse = np.sqrt(np.mean((mu_all[idx_hold] - dS.detach().cpu().numpy()[idx_hold])**2))
    # dh_s = np.clip(1 - hold_rmse / (np.linalg.norm(np.ptp(pts_np, axis=0)) * dt), 0, 1)
    dh_s = np.clip(1 - hold_rmse / (2 * np.linalg.norm(np.ptp(pts_np, axis=0)) * dt), 0, 1)
    # direction accuracy unused in composite but available as mu_dir_acc = np.mean(np.einsum('ij,ij->i', mu_all[idx_hold], dS.detach().cpu().numpy()[idx_hold]) > 0)

    # 6) Leave-sequence-out PSIS-LOO
    seq_scores = []
    for start in range(0, N-1 - window_len + 1, window_len):
        mask = np.ones(N-1, bool)
        mask[start:start+window_len] = False
        loo_lps = lps[mask]
        ll_da2   = xr.DataArray(np.tile(loo_lps, (4, 1))[None], dims=("chain", "draw", "obs"))
        idata2   = az.from_dict(posterior={"dummy": post_da}, log_likelihood={"y": ll_da2})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Estimated shape parameter")
            score = az.loo(idata2, pointwise=False).elpd_loo
        seq_scores.append(score)
    seq_norm = np.clip((np.mean(seq_scores) / (N - 1 - window_len) + 1.0) / 5.0, 0, 1)

    # 7) Mid-edge start-ups
    mids = (pts_np[:-1] + pts_np[1:]) / 2
    sims_mid = []
    for m in mids:
        sim = model["simulate"](m.tolist(), 1, dt, True)[1]
        sims_mid.append(sim)
    sims_mid = np.array(sims_mid)
    dists = np.min(np.linalg.norm(sims_mid[:,None,:] - pts_np[None,:,:], axis=-1), axis=1)
    me_s = 1 - np.mean(dists < mid_tol)

    # Composite REAL′
    REALp = (0.3 * psis_s + 0.2 * rho_s + 0.1 * ari_s + 0.1 * R_s
             + 0.1 * dh_s + 0.1 * seq_norm + 0.1 * me_s)

    return REALp, psis_s, rho_s, ari_s, R_s, dh_s, seq_norm, me_s




def train_mdn_capture(
    points,
    hidden_sizes=[20, 64, 96, 150],
    iters=8000,
    noise_temp=1.0,
    smooth_window=200,
    csv_summary_path='mdn_train_summary.csv',
):
    """
    For each hidden size, call the registered mdn trainer (_train_mdn)
    with lr=3e-3 and clip=8.0, capture its per-iteration losses,
    compute summary stats, write CSV, and plot raw+smoothed curves.
    """
#     import numpy as np
    import pandas as pd
#     import matplotlib.pyplot as plt

    summary = []
    loss_histories = {}
    smooth_histories = {}

    for h in hidden_sizes:
        # call your stable trainer with the exact lr & clip defaults
        result = _train_mdn(
            points,
            hidden=h,
            iters=iters,
            lr=3e-3,          # exact as in the @register
            clip=8.0,         # exact as in the @register
            noise_temp=noise_temp,
            return_history=True,
        )

        losses = np.array(result['loss_history'])
        smooth = pd.Series(losses).rolling(window=smooth_window, min_periods=1).mean().to_numpy()

        idx_min = int(losses.argmin())
        stats = {
            'hidden':               h,
            'mean_train_loss':      float(losses.mean()),
            'std_train_loss':       float(losses.std()),
            'final_train_loss':     float(losses[-1]),
            'min_train_loss':       float(losses[idx_min]),
            'iter_min_train_loss':  idx_min + 1,
            'overtrain_metric':     float(losses[-1] - losses[idx_min]),
            'actual_iters_ran':     result['iters'],
        }
        summary.append(stats)
        loss_histories[h] = losses
        smooth_histories[h] = smooth

        if stats['overtrain_metric'] > 0:
            print(f"[h={h}] risk of over-training (Δ = {stats['overtrain_metric']:.4f})")
        else:
            print(f"[h={h}] no obvious over-training")

    # write summary CSV
    df = pd.DataFrame(summary)
    df.to_csv(csv_summary_path, index=False)
    print(f"Saved training summary to {csv_summary_path}")

    # plot raw losses
    plt.figure(figsize=(8,5))
    for h, l in loss_histories.items():
        plt.plot(l, alpha=0.3, label=f'h={h} raw')
        mi = int(np.argmin(l))
        plt.scatter(mi, l[mi], marker='x')
    plt.xlabel('Iteration')
    plt.ylabel('Train Loss')
    plt.title('MDN Training Loss (raw)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot smoothed losses
    plt.figure(figsize=(8,5))
    for h, s in smooth_histories.items():
        plt.plot(s, label=f'h={h}')
    plt.xlabel('Iteration')
    plt.ylabel(f'Loss (rolling mean, window={smooth_window})')
    plt.title('MDN Training Loss (smoothed)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df, loss_histories, smooth_histories


#ORYGINAL
# @register("mdn")
# def _train_mdn(points, hidden=100, iters=8000, lr=3e-3, clip=8.0, noise_temp=1.0):
#     # def _train_mdn(points, hidden=10, iters=3000, lr=3e-3, clip=8.0, noise_temp=1.0):
#     """
#     Much stabler training:
#
#     • σ is forced positive with softplus
#     • log-sum-exp trick for the mixture likelihood
#     • gradient clipping prevents runaway updates
#     """
#     import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
#     import numpy as np
#
#     torch.manual_seed(42)
#
#     S = torch.from_numpy(points.astype(np.float32))
#     X = S[:-1]  # shape (N-1, 3)
#     ΔS = S[1:] - S[:-1]  # target velocities
#
#     class MDN(nn.Module):
#         def __init__(self, h):
#             super().__init__()
#             self.feat = nn.Sequential(
#                 nn.Linear(3, h),
#                 nn.Tanh(),
#                 nn.Linear(h, h),
#                 nn.Tanh()
#             )
#             self.out = nn.Linear(h, 3 + 3 + 2 + 1)  # μ1|μ2|logσ1|logσ2|logitπ
#
#         def forward(self, x):
#             t = self.out(self.feat(x))
#             μ1, μ2 = t[..., :3], t[..., 3:6]
#             # softplus → strictly positive σ, bottom-clamped at 1e-3
#             σ1 = F.softplus(t[..., 6:7]) + 1e-3
#             σ2 = F.softplus(t[..., 7:8]) + 1e-3
#             logitπ = t[..., 8:]
#             return μ1, μ2, σ1, σ2, logitπ.squeeze(-1)
#
#     net = MDN(hidden)
#     opt = optim.Adam(net.parameters(), lr=lr)
#
#     for _ in range(iters):
#         μ1, μ2, σ1, σ2, logitπ = net(X)
#
#         # Mahalanobis /2 + log|Σ|
#         def nll(μ, σ):
#             return 0.5 * ((ΔS - μ) / σ).pow(2).sum(-1) + 3 * torch.log(σ.squeeze(-1))
#
#         logp1 = torch.log(torch.sigmoid(logitπ) + 1e-9) - nll(μ1, σ1)
#         logp2 = torch.log(torch.sigmoid(-logitπ) + 1e-9) - nll(μ2, σ2)
#         # log-sum-exp trick
#         loss = -torch.logsumexp(
#             torch.stack([logp1, logp2], dim=0), dim=0
#         ).mean()
#
#         opt.zero_grad()
#         loss.backward()
#         nn.utils.clip_grad_norm_(net.parameters(), clip)  # ← stops divergent jumps
#         opt.step()
#
#     axis_min, axis_max = points.min(0), points.max(0)
#
#     @torch.no_grad()
#     def simulate(start, steps=2, dt=0.000001, sample=True, noise_temp=noise_temp):
#         s = torch.tensor(start, dtype=torch.float32)
#         out = [s.numpy().tolist()]
#         a_min = torch.from_numpy(axis_min + 0.1).float()
#         a_max = torch.from_numpy(axis_max - 0.1).float()
#         rng = torch.distributions.Normal(torch.zeros(3), torch.ones(3))
#
#         for _ in range(steps):
#             μ1, μ2, σ1, σ2, logitπ = net(s)
#             π = torch.sigmoid(logitπ)
#
#             if sample and torch.rand(1).item() < π.item():
#                 v = μ1 + rng.sample() * σ1 * noise_temp  # ← scaled
#             elif sample:
#                 v = μ2 + rng.sample() * σ2 * noise_temp  # ← scaled
#             else:
#                 v = μ1  # deterministic
#
#             s = s + v * dt
#             s = torch.max(torch.min(s, a_max), a_min)
#             out.append(s.numpy().tolist())
#
#         return out
#
#     # return {"kind": "mdn", "axis_min": axis_min, "axis_max": axis_max, "simulate": simulate}
#     return {
#         "kind": "mdn",
#         "net": net,  # ← NEW: gives evaluate_real access
#         "hidden": hidden,  # ←  optional, but useful for bookkeeping
#         "iters": iters,  # ←  optional
#         "axis_min": axis_min,
#         "axis_max": axis_max,
#         "simulate": simulate
#     }


def compute_patience(h: float) -> float:
    if h <= 150:
        return 480
    elif h >= 200:
        return 300
    else:
        # linearly interpolate from (h=150, p=480) to (h=200, p=300)
        return 480 + (300 - 480) * (h - 150) / (200 - 150)


@register("mdn")
def _train_mdn(
    points,
    hidden=100,
    iters=8000,
    lr=3e-3,
    clip=8.0,
    noise_temp=1.0,
    return_history=False,        # ← new flag
    path_ids=None,               # ← multi-path support
    deterministic=True,          # ← seed all RNGs for reproducible results
    seed=42,                     # ← global seed used when deterministic=True
):
    """
    Much stabler training with:
      • σ forced positive with softplus
      • log-sum-exp trick for mixture likelihood
      • gradient clipping (constant)
      • AdamW + weight‐decay annealing (no LR decay)
      • input jitter + dynamic clipping
      • rolling‐mean early‐stop (window=200, patience=800)
      • π‐logit warmup (smaller init scale)
      • dynamic dropout for big nets (h ≥ 300)
      • optional LBFGS polish (100 steps at lr=0.2)

    Multi-path: when path_ids is provided, velocity pairs that cross
    path boundaries are excluded so multiple independent semantic paths
    can train a unified flow field without spurious boundary velocities.
    """
    import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
#     import numpy as np

    if deterministic:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    S = torch.from_numpy(points.astype(np.float32))

    # Build trajectory pairs, skipping path boundaries when multi-path
    if path_ids is not None:
        pid = np.asarray(path_ids)
        # Valid pair: consecutive points belong to the same path
        valid = pid[:-1] == pid[1:]  # (N-1,) boolean mask
        X = S[:-1][valid]            # only same-path positions
        ΔS = (S[1:] - S[:-1])[valid] # only same-path velocities
    else:
        X = S[:-1]                       # (N-1, 3)
        ΔS = S[1:] - S[:-1]              # target velocities

    # Pre-compute axes for jitter + clamping
    axis_min, axis_max = S.min(0)[0], S.max(0)[0]
    axis_range = axis_max - axis_min
    jitter_frac = 0.01

    class MDN(nn.Module):
        def __init__(self, h):
            super().__init__()
            # dynamic dropout if large net
            # if h >= 300:
            p = min(0.2, 0.1 * (h / 300) ** 1.5)

            self.feat = nn.Sequential(
                nn.Linear(3, h), nn.Tanh(),
                nn.Dropout(p),
                nn.Linear(h, h), nn.Tanh(),
            )
            # else:
            #     self.feat = nn.Sequential(
            #         nn.Linear(3, h), nn.Tanh(),
            #         nn.Linear(h, h), nn.Tanh(),
            #     )
            # outputs: μ1, μ2 (3 each), σ1, σ2 (1 each), logitπ (1)
            self.out = nn.Linear(h, 3*2 + 2*1 + 1)

        def forward(self, x):
            t = self.out(self.feat(x))
            μ1, μ2 = t[..., :3], t[..., 3:6]
            σ1 = F.softplus(t[..., 6:7]) + 1e-3
            σ2 = F.softplus(t[..., 7:8]) + 1e-3
            logitπ = t[..., 8]
            return μ1, μ2, σ1, σ2, logitπ

    net = MDN(hidden)
    # π‐logit warm‐up: shrink initial logit weights & zero bias
    with torch.no_grad():
        init_scale = 0.1
        net.out.weight[8].mul_(init_scale)
        net.out.bias[8].zero_()

    # AdamW + initial weight-decay
    initial_wd = 3e-4
    opt = optim.AdamW(net.parameters(), lr=lr/2, weight_decay=initial_wd)
    # no LR scheduler (we start at lr/2 and keep it constant)

    def nll(μ, σ):
        return 0.5 * ((ΔS - μ) / σ).pow(2).sum(-1) + 3 * torch.log(σ.squeeze(-1))

    # Early-stop via rolling-loss
    window = 200
    patience = compute_patience(hidden)
    roll_losses = []
    best_roll = float('inf')
    wait = 0

    # prepare history collector
    if return_history:
        loss_history = []

    clip_start = 8.0
    clip_end = 2.0  # value reached at the very last step
    decay_rate = np.log(clip_start / clip_end) / iters  # solved so that f(iters)=clip_end

    for i in range(1, iters + 1):
        # Input jitter (~1% of coord-range)
        noise = torch.randn_like(X) * (jitter_frac * axis_range)
        X_noisy = (X + noise).clamp(axis_min, axis_max)

        μ1, μ2, σ1, σ2, logitπ = net(X_noisy)
        logp1 = torch.log(torch.sigmoid(logitπ) + 1e-9) - nll(μ1, σ1)
        logp2 = torch.log(torch.sigmoid(-logitπ) + 1e-9) - nll(μ2, σ2)
        loss = -torch.logsumexp(torch.stack([logp1, logp2], dim=0), dim=0).mean()

        # record for history
        if return_history:
            loss_history.append(loss.item())

        # rolling-mean & early stop
        roll_losses.append(loss.item())
        if len(roll_losses) > window:
            roll_losses.pop(0)
            mean_roll = sum(roll_losses) / window
            if mean_roll < best_roll:
                best_roll, wait = mean_roll, 0
            else:
                wait += 1
            if wait > patience:
                print(f"Early stop at iter={i}")
                break

        opt.zero_grad()
        loss.backward()
        # constant gradient clipping
        clip_val = clip_start * np.exp(-decay_rate * i)
        nn.utils.clip_grad_norm_(net.parameters(), clip_val)
        opt.step()

        # weight-decay annealing (linearly to zero)
        wd = initial_wd * (1 - min(i, iters) / iters)
        opt.param_groups[0]['weight_decay'] = wd

    # # Optional LBFGS polish
    # opt_lbfgs = optim.LBFGS(net.parameters(), lr=0.2, max_iter=100)
    # def closure():
    #     opt_lbfgs.zero_grad()
    #     μ1, μ2, σ1, σ2, logitπ = net(X)
    #     logp1 = torch.log(torch.sigmoid(logitπ) + 1e-9) - nll(μ1, σ1)
    #     logp2 = torch.log(torch.sigmoid(-logitπ) + 1e-9) - nll(μ2, σ2)
    #     loss_lbf = -torch.logsumexp(torch.stack([logp1, logp2], dim=0), dim=0).mean()
    #     loss_lbf.backward()
    #     return loss_lbf
    # opt_lbfgs.step(closure)

    @torch.no_grad()
    def simulate(start, steps=2, dt=1e-6, sample=True, noise_temp=noise_temp):
        s = torch.tensor(start, dtype=torch.float32)
        out = [s.numpy().tolist()]
        a_min = axis_min + 0.1
        a_max = axis_max - 0.1
        rng = torch.distributions.Normal(torch.zeros(3), torch.ones(3))
        for _ in range(steps):
            μ1, μ2, σ1, σ2, logitπ = net(s)
            π = torch.sigmoid(logitπ)
            if sample and torch.rand(1).item() < π.item():
                v = μ1 + rng.sample() * σ1 * noise_temp
            elif sample:
                v = μ2 + rng.sample() * σ2 * noise_temp
            else:
                v = μ1
            s = (s + v * dt).clamp(a_min, a_max)
            out.append(s.numpy().tolist())
        return out

    # assemble return dict
    out = {
        "kind": "mdn",
        "net": net,
        "hidden": hidden,
        "iters": i,
        "axis_min": axis_min,
        "axis_max": axis_max,
        "simulate": simulate
    }
    if return_history:
        out["loss_history"] = loss_history

    return out


# ────────────────────────────────────────────────────────────────────────────
#  RBF flow model — lightweight scipy-based alternative to MDN
# ────────────────────────────────────────────────────────────────────────────

@register("rbf")
def _train_rbf(points, kernel="thin_plate_spline", smoothing=0.1, path_ids=None):
    """RBF (Radial Basis Function) interpolation flow model.

    Uses scipy's RBFInterpolator to fit a smooth velocity field from
    point-to-point displacements. Good for smaller datasets or when
    PyTorch is not available. Produces smoother, more conservative
    flows compared to MDN.

    Args:
        points: (N, 3) array of 3D positions (in sequence order).
        kernel: RBF kernel type. Options: "thin_plate_spline" (default),
                "cubic", "linear", "quintic".
        smoothing: Regularization (0 = exact interpolation, >0 = smooth).
        path_ids: Optional per-point path IDs for multi-path support.
    """
    from scipy.interpolate import RBFInterpolator

    _SUPPORTED_KERNELS = {"linear", "thin_plate_spline", "cubic", "quintic"}
    if kernel not in _SUPPORTED_KERNELS:
        raise ValueError(
            f"RBF kernel '{kernel}' is not supported. "
            f"Supported kernels: {', '.join(sorted(_SUPPORTED_KERNELS))}. "
            f"Kernels like 'multiquadric' and 'gaussian' require an epsilon "
            f"parameter that is not currently exposed."
        )

    N = len(points)

    # Build velocity vectors from sequential displacements
    if path_ids is not None:
        path_ids = np.asarray(path_ids)
        # Only use consecutive pairs within the same path
        mask = np.array([
            path_ids[i] == path_ids[i + 1] for i in range(N - 1)
        ])
        X = points[:-1][mask]
        V = (points[1:] - points[:-1])[mask]
    else:
        X = points[:-1]
        V = points[1:] - points[:-1]

    if len(X) < 3:
        raise ValueError("Need at least 3 valid velocity pairs for RBF")

    # Fit one RBFInterpolator per velocity component
    rbf_models = []
    for d in range(3):
        rbf = RBFInterpolator(X, V[:, d], kernel=kernel, smoothing=smoothing)
        rbf_models.append(rbf)

    # Bounding box with 10% margin
    margin = 0.10
    span = points.max(axis=0) - points.min(axis=0)
    axis_min = (points.min(axis=0) - margin * span).astype(np.float32)
    axis_max = (points.max(axis=0) + margin * span).astype(np.float32)

    def predict_velocity(query_points):
        """Predict velocity at query_points (M, 3) → (M, 3)."""
        qp = np.asarray(query_points, dtype=np.float64)
        if qp.ndim == 1:
            qp = qp.reshape(1, -1)
        vel = np.zeros((len(qp), 3), dtype=np.float32)
        for d in range(3):
            vel[:, d] = rbf_models[d](qp).astype(np.float32)
        return vel

    def simulate(start, steps=32, dt=0.1, sample_jumps=False):
        """Simulate a trajectory from start position."""
        s = np.array(start, dtype=np.float32)
        out = [s.tolist()]
        for _ in range(steps):
            v = predict_velocity(s.reshape(1, -1))[0]
            s = s + v * dt
            s = np.clip(s, axis_min, axis_max)
            out.append(s.tolist())
        return out

    return {
        "kind": "rbf",
        "predict_velocity": predict_velocity,
        "axis_min": axis_min,
        "axis_max": axis_max,
        "simulate": simulate,
    }


# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def visualize_flow(
    mdn_model,
    training_points,
    grid_size=30,
    scatter_alpha=0.3,
    scatter_size=5,
    data_scatter_size=20,
    data_scatter_alpha=0.6,
    n_arrows=2000,
    arrow_length=0.03,
    figsize=(10, 10),
    random_seed=42
):
    """
    Visualize 3D flow magnitude & direction together with training data points.

    Parameters
    ----------
    mdn_model : dict
        Returned by _train_mdn. Must contain:
          - 'net': the trained MDN
          - 'axis_min', 'axis_max': array-like (3,)
    training_points : array-like, shape (N,3)
        The original points the MDN was trained on (loaded from JSON).
    grid_size : int
        Number of samples per axis (total = grid_size^3).
    scatter_alpha : float
        Transparency of the speed‐colored scatter (0 = fully transparent, 1 = opaque).
    scatter_size : float
        Marker size for the flow scatter points.
    data_scatter_size : float
        Marker size for the training data scatter.
    data_scatter_alpha : float
        Transparency for training data points.
    n_arrows : int
        How many arrows to draw (randomly sampled).
    arrow_length : float
        Length of every arrow, in fraction of the axis‐range.
    figsize : tuple
        Figure size in inches.
    random_seed : int
        For arrow sampling reproducibility.
    """
#     import numpy as np
#     import torch
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D

    net = mdn_model['net']
    amin = np.asarray(mdn_model['axis_min'], dtype=float)
    amax = np.asarray(mdn_model['axis_max'], dtype=float)
    span = amax - amin

    # 1) build grid
    xs = np.linspace(amin[0], amax[0], grid_size)
    ys = np.linspace(amin[1], amax[1], grid_size)
    zs = np.linspace(amin[2], amax[2], grid_size)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # 2) get expected flow vectors
    with torch.no_grad():
        t = torch.from_numpy(pts.astype(np.float32))
        mu1, mu2, sigma1, sigma2, logit_pi = net(t)
        p1 = torch.sigmoid(logit_pi).unsqueeze(-1)
        flow = (p1 * mu1 + (1 - p1) * mu2).numpy()

    # 3) speed for coloring
    speed = np.linalg.norm(flow, axis=1)

    # 4) plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 4a) training data points
    tp = np.asarray(training_points, dtype=float)
    ax.scatter(
        tp[:, 0], tp[:, 1], tp[:, 2],
        c='blue', s=data_scatter_size,
        alpha=data_scatter_alpha, label='training points'
    )

    # 4b) flow field scatter colored by speed
    sc = ax.scatter(
        X.ravel(), Y.ravel(), Z.ravel(),
        c=speed, cmap='hot',
        alpha=scatter_alpha, s=scatter_size,
        label='flow speed'
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('Flow speed (‖v‖)')

    # 5) arrows for direction
    rng = np.random.default_rng(random_seed)
    total_pts = pts.shape[0]
    idx = rng.choice(total_pts, size=min(n_arrows, total_pts), replace=False)
    ax.quiver(
        X.ravel()[idx], Y.ravel()[idx], Z.ravel()[idx],
        flow[idx,0], flow[idx,1], flow[idx,2],
        length=arrow_length * np.linalg.norm(span),
        normalize=True, linewidth=0.5,
        color='black'
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Flow & Training Points')
    ax.legend()
    plt.tight_layout()
    plt.show()


# def visualize_flow(
#     mdn_model,
#     grid_size=30,
#     scatter_alpha=0.3,
#     scatter_size=5,
#     n_arrows=2000,
#     arrow_length=0.03,
#     figsize=(10, 10),
#     random_seed=42
# ):
#     """
#     Visualize 3D flow magnitude & direction with better transparency
#     and uniform, fixed-length arrows.
#
#     Parameters
#     ----------
#     mdn_model : dict
#         Returned by _train_mdn. Must contain:
#           - 'net': the trained MDN
#           - 'axis_min', 'axis_max': array-like (3,)
#     grid_size : int
#         Number of samples per axis (total = grid_size^3).
#     scatter_alpha : float
#         Transparency of the speed‐colored scatter (0 = fully transparent,
#         1 = opaque). Try ~0.2–0.4.
#     scatter_size : float
#         Marker size for the scatter points.
#     n_arrows : int
#         How many arrows to draw (randomly sampled).
#     arrow_length : float
#         Length of every arrow, in fraction of the axis‐range.
#     figsize : tuple
#         Figure size in inches.
#     random_seed : int
#         For arrow sampling reproducibility.
#     """
#     net = mdn_model['net']
#     amin = np.asarray(mdn_model['axis_min'], dtype=float)
#     amax = np.asarray(mdn_model['axis_max'], dtype=float)
#     span = amax - amin
#
#     # 1) build grid
#     xs = np.linspace(amin[0], amax[0], grid_size)
#     ys = np.linspace(amin[1], amax[1], grid_size)
#     zs = np.linspace(amin[2], amax[2], grid_size)
#     X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
#     pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
#
#     # 2) get expected flow vectors
#     with torch.no_grad():
#         t = torch.from_numpy(pts.astype(np.float32))
#         μ1, μ2, σ1, σ2, logitπ = net(t)
#         p1 = torch.sigmoid(logitπ).unsqueeze(-1)
#         flow = (p1 * μ1 + (1 - p1) * μ2).numpy()
#
#     # 3) speed for coloring
#     speed = np.linalg.norm(flow, axis=1)
#
#     # 4) plot
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111, projection='3d')
#
#     # scatter colored by speed, small & semi‐transparent
#     sc = ax.scatter(
#         X.ravel(), Y.ravel(), Z.ravel(),
#         c=speed, cmap='hot',
#         alpha=scatter_alpha, s=scatter_size
#     )
#     cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
#     cbar.set_label('Flow speed (‖v‖)')
#
#     # 5) pick random subset of points for arrows
#     rng = np.random.default_rng(random_seed)
#     total_pts = pts.shape[0]
#     if n_arrows < total_pts:
#         idx = rng.choice(total_pts, size=n_arrows, replace=False)
#     else:
#         idx = np.arange(total_pts)
#
#     # arrows all same fixed length, normalized direction
#     ax.quiver(
#         X.ravel()[idx], Y.ravel()[idx], Z.ravel()[idx],
#         flow[idx,0], flow[idx,1], flow[idx,2],
#         length=arrow_length * np.linalg.norm(span),
#         normalize=True,
#         linewidth=0.5
#     )
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('3D Flow: Speed Heatmap + Uniform‐length Direction Arrows')
#     plt.tight_layout()
#     plt.show()



# import numpy as np
# import torch, torch.nn.functional as F
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – makes 3‑D scatter work
#
#
# def generate_flow_animation(
#     net,
#     axis_min,
#     axis_max,
#     *,
#     grid: int = 25,
#     dt: float = 1e-4,
#     steps: int = 300,
#     sample: bool = True,
#     noise_temp: float = 1.0,
#     margin: float = 0.05,
#     save_path: str | None = None,
# ):
#     """Animate a dense 3‑D flow field driven by a trained MDN.
#
#     Parameters
#     ----------
#     net : torch.nn.Module
#         The MDN network returned from ``_train_mdn``.
#     axis_min, axis_max : array‑like, shape (3,)
#         Bounding box of the training data; points are wrapped inside.
#     grid : int, default 25
#         Number of lattice samples per axis → grid³ total points.
#     dt : float, default 1e‑4
#         Euler integration step.
#     steps : int, default 300
#         Animation frames to generate.
#     sample : bool, default True
#         If *True* sample from the mixture; otherwise use the first mean μ₁.
#     noise_temp : float, default 1.0
#         Temperature multiplier for the sampled Gaussian noise.
#     margin : float, default 0.05
#         Padding from the box walls to avoid numerical clipping.
#     save_path : str | None
#         If set, the animation is encoded with ffmpeg (mp4) or imagemagick (gif)
#         depending on the extension.
#
#     Returns
#     -------
#     matplotlib.animation.FuncAnimation
#     """
#
#     # ── build a regular lattice of points ─────────────────────────────────────
#     xs = np.linspace(axis_min[0] + margin, axis_max[0] - margin, grid)
#     ys = np.linspace(axis_min[1] + margin, axis_max[1] - margin, grid)
#     zs = np.linspace(axis_min[2] + margin, axis_max[2] - margin, grid)
#     xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
#     P = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
#     P_torch = torch.from_numpy(P)
#
#     # setup figure ----------------------------------------------------------------
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.set(
#         xlim=(axis_min[0], axis_max[0]),
#         ylim=(axis_min[1], axis_max[1]),
#         zlim=(axis_min[2], axis_max[2]),
#         xlabel="X",
#         ylabel="Y",
#         zlabel="Z",
#         title="MDN flow field",
#     )
#
#     # initial colours by deterministic velocity ------------------------------------
#     with torch.no_grad():
#         mu1, mu2, sigma1, sigma2, logit_pi = net(P_torch)
#         probs = torch.sigmoid(logit_pi)
#         V = torch.where(probs.unsqueeze(-1) > 0.5, mu1, mu2)
#         speed = torch.linalg.norm(V, dim=-1).numpy()
#
#     scat = ax.scatter(
#         P[:, 0],
#         P[:, 1],
#         P[:, 2],
#         s=1,
#         c=speed,
#         cmap="turbo",
#         vmin=speed.min(),
#         vmax=speed.max(),
#     )
#     cbar = fig.colorbar(scat, ax=ax, pad=0.1, shrink=0.7)
#     cbar.set_label("speed ‖v‖")
#
#     # ── animation callback ────────────────────────────────────────────────────
#     def step(frame):
#         nonlocal P, P_torch
#
#         with torch.no_grad():
#             mu1, mu2, sigma1, sigma2, logit_pi = net(P_torch)
#
#             if sample:
#                 rng = torch.randn_like(mu1)
#                 probs = torch.sigmoid(logit_pi)
#                 choose_mu1 = (torch.rand_like(probs) < probs).unsqueeze(-1)
#                 V = choose_mu1 * (mu1 + rng * sigma1 * noise_temp) + (~choose_mu1) * (
#                     mu2 + rng * sigma2 * noise_temp
#                 )
#             else:
#                 V = mu1  # deterministic flow
#
#             speed = torch.linalg.norm(V, dim=-1).numpy()
#
#         # Euler integration
#         P += V.numpy() * dt
#
#         # wrap around to keep points inside the box (makes the loop seamless)
#         for i in range(3):
#             span = axis_max[i] - axis_min[i] - 2 * margin
#             P[:, i] = axis_min[i] + margin + np.mod(P[:, i] - (axis_min[i] + margin), span)
#
#         P_torch = torch.from_numpy(P)
#
#         # update scatter artists
#         scat._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
#         scat.set_array(speed)
#         return (scat,)
#
#     ani = animation.FuncAnimation(
#         fig, step, frames=steps, interval=30, blit=False, repeat=True
#     )
#
#     if save_path is not None:
#         ani.save(save_path, fps=30, bitrate=1800)
#     return ani


# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – enables 3-D scatter


# def generate_flow_animation(
#     net,
#     axis_min,
#     axis_max,
#     *,
#     grid: int = 25,
#     dt: float = 1e-4,
#     steps: int = 300,
#     lifespan: int = 32,
#     sample: bool = True,
#     noise_temp: float = 1.0,
#     margin: float = 0.05,
#     alpha: float = 0.2,
#     save_path: str | None = None,
# ):
#     """Visualise an MDN-driven 3-D flow as a looping particle field.
#
#     Each lattice point becomes a *short-lived* particle: after *lifespan* frames it
#     is reset to its original position, so the flow stays full even when most
#     velocities converge on attractors.
#
#     Parameters
#     ----------
#     net        : torch.nn.Module – trained MDN (from ``_train_mdn``)
#     axis_min/
#     axis_max   : (3,) array-likes – data bounding box used for wrapping
#     grid       : int  – lattice resolution per axis → grid³ particles
#     dt         : float – Euler integration step
#     steps      : int  – total animation frames
#     lifespan   : int  – frames before a particle respawns
#     sample     : bool – sample mixture (True) or use μ₁ deterministically
#     noise_temp : float – scale of injected Gaussian noise when sampling
#     margin     : float – padding from the walls to avoid clipping artefacts
#     alpha      : float – particle transparency (0 opaque → 1 invisible)
#     save_path  : str  – if given, write **.mp4** / **.gif** via ffmpeg / ImageMagick
#     """
#
#     # ── build regular lattice ──────────────────────────────────────────────────
#     xs = np.linspace(axis_min[0] + margin, axis_max[0] - margin, grid)
#     ys = np.linspace(axis_min[1] + margin, axis_max[1] - margin, grid)
#     zs = np.linspace(axis_min[2] + margin, axis_max[2] - margin, grid)
#     xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
#     P = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
#
#     P_init = P.copy()  # where particles respawn
#     age = np.random.randint(0, lifespan, size=len(P))  # stagger resets
#     P_torch = torch.from_numpy(P)
#
#     # ── figure & initial scatter ───────────────────────────────────────────────
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.set(
#         xlim=(axis_min[0], axis_max[0]),
#         ylim=(axis_min[1], axis_max[1]),
#         zlim=(axis_min[2], axis_max[2]),
#         xlabel="X",
#         ylabel="Y",
#         zlabel="Z",
#         title="MDN flow field (short-lived particles)",
#     )
#
#     with torch.no_grad():
#         mu1, mu2, _, _, logit_pi = net(P_torch)
#         V0 = torch.where(torch.sigmoid(logit_pi).unsqueeze(-1) > 0.5, mu1, mu2)
#         speed0 = torch.linalg.norm(V0, dim=-1).numpy()
#
#     scat = ax.scatter(
#         P[:, 0],
#         P[:, 1],
#         P[:, 2],
#         s=0.5,
#         c=speed0,
#         cmap="turbo",
#         vmin=speed0.min(),
#         vmax=speed0.max(),
#         alpha=alpha,
#     )
#     cbar = fig.colorbar(scat, ax=ax, pad=0.1, shrink=0.7)
#     cbar.set_label("‖v‖ speed")
#
#     # ── animation callback ────────────────────────────────────────────────────
#     def step(frame):
#         nonlocal P, P_torch, age
#
#         with torch.no_grad():
#             mu1, mu2, sigma1, sigma2, logit_pi = net(P_torch)
#
#             if sample:
#                 rng = torch.randn_like(mu1)
#                 probs = torch.sigmoid(logit_pi)
#                 choose_mu1 = (torch.rand_like(probs) < probs).unsqueeze(-1)
#                 V = choose_mu1 * (mu1 + rng * sigma1 * noise_temp) + (~choose_mu1) * (
#                     mu2 + rng * sigma2 * noise_temp
#                 )
#             else:
#                 V = mu1
#
#             speed = torch.linalg.norm(V, dim=-1).numpy()
#
#         # integrate positions & ages ------------------------------------------------
#         P += V.numpy() * dt
#         age += 1
#
#         # wrap inside padded box ----------------------------------------------------
#         for i in range(3):
#             span = axis_max[i] - axis_min[i] - 2 * margin
#             P[:, i] = axis_min[i] + margin + np.mod(P[:, i] - (axis_min[i] + margin), span)
#
#         # respawn expired particles -------------------------------------------------
#         mask = age >= lifespan
#         if mask.any():
#             P[mask] = P_init[mask]
#             age[mask] = 0
#
#         P_torch = torch.from_numpy(P)
#
#         # update scatter plot -------------------------------------------------------
#         scat._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
#         scat.set_array(speed)
#         return (scat,)
#
#     ani = animation.FuncAnimation(
#         fig, step, frames=steps, interval=30, blit=False, repeat=True
#     )
#
#     if save_path:
#         ani.save(save_path, fps=30, bitrate=1800)
#     return ani


# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from matplotlib import animation



def generate_flow_animation(
    net,
    axis_min,
    axis_max,
    *,
    training_points=None,
    grid: int = 25,
    dt: float = 1e-4,
    steps: int = 200,
    lifespan: int = 38,
    sample: bool = True,
    noise_temp: float = 1.0,
    margin: float = 0.05,
    alpha: float = 0.16,
    save_path: str | None = None,
):
    """Visualise an MDN-driven 3-D flow as a looping particle field, overlay training points, and save to a video file.
    The view rotates over time. Training points are shown in vivid red, fully opaque.
    """
    # Convert axis bounds from torch.Tensor to numpy
    amin = np.asarray(axis_min, dtype=np.float32)
    amax = np.asarray(axis_max, dtype=np.float32)

    # build regular lattice of particles
    xs = np.linspace(amin[0] + margin, amax[0] - margin, grid)
    ys = np.linspace(amin[1] + margin, amax[1] - margin, grid)
    zs = np.linspace(amin[2] + margin, amax[2] - margin, grid)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

    P_init = P.copy()  # for respawning
    age = np.random.randint(0, lifespan, size=len(P))
    P_torch = torch.from_numpy(P)

    # set up figure without display
    plt.ioff()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set(
        xlim=(amin[0], amax[0]),
        ylim=(amin[1], amax[1]),
        zlim=(amin[2], amax[2]),
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        title="Rotating MDN flow with training cloud",
    )

    # overlay original training points in vivid red
    if training_points is not None:
        tp = np.asarray(training_points, dtype=np.float32)
        ax.scatter(
            tp[:, 0], tp[:, 1], tp[:, 2],
            c='#ff0000', s=8, alpha=1.0, label='Training points'
        )

    # initial colours by deterministic flow
    with torch.no_grad():
        mu1, mu2, _, _, logit_pi = net(P_torch)
        V0 = torch.where(torch.sigmoid(logit_pi).unsqueeze(-1) > 0.5, mu1, mu2)
        speed0 = torch.linalg.norm(V0, dim=-1).numpy()

    scat = ax.scatter(
        P[:, 0], P[:, 1], P[:, 2],
        s=0.5,
        c=speed0,
        cmap="turbo",
        vmin=speed0.min(),
        vmax=speed0.max(),
        alpha=alpha,
    )
    cbar = fig.colorbar(scat, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("‖v‖ speed")

    def step(frame):
        nonlocal P, P_torch, age
        with torch.no_grad():
            mu1, mu2, sigma1, sigma2, logit_pi = net(P_torch)
            if sample:
                rng = torch.randn_like(mu1)
                probs = torch.sigmoid(logit_pi)
                choose_mu1 = (torch.rand_like(probs) < probs).unsqueeze(-1)
                V = choose_mu1 * (mu1 + rng * sigma1 * noise_temp) + (
                    ~choose_mu1
                ) * (mu2 + rng * sigma2 * noise_temp)
            else:
                V = mu1
            speed = torch.linalg.norm(V, dim=-1).numpy()

        # Euler integration
        P += V.numpy() * dt
        age += 1
        # wrap inside padded box
        span = amax - amin - 2 * margin
        for i in range(3):
            P[:, i] = amin[i] + margin + np.mod(P[:, i] - (amin[i] + margin), span[i])
        # respawn expired particles
        mask = age >= lifespan
        if mask.any():
            P[mask] = P_init[mask]
            age[mask] = 0
        P_torch = torch.from_numpy(P)

        # update scatter positions and colours
        scat._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
        scat.set_array(speed)

        # rotate the view around the z-axis
        azim = (frame / steps) * 360
        ax.view_init(elev=30, azim=azim)

        return (scat,)

    ani = animation.FuncAnimation(
        fig, step, frames=steps, interval=30, blit=False, repeat=True
    )

    if save_path:
        writer = animation.FFMpegWriter(fps=30, bitrate=1800)
        ani.save(save_path, writer=writer)
    plt.close(fig)
    return ani



# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from matplotlib import animation
from scipy.interpolate import RegularGridInterpolator


# ──────────────────────────────────────────────────────────────
#  Debug visualiser – ONLY executed on PC
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json, os, sys
    from pathlib import Path

    p = argparse.ArgumentParser(description="Offline-debug visualiser")
    p.add_argument("dump", help="*.json produced by Android")
    p.add_argument("--mode", choices=_MODE_FNS, default="gpssm",
                   help="model to train (default: gpssm)")
    p.add_argument("--start", metavar=("X","Y","Z"), nargs=3, type=float,
                   help="initial coord, default = cloud centroid")
    # p.add_argument("--steps", type=int, default=120000)
    # p.add_argument("--dt",    type=float, default=1.0)
    # p.add_argument("--steps", type=int, default=2000)
    # p.add_argument("--dt",    type=float, default=0.2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--dt",    type=float, default=0.1)
    args = p.parse_args()

    with open(args.dump, "r", encoding="utf-8") as f:
        dump = json.load(f)

    # print(f"Loaded {len(dump['points'])} points – training {_MODE_FNS[args.mode].__name__[7:]} …")
    # train_models(json.dumps(dump["points"]), args.mode)

    start = (args.start if args.start else
             (np.min(dump["points"],0)+np.max(dump["points"],0))/2)

    # path = json.loads(simulate_path(args.mode, *start,
    #                                 steps=args.steps, dt=args.dt,
    #                                 sample_jumps=True))

    # heavy imports NOW – never on Android
#     import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D           # noqa: F401

    pts = np.asarray(dump["points"])
    # path = np.asarray(path)

    print(pts[0])

    with open(args.dump, "r") as f:
        dump = json.load(f)
    points = np.asarray(dump["points"])

    # train MDN


    # # instead of building + smoothing a path, just visualise the vector field:
    # visualize_flow(
    #     mdn_model=mdl,
    #     grid_size=100,
    #     scatter_alpha=0.03,  # keep your preferred transparency
    #     scatter_size=4,  # your preferred point size
    #     n_arrows=11000,  # how many arrows you want
    #     arrow_length=0.025  # fixed arrow length (fraction of axis-range)
    # )

    # train_mdn_capture(points,
    #                       hidden_sizes=[48,64,80,96,112,144,164,200,300],
    #                       iters=17000)


    # mdl = _train_mdn(pts, hidden=20, iters=8000, lr=3e-3)
    #
    # net = mdl["net"]
    # axis_min = mdl["axis_min"]
    # axis_max = mdl["axis_max"]
    #
    #
    # ani = generate_flow_animation(
    #     net,
    #     axis_min,
    #     axis_max,
    #     grid=40,
    #     dt=args.dt,
    #     steps=args.steps,
    #     sample=True,
    #     save_path="output_20.mp4",
    # )
    #
    #
    #
    #
    # mdl = _train_mdn(pts, hidden=64, iters=8000, lr=3e-3)
    #
    # net = mdl["net"]
    # axis_min = mdl["axis_min"]
    # axis_max = mdl["axis_max"]
    #
    # ani = generate_flow_animation(
    #     net,
    #     axis_min,
    #     axis_max,
    #     grid=40,
    #     dt=args.dt,
    #     steps=args.steps,
    #     sample=True,
    #     save_path="output_64.mp4",
    # )



    mdl = _train_mdn(pts, hidden=100, iters=20000, lr=3e-3)

    net = mdl["net"]
    axis_min = mdl["axis_min"]
    axis_max = mdl["axis_max"]

    ani = generate_flow_animation(
        net,
        axis_min,
        axis_max,
        training_points=points,
        grid=40,
        dt=args.dt,
        steps=args.steps,
        sample=True,
        save_path="aligment_test.mp4",
    )





    # mdl = _train_mdn(pts, hidden=150, iters=8000, lr=3e-3)
    #
    # net = mdl["net"]
    # axis_min = mdl["axis_min"]
    # axis_max = mdl["axis_max"]
    #
    # ani = generate_flow_animation(
    #     net,
    #     axis_min,
    #     axis_max,
    #     grid=40,
    #     dt=args.dt,
    #     steps=args.steps,
    #     sample=True,
    #     save_path="output_150.mp4",
    # )
    #
    #
    # mdl = _train_mdn(pts, hidden=200, iters=8000, lr=3e-3)
    #
    # net = mdl["net"]
    # axis_min = mdl["axis_min"]
    # axis_max = mdl["axis_max"]
    #
    # ani = generate_flow_animation(
    #     net,
    #     axis_min,
    #     axis_max,
    #     grid=40,
    #     dt=args.dt,
    #     steps=args.steps,
    #     sample=True,
    #     save_path="output_200.mp4",
    # )

    #h= 64, it=8000, dt=0.1  →  REAL′= 0.8336
    #h= 16, it=3000, dt=0.25  →  REAL′= 0.8679


#     import numpy as np


    import pandas as pd
    out_csv = 'mdn_grid_search_results_3.csv'

    # --- define columns in the order you want them in the CSV ---
    metric_names = ['REALp', 'loo', 'w1', 'ari', 'R', 'dh', 'seq_norm', 'me']
    param_names = ['dt', 'hidden', 'iters']
    columns = metric_names + param_names
    # grids = dict(
    #     dt=[0.1, 0.2, 0.3],
    #     hidden=[20, 64, 96, 150],
    #     iters=[3000, 8200],
    # )

    grids = dict(
        dt=[0.10],
        hidden=[400],
        iters=[17000],
    )


    # --- initialize file (overwrite if exists) ---
    with open(out_csv, 'w', newline='') as f:
        pd.DataFrame(columns=columns).to_csv(f, index=False)

    results = []

    # --- run your grid search ---
    for dt_, h_, it_ in itertools.product(*grids.values()):
        mdl = _train_mdn(np.asarray(dump["points"]), hidden=h_, iters=it_)
        mdl.update({'hidden': h_, 'iters': it_})  # remember hp

        # compute all metrics at once
        REALp, loo, w1, ari, R, dh, seq_norm, me = evaluate_real(
            mdl, np.asarray(dump["points"]), dt_
        )
        print(f"h={h_:>3}, it={it_:>5}, dt={dt_:>.3g}  →  REAL′={REALp:7.4f}")

        # build one row as a dict
        row = {
            'REALp': REALp,
            'loo': loo,
            'w1': w1,
            'ari': ari,
            'R': R,
            'dh': dh,
            'seq_norm': seq_norm,
            'me': me,
            'dt': dt_,
            'hidden': h_,
            'iters': it_,
        }

        # append to CSV immediately
        with open(out_csv, 'a', newline='') as f:
            pd.DataFrame([row]).to_csv(f, header=False, index=False)

        # keep for in-memory selection / plotting
        results.append((REALp, dt_, h_, it_, mdl))

    # --- pick best model and plot ---
    BEST_REALp, best_dt, best_h, best_it, best_mdl = max(results, key=lambda x: x[0])
    print(f"\nBEST  hidden={best_h}, iters={best_it}, dt={best_dt}, REAL′={BEST_REALp:.4f}")

    # optional elbow plot on hidden vs REAL'
    hiddens, scores = zip(*[(h, REALp) for REALp, _, h, _, _ in results])
    plt.figure()
    plt.plot(hiddens, scores, '.-')
    plt.xlabel('hidden size')
    plt.ylabel("REAL′ score")
    plt.show()

    # visualise the path of the best model
    start = np.mean(dump["points"], axis=0)
    path = best_mdl['simulate'](start.tolist(), steps=args.steps, dt=best_dt, sample=True)
    path = np.asarray(path)





    if args.mode == "mdn":
        def tj(p0, p1, alpha=0.5):
            """Centripetal parameterization: returns the “knot” spacing between p0→p1."""
            return ((np.linalg.norm(p1 - p0)) ** alpha)


        def catmull_rom_spline(P, n_points=20, alpha=0.5):
            """
            Given 4 control points P = [P0,P1,P2,P3] (shape (4,dim)),
            return n_points samples along the centripetal Catmull–Rom curve between P1→P2.
            """
            # compute knot vector
            t0 = 0.0
            t1 = t0 + tj(P[0], P[1], alpha)
            t2 = t1 + tj(P[1], P[2], alpha)
            t3 = t2 + tj(P[2], P[3], alpha)

            # function to interpolate between Pi and Pi+1
            def interp(p_i, p_j, ti, tj_, t):
                return ((tj_ - t) / (tj_ - ti)) * p_i + ((t - ti) / (tj_ - ti)) * p_j

            # sample parameters between t1 and t2
            ts = np.linspace(t1, t2, n_points)
            out = []
            for t in ts:
                # first level
                A1 = interp(P[0], P[1], t0, t1, t)
                A2 = interp(P[1], P[2], t1, t2, t)
                A3 = interp(P[2], P[3], t2, t3, t)
                # second level
                B1 = interp(A1, A2, t0, t2, t)
                B2 = interp(A2, A3, t1, t3, t)
                # third level (on-curve)
                C = interp(B1, B2, t1, t2, t)
                out.append(C)
            return np.array(out)


        # ─── build a global smoothed path by sliding a 5-sample window ─────────────
        window = 5  # number of raw samples per spline pass
        interp_per_segment = 10  # how many points to generate between each P2→P3

        smoothed = []
        for i in range(len(path) - window + 1):
            block = path[i: i + window]  # shape (5,3)
            # we’ll do two Catmull–Rom segments here: one for points [0..3] and one for [1..4]
            for seg_start in (0, 1):
                ctrl = block[seg_start: seg_start + 4]  # four control points
                pts = catmull_rom_spline(ctrl, n_points=interp_per_segment, alpha=0.5)
                smoothed.append(pts)
        # stitch and (optionally) remove duplicates at joins
        smoothed_path = np.vstack(smoothed)
        path = smoothed_path
    # ───────────────────────────────────────────────────────────────────────────

    print(f"Raw points: {len(path)} → Smoothed spline points: {len(path)}")
    # ─────────────────────────────────────────────────────────────────────────

    # print(path)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               s=2, alpha=0.25, label="cloud")
    ax.plot(path[:, 0], path[:, 1], path[:, 2],
            linewidth=1.0, alpha=1.0, label="smoothed path")
    ax.set_xlabel("X");
    ax.set_ylabel("Y");
    ax.set_zlabel("Z")
    ax.set_title(f"{args.mode} – {args.steps} steps, dt={args.dt}")
    ax.legend()
    plt.tight_layout();
    plt.show()

    last_pt = path[-1]
    print("Last point of path:", last_pt)