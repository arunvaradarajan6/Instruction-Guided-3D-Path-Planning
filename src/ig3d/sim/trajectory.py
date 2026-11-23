"""Planned voxel path -> a reference a quadrotor can fly.

1. Cost-aware shortcutting. A shortcut is kept only if it is no more expensive under the
   PLANNING field (the one the planner used, i.e. the predicted field) and does not bring
   the path closer to obstacles than the drone's radius plus a margin (or than the
   original path already was). The instruction's preference survives; the 26-connected
   zig-zag does not.
2. Corner smoothing (Chaikin), with the same clearance and cost checks.
3. Time parameterisation along the curve: horizontal speed limit, separate climb and
   descent rate limits, lateral acceleration limit (v^2 * curvature), and a tangential
   acceleration limit, from rest to rest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from ..costfield import polyline_cost, segment_costs
from .geometry import Clearance, vox2enu

PHASES = ("takeoff", "hover", "cruise", "goal_hover")


@dataclass
class Limits:
    """Defaults are conservative for a PX4 x500 (MPC_* defaults are higher)."""
    v_xy: float = 3.0
    vz_up: float = 1.5
    vz_dn: float = 1.0
    a_tan: float = 1.5
    a_lat: float = 2.0
    robot_radius: float = 0.38
    margin: float = 0.07
    rate_hz: float = 20.0

    @property
    def r_safe(self) -> float:
        return self.robot_radius + self.margin

    def as_dict(self):
        return asdict(self)


def shortcut(path, field, clr: Clearance, r_safe: float, tol: float = 0.0):
    """Greedy farthest-first shortcutting of a voxel-coordinate path."""
    P = np.asarray(path, float)
    N = len(P)
    if N < 3:
        return P
    seg = segment_costs(field, P[:-1], P[1:], 1.0)[0]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    seg_clear = clr.along(P[:-1], P[1:])
    keep, i = [0], 0
    while i < N - 1:
        j = i + 1
        js = np.arange(N - 1, i + 1, -1)
        if js.size:
            A = np.repeat(P[i][None], js.size, axis=0)
            fwd = segment_costs(field, A, P[js], 1.0)[0]
            ok = fwd <= (1.0 + tol) * (cum[js] - cum[i]) + 1e-9
            if ok.any():
                orig = np.minimum.accumulate(seg_clear[i:])[js - 1 - i]
                need = np.minimum(r_safe, orig) - 1e-6
                cand = np.flatnonzero(ok)
                c = clr.along(A[cand], P[js[cand]])
                good = cand[c >= need[cand]]
                if good.size:
                    j = int(js[good[0]])
        keep.append(j)
        i = j
    return P[keep]


def subdivide(P, max_len: float):
    P = np.asarray(P, float)
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / max_len)))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        out.append(a * (1 - t) + b * t)
    return np.concatenate(out)


def chaikin(P, iters: int):
    P = np.asarray(P, float)
    for _ in range(iters):
        if len(P) < 3:
            break
        a, b = P[:-1], P[1:]
        q, r = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
        mid = np.stack([q, r], axis=1).reshape(-1, 3)[1:-1]
        P = np.concatenate([P[:1], mid, P[-1:]])
    return P


def smooth(P, field, clr: Clearance, r_safe: float, tol: float = 0.02,
           max_len: float = 3.0, iters: int = 4):
    """(curve, iterations used). Falls back to fewer iterations until every check passes."""
    base = subdivide(P, max_len)
    need = min(r_safe, clr.polyline(P)) - 1e-6
    c0 = polyline_cost(field, P)
    for it in range(iters, 0, -1):
        Q = chaikin(base, it)
        if clr.polyline(Q) >= need and polyline_cost(field, Q) <= (1 + tol) * c0 + 1e-9:
            return Q, it
    return base, 0


def _resample(X, ds):
    d = np.linalg.norm(np.diff(X, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < ds:
        return X[[0, -1]], np.array([0.0, s[-1]])
    sq = np.linspace(0.0, s[-1], int(np.ceil(s[-1] / ds)) + 1)
    return np.stack([np.interp(sq, s, X[:, k]) for k in range(3)], axis=1), sq


def time_parameterise(X_enu, lim: Limits, ds: float = 0.05):
    """(t, X, speed, tangent) along a curve in ENU metres, rest to rest."""
    X, s = _resample(np.asarray(X_enu, float), ds)
    if len(X) < 3:
        L = s[-1]
        T = np.repeat(((X[-1] - X[0]) / max(L, 1e-9))[None], len(X), axis=0)
        v = np.zeros(len(X))
        t = np.array([0.0, 2 * np.sqrt(L / max(lim.a_tan, 1e-6))])[:len(X)]
        return t, X, v, T
    h = np.diff(s).mean()
    T = np.gradient(X, h, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    kappa = np.linalg.norm(np.gradient(T, h, axis=0), axis=1)
    kappa = maximum_filter1d(kappa, size=max(3, int(0.5 / h)))
    th, tz = np.hypot(T[:, 0], T[:, 1]), T[:, 2]
    cap = np.full(len(X), np.inf)
    cap = np.minimum(cap, lim.v_xy / np.maximum(th, 1e-6))
    up, dn = tz > 1e-6, tz < -1e-6
    cap[up] = np.minimum(cap[up], lim.vz_up / tz[up])
    cap[dn] = np.minimum(cap[dn], lim.vz_dn / -tz[dn])
    cap = np.minimum(cap, np.sqrt(lim.a_lat / np.maximum(kappa, 1e-9)))
    v = cap.copy()
    v[0] = v[-1] = 0.0
    step = np.diff(s)
    for i in range(len(v) - 1):
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * lim.a_tan * step[i]))
    for i in range(len(v) - 1, 0, -1):
        v[i - 1] = min(v[i - 1], np.sqrt(v[i] ** 2 + 2 * lim.a_tan * step[i - 1]))
    dt = 2 * step / np.maximum(v[:-1] + v[1:], 1e-6)
    return np.concatenate([[0.0], np.cumsum(dt)]), X, v, T


def _sample(t, X, v, T, rate):
    tq = np.arange(0.0, t[-1] + 1e-9, 1.0 / rate)
    if tq[-1] < t[-1]:
        tq = np.append(tq, t[-1])
    pos = np.stack([np.interp(tq, t, X[:, k]) for k in range(3)], axis=1)
    sp = np.interp(tq, t, v)
    tan = np.stack([np.interp(tq, t, T[:, k]) for k in range(3)], axis=1)
    vel = tan * sp[:, None]
    acc = np.gradient(vel, tq, axis=0) if len(tq) > 2 else np.zeros_like(vel)
    return tq, pos, vel, acc


def build_reference(cruise_enu, spawn_enu, lim: Limits, hover_s=2.0, goal_hover_s=3.0):
    """Full reference: vertical takeoff from the spawn pose, hover, cruise, hover at goal."""
    rate = lim.rate_hz
    parts = []
    start = np.asarray(cruise_enu[0], float)
    lift = np.stack([np.asarray(spawn_enu, float), start])
    parts.append((0, _sample(*time_parameterise(lift, lim), rate)))
    n_h = int(round(hover_s * rate))
    parts.append((1, (np.arange(n_h) / rate, np.repeat(start[None], n_h, 0),
                      np.zeros((n_h, 3)), np.zeros((n_h, 3)))))
    parts.append((2, _sample(*time_parameterise(cruise_enu, lim), rate)))
    goal = np.asarray(cruise_enu[-1], float)
    n_g = int(round(goal_hover_s * rate))
    parts.append((3, (np.arange(n_g) / rate, np.repeat(goal[None], n_g, 0),
                      np.zeros((n_g, 3)), np.zeros((n_g, 3)))))
    T, P, V, A, PH = [], [], [], [], []
    t0 = 0.0
    for ph, (t, p, v, a) in parts:
        if len(t) == 0:
            continue
        T.append(t + t0)
        P.append(p)
        V.append(v)
        A.append(a)
        PH.append(np.full(len(t), ph, np.int8))
        t0 = T[-1][-1] + 1.0 / rate
    return dict(t=np.concatenate(T), pos=np.concatenate(P), vel=np.concatenate(V),
                acc=np.concatenate(A), phase=np.concatenate(PH))


def feasibility(ref, cruise_vox, clr: Clearance, lim: Limits) -> dict:
    """What the reference asks of the vehicle, and how close it passes to obstacles."""
    m = ref["phase"] == 2
    v, a = ref["vel"][m], ref["acc"][m]
    return dict(
        duration_s=float(ref["t"][m][-1] - ref["t"][m][0]) if m.any() else 0.0,
        max_speed=float(np.linalg.norm(v, axis=1).max()) if m.any() else 0.0,
        max_climb_rate=float(max(0.0, v[:, 2].max())) if m.any() else 0.0,
        max_descent_rate=float(max(0.0, -v[:, 2].min())) if m.any() else 0.0,
        max_accel=float(np.linalg.norm(a, axis=1).max()) if m.any() else 0.0,
        min_clearance_m=clr.polyline(cruise_vox),
        r_safe_m=lim.r_safe,
        clearance_ok=bool(clr.polyline(cruise_vox) >= lim.robot_radius),
    )


def make_flyable(path_vox, plan_field, occ, voxel_m, lim: Limits):
    """(shortcut, smooth, cruise ENU, info) for one planned voxel path."""
    clr = Clearance(occ, voxel_m)
    sc = shortcut(path_vox, plan_field, clr, lim.r_safe)
    # corner cuts of at most ~3 m (or 1.5 voxels on coarse grids) keep smoothing local
    sm, it = smooth(sc, plan_field, clr, lim.r_safe, max_len=max(1.5, 3.0 / voxel_m))
    return sc, sm, vox2enu(sm, voxel_m), dict(chaikin_iters=it, n_grid=len(path_vox),
                                              n_shortcut=len(sc), clr=clr)
