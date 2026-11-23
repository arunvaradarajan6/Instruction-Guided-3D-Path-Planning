"""Score a flight: is the path that was actually FLOWN still good under the true field?

Regret is reported at three stages, all under the true field and all against the same
optimum (the exact planner on the true field), with one cost discretisation
(`polyline_cost`, 0.5-voxel sampling) so the stages are comparable:

    planned    the grid path the planner returned on the predicted field
    reference  after shortcutting, smoothing and time parameterisation
    flown      the executed trajectory (cruise phase), from the flight log

Smoothing can cut corners the 26-connected grid cannot, so reference and flown regret
can be slightly negative. `grid_regret` is the paper's number for the planned path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..config import DEFAULT
from ..costfield import gt_field_from_features, polyline_cost, volume_features
from ..metrics import behaviour, regret
from ..planner_grid import path_cost
from .geometry import Clearance, enu2vox
from .mission import load_mission, load_volume

STEP = 0.5
P_HOVER, MASS, G = 220.0, 2.0, 9.81
K_DRAG = 0.5 * 1.225 * 0.10


def _arc_resample(P, ds):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < ds:
        return P[[0, -1]]
    sq = np.linspace(0, s[-1], int(np.ceil(s[-1] / ds)) + 1)
    return np.stack([np.interp(sq, s, P[:, k]) for k in range(3)], axis=1)


def energy_proxy(t, vel):
    """Joules: hover power + climb power + parasitic drag. A proxy, not a battery model."""
    if len(t) < 2:
        return 0.0
    sp = np.linalg.norm(vel, axis=1)
    pw = P_HOVER + MASS * G * np.maximum(vel[:, 2], 0.0) + K_DRAG * sp ** 3
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(pw, t))


def analyse(mission_dir) -> dict:
    d = Path(mission_dir)
    m, arr, ref = load_mission(d)
    flight = np.load(d / "flight.npz")
    finfo = json.loads((d / "flight.json").read_text())
    vol, s = load_volume(m["source"], m["index"])
    clr = Clearance(vol.occ, s)
    feats = volume_features(vol, DEFAULT.cost)
    out = dict(source=m["source"], index=m["index"], model=m["model"], cls=m["cls"],
               sentence=m["sentence"], backend=finfo["backend"], wind=finfo.get("wind"),
               feasibility=m["feasibility"])

    pos = flight["pos_enu"]
    cruise = flight["phase"] == 2
    pv = enu2vox(pos, s)
    goal_enu = np.array([(m["goal"][2] + .5) * s, (m["goal"][1] + .5) * s, (m["goal"][0] + .5) * s])
    d_goal = np.linalg.norm(pos - goal_enu, axis=1)
    reached = bool(d_goal.min() <= max(1.0, 0.75 * s))
    c_fly = clr(pv[flight["phase"] >= 1]) if (flight["phase"] >= 1).any() else np.array([np.inf])
    min_clear = float(c_fly.min())
    collided = bool(min_clear < m["limits"]["robot_radius"] * 0.5)
    out.update(reached_goal=reached, completed=finfo["completed"], crashed=finfo["crashed"],
               min_clearance_m=min_clear, collision=collided,
               success=bool(reached and finfo["completed"] and not finfo["crashed"] and not collided))

    if cruise.sum() >= 2:
        ref_cruise = ref["pos"][ref["phase"] == 2]
        tree = cKDTree(ref_cruise)
        xt = tree.query(pos[cruise])[0]
        out.update(cross_track_rms=float(np.sqrt((xt ** 2).mean())), cross_track_max=float(xt.max()),
                   cruise_time_s=float(flight["t"][cruise][-1] - flight["t"][cruise][0]),
                   ref_cruise_time_s=m["feasibility"]["duration_s"],
                   energy_proxy_J=energy_proxy(flight["t"][cruise], flight["vel_enu"][cruise]),
                   max_lag_m=float(flight["lag"].max()))

    if m["cls"]:
        truth = gt_field_from_features(vol.occ, feats, m["cls"], DEFAULT.cost, 3)
        opt = arr["opt_path"]
        c_opt = polyline_cost(truth, opt, STEP)
        flown = _arc_resample(pv[cruise], 0.25) if cruise.sum() >= 2 else None
        out.update(
            grid_regret=regret(path_cost(truth, arr["grid_path"]), path_cost(truth, opt)),
            planned_regret=polyline_cost(truth, arr["grid_path"], STEP) / c_opt - 1,
            reference_regret=polyline_cost(truth, arr["smooth"], STEP) / c_opt - 1,
            flown_regret=(polyline_cost(truth, flown, STEP) / c_opt - 1) if flown is not None else None)
        if flown is not None:
            b = behaviour(vol, flown, feats["edt"])
            bp = behaviour(vol, arr["grid_path"], feats["edt"])
            for k in ("vertical", "mean_alt", "length", "mean_clear", "narrow_frac",
                      "climb_clear", "descend_clear"):
                out["flown_" + k], out["planned_" + k] = b[k], bp[k]
    (d / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    return out
