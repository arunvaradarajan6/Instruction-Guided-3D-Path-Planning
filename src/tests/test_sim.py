import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ig3d.costfield import gt_field, polyline_cost
from ig3d.planner_grid import GridGraph
from ig3d.sim.geometry import Clearance, enu2ned, enu2vox, merge_boxes, ned2enu, vox2enu
from ig3d.sim.trajectory import Limits, build_reference, make_flyable, time_parameterise


def test_merge_boxes_covers_exactly(vols):
    occ = vols[0].occ
    cover = np.zeros(occ.shape, int)
    for z0, z1, y0, y1, x0, x1 in merge_boxes(occ):
        cover[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] += 1
    assert (cover == occ.astype(int)).all()


def test_frames_round_trip():
    p = np.array([[3.0, 7.0, 11.0], [0.0, 0.0, 0.0]])
    assert np.allclose(enu2vox(vox2enu(p, 1.5), 1.5), p)
    d = np.array([1.0, 2.0, 3.0])
    assert np.allclose(ned2enu(enu2ned(d)), d)
    assert np.allclose(vox2enu([0, 0, 0], 2.0), [1.0, 1.0, 1.0])


def test_flyable_path_is_clear_and_no_more_expensive(vols, cfg):
    v = vols[0]
    f = gt_field(v, "level", cfg.cost)
    path, _ = GridGraph(v.occ).plan(f, v.start, v.goal)
    lim = Limits()
    sc, sm, cruise, info = make_flyable(path, f, v.occ, 1.0, lim)
    clr = info["clr"]
    assert clr.polyline(sm) >= min(lim.r_safe, clr.polyline(path)) - 1e-6
    assert polyline_cost(f, sc) <= polyline_cost(f, path) * (1 + 1e-6)
    assert np.allclose(sm[0], path[0]) and np.allclose(sm[-1], path[-1])


def test_time_parameterisation_respects_limits():
    lim = Limits(v_xy=2.0, vz_up=1.0, vz_dn=0.5, a_tan=1.0)
    X = np.array([[0, 0, 0], [10, 0, 0], [10, 0, 5], [20, 0, 0]], float)
    t, P, v, T = time_parameterise(X, lim)
    assert v[0] == 0 and v[-1] == 0
    vel = T * v[:, None]
    assert np.hypot(vel[:, 0], vel[:, 1]).max() <= lim.v_xy + 1e-6
    assert vel[:, 2].max() <= lim.vz_up + 1e-6
    assert -vel[:, 2].min() <= lim.vz_dn + 1e-6
    ref = build_reference(X, np.array([0, 0, -3.0]), lim)
    assert set(np.unique(ref["phase"])) == {0, 1, 2, 3}
    assert np.all(np.diff(ref["t"]) > 0)


def test_mission_dryrun_analyse_end_to_end(tmp_path):
    from ig3d.sim.analyse import analyse
    from ig3d.sim.dryrun import fly
    from ig3d.sim.mission import build_mission
    d = tmp_path / "m"
    build_mission(d, "synthetic", 0, "oracle", "level")
    assert (d / "world.sdf").exists()
    fly(d)
    r = analyse(d)
    assert r["success"] and not r["collision"]
    assert abs(r["planned_regret"]) < 1e-9
    assert r["flown_regret"] < 0.05


def test_corpus_v2_keeps_paper_indices_and_held_out_split():
    code = ("from ig3d.instructions import *; tr, he = split_indices(seed=1); "
            "print(len(ALL_SENTENCES), ALL_SENTENCES[127], list(he))")
    src = Path(__file__).resolve().parents[1]
    out = {c: subprocess.run([sys.executable, "-c", code], cwd=src, capture_output=True, text=True,
                             env=dict(os.environ, IG3D_CORPUS=c)).stdout.strip()
           for c in ("v1", "v2")}
    n1, rest1 = out["v1"].split(" ", 1)
    n2, rest2 = out["v2"].split(" ", 1)
    assert (n1, n2) == ("128", "384")
    assert rest1 == rest2


def test_city_resolution_never_mixes_train_and_heldout():
    from ig3d.cities import HELDOUT, TRAIN, resolve
    assert not set(TRAIN) & set(HELDOUT)
    assert set(resolve(["train"])) == set(TRAIN)
    assert "austin" in HELDOUT


def test_clearance_is_zero_on_faces():
    occ = np.zeros((5, 5, 5), np.uint8)
    occ[2, 2, 2] = 1
    c = Clearance(occ, 1.0)
    assert c([[2, 2, 2]])[0] < 0
    assert abs(c([[2, 2, 2.5]])[0]) < 1e-6


def test_ept_node_bounds_tile_the_root_cube():
    from ig3d.cities import _overlaps, node_bounds, utm_epsg
    cube = [0.0, 0.0, 0.0, 8.0, 8.0, 8.0]
    assert node_bounds(cube, "0-0-0-0") == (0.0, 0.0, 8.0, 8.0)
    assert node_bounds(cube, "2-3-1-0") == (6.0, 2.0, 8.0, 4.0)
    assert _overlaps(node_bounds(cube, "1-1-1-0"), (5.0, 5.0, 6.0, 6.0))
    assert not _overlaps(node_bounds(cube, "1-0-0-0"), (5.0, 5.0, 6.0, 6.0))
    assert utm_epsg(-122.33, 47.6) == 32610 and utm_epsg(-80.19, 25.76) == 32617
