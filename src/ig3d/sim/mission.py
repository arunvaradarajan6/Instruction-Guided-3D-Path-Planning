"""One mission: (block, instruction, model) -> planned path -> flyable reference -> world."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import DEFAULT, CostConfig
from ..costfield import gt_field_from_features, volume_features
from ..instructions import CLASSES
from ..planner_grid import GridGraph
from .geometry import ground_below, vox2enu
from .trajectory import Limits, build_reference, feasibility, make_flyable
from .world import WORLD_NAME, write_world

SYNTHETIC_TEST_SEED = 3000
SYNTHETIC_VOXEL_M = 1.0


def load_volume(source: str, index: int):
    """(volume, voxel size in metres). source: 'synthetic' or a LiDAR city name."""
    if source == "synthetic":
        from ..volumes import generate
        return generate(index + 1, SYNTHETIC_TEST_SEED, DEFAULT.volume)[index], SYNTHETIC_VOXEL_M
    from ..lidar import VOXEL_M, load_blocks
    vols = load_blocks(city=source)
    if not 0 <= index < len(vols):
        raise SystemExit("{} has {} blocks; index {} is out of range".format(source, len(vols), index))
    return vols[index], VOXEL_M


def checkpoint_path(model: str, seed: int = 0) -> Path:
    p = Path(model)
    if p.suffix == ".pt" and p.exists():
        return p
    from ..multiseed import run_dir
    return run_dir(model, seed) / "best.pt"


def condition(meta, cls: str | None, text: str | None, sentences: str, rng):
    """(vector, sentence used or None). One-hot reads the class; text models read text."""
    from ..encoder import encode_free_text, sentence_pool
    if text:
        return encode_free_text(meta, text), text
    if cls is None:
        raise SystemExit("give an instruction class (--cls) or free text (--text)")
    ci = CLASSES.index(cls)
    if meta["conditioning"] == "onehot":
        return np.eye(len(CLASSES), dtype=np.float32)[ci], None
    vec, lab, sents = sentence_pool(meta, sentences)
    si = int(rng.choice(np.flatnonzero(lab == ci)))
    return vec[si], sents[si]


def build_mission(out_dir, source: str, index: int, model: str = "oracle", cls: str | None = None,
                  text: str | None = None, sentences: str = "held", seed: int = 0,
                  lim: Limits | None = None, draw_path: bool = True) -> dict:
    """Write mission.json, mission.npz and world.sdf into out_dir. Returns the mission dict."""
    lim = lim or Limits()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vol, s = load_volume(source, index)
    if source != "synthetic":
        from ..cities import HELDOUT
        if source not in HELDOUT:
            print("WARNING: {} is a TRAINING city; flights there are not a held-out test".format(
                source))
        lim = Limits(**{**lim.as_dict(), "v_xy": max(lim.v_xy, 4.0)}) if s >= 4 else lim

    ccfg = DEFAULT.cost
    sent = None
    if model == "oracle":
        if cls is None:
            raise SystemExit("the oracle needs --cls")
        field, ckpt = None, None
    else:
        from ..device import get_device
        from ..train import load_model, predict
        ckpt = checkpoint_path(model, seed)
        dev = get_device()
        net, meta = load_model(ckpt, dev)
        ccfg = CostConfig(**meta["config"]["cost"])
        rng = np.random.default_rng((index, CLASSES.index(cls) if cls else 99, seed))
        vec, sent = condition(meta, cls, text, sentences, rng)
        field = predict(net, vol.occ, vec, ccfg, dev)

    feats = volume_features(vol, ccfg)
    G = GridGraph(vol.occ)
    truth = gt_field_from_features(vol.occ, feats, cls, ccfg, 3) if cls else None
    plan_field = truth if field is None else field
    path, _ = G.plan(plan_field, vol.start, vol.goal)
    if path is None:
        raise RuntimeError("no path from start to goal")
    opt_path = G.plan(truth, vol.start, vol.goal)[0] if truth is not None else None

    lim = lim.for_voxel(s)
    sc, sm, cruise_enu, info = make_flyable(path, plan_field, vol.occ, s, lim)
    clr = info.pop("clr")
    g = ground_below(vol.occ, vol.start)
    start_enu = vox2enu(vol.start, s)
    spawn_enu = np.array([start_enu[0], start_enu[1], g * s])
    ref = build_reference(cruise_enu, spawn_enu, lim)
    feas = feasibility(ref, sm, clr, lim)

    world = write_world(vol.occ, s, out / "world.sdf", cruise_enu if draw_path else None,
                        start_enu, vox2enu(vol.goal, s))
    mission = dict(source=source, index=index, voxel_m=s, model=model, ckpt=str(ckpt) if ckpt else None,
                   seed=seed, cls=cls, text=text, sentence=sent, sentences=sentences,
                   start=list(map(int, vol.start)), goal=list(map(int, vol.goal)),
                   spawn_enu=spawn_enu.tolist(), world=str(world), world_name=WORLD_NAME,
                   limits=lim.as_dict(), feasibility=feas, smoothing=info)
    np.savez_compressed(out / "mission.npz", grid_path=path,
                        opt_path=opt_path if opt_path is not None else np.zeros((0, 3)),
                        shortcut=sc, smooth=sm, cruise_enu=cruise_enu,
                        field=plan_field.astype(np.float16), **{"ref_" + k: v for k, v in ref.items()})
    (out / "mission.json").write_text(json.dumps(mission, indent=1, default=float))
    return mission


def load_mission(d):
    d = Path(d)
    m = json.loads((d / "mission.json").read_text())
    z = np.load(d / "mission.npz")
    arrays = {k: z[k] for k in z.files}
    ref = {k[4:]: v for k, v in arrays.items() if k.startswith("ref_")}
    return m, arrays, ref
