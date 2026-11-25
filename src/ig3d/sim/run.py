"""Command line for the simulation pipeline.

Single mission:
    python -m ig3d.sim.run mission --source seattle_downtown --index 3 --model k3_nli_adapter_real \
        --cls level --out ../datasets/sim/demo
    python -m ig3d.sim.run fly ../datasets/sim/demo --gui          # PX4 + Gazebo
    python -m ig3d.sim.run dryrun ../datasets/sim/demo             # no PX4
    python -m ig3d.sim.run analyse ../datasets/sim/demo

Batch over held-out blocks (resumes; each PX4 flight runs in its own process):
    python -m ig3d.sim.run batch --run heldout_v2 --sources heldout --n 10 \
        --models oracle k1_onehot_real k3_onehot_real k3_nli_adapter_real --backend px4
    python -m ig3d.sim.run report --run heldout_v2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..instructions import CLASSES
from ..paths import SIM, SRC
from .trajectory import Limits


def _wind(a):
    if not a.wind:
        return None
    return dict(speed=a.wind, dir=a.wind_dir, gust=a.gust, period=a.gust_period)


def _limits(a):
    return Limits(v_xy=a.v_xy, vz_up=a.vz_up, vz_dn=a.vz_dn, a_tan=a.a_tan, a_lat=a.a_lat,
                  robot_radius=a.robot_radius)


def _sources(names):
    from ..cities import HELDOUT, resolve
    out = []
    for n in names:
        if n == "synthetic":
            out.append(n)
        elif n == "heldout":
            out += list(HELDOUT)
        else:
            out += resolve([n])
    return list(dict.fromkeys(out))


def _classes_for(source):
    from ..lidar import REAL_CLASSES
    return list(CLASSES) if source == "synthetic" else list(REAL_CLASSES)


def _n_blocks(source):
    if source == "synthetic":
        return 10 ** 6
    from ..lidar import load_blocks
    return len(load_blocks(city=source))


def climbing(sources) -> list[str]:
    """'city:index' of every block whose true shortest path climbs at least once.

    The same test as the ladder's informative subset (`lidar._informative`): only on these
    blocks can a level instruction cost K=1 anything, so a level flight test needs them."""
    from ..config import DEFAULT
    from ..costfield import gt_field_from_features, volume_features
    from ..planner_grid import GridGraph
    from .mission import load_volume
    out = []
    for src in _sources(sources):
        for i in range(_n_blocks(src)):
            vol, _ = load_volume(src, i)
            f = gt_field_from_features(vol.occ, volume_features(vol, DEFAULT.cost), "shortest",
                                       DEFAULT.cost, 3)
            path, _ = GridGraph(vol.occ).plan(f, vol.start, vol.goal)
            if path is not None and np.any(np.diff(np.asarray(path)[:, 0]) > 0):
                out.append("{}:{}".format(src, i))
    return out


def batch(a):
    from .analyse import analyse
    from .mission import build_mission
    run = SIM / a.run
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(a), indent=1, default=str))
    jobs = []
    picks = {}
    if a.blocks:  # explicit "city:index" list, e.g. from `run.py climbing`
        for b in a.blocks:
            src, i = b.rsplit(":", 1)
            picks.setdefault(src, []).append(int(i))
    else:
        for src in _sources(a.sources):
            n = min(a.n, _n_blocks(src))
            picks[src] = np.random.default_rng(a.seed).permutation(_n_blocks(src))[:n] \
                if src != "synthetic" else np.arange(n)
    for src, idx in picks.items():
        for i in sorted(int(v) for v in idx):
            for model in a.models:
                for cls in (a.classes or _classes_for(src)):
                    jobs.append((src, i, model, cls))
    print("{} flights in {}".format(len(jobs), run), flush=True)
    for k, (src, i, model, cls) in enumerate(jobs):
        d = run / "{}_{:03d}".format(src, i) / Path(model).stem / cls
        if (d / "metrics.json").exists():
            continue
        t0 = time.time()
        try:
            if not (d / "mission.json").exists():
                build_mission(d, src, i, model, cls, sentences=a.sentences, seed=a.model_seed,
                              lim=_limits(a), draw_path=False)
            if a.backend == "px4":
                cmd = [sys.executable, "-m", "ig3d.sim.run", "fly", str(d), "--port", str(a.port)]
                if a.px4_dir:
                    cmd += ["--px4-dir", a.px4_dir]
                if a.wind:
                    cmd += ["--wind", str(a.wind), "--wind-dir", str(a.wind_dir), "--gust",
                            str(a.gust), "--gust-period", str(a.gust_period)]
                rc = subprocess.run(cmd, cwd=SRC, timeout=a.flight_timeout).returncode
                if rc != 0:
                    raise RuntimeError("flight process exited {}".format(rc))
            else:
                from .dryrun import fly as dry
                dry(d, _wind(a))
            r = analyse(d)
            print("[{}/{}] {} #{} {} {}: success={} flown_regret={} ({:.0f}s)".format(
                k + 1, len(jobs), src, i, model, cls, r["success"],
                None if r.get("flown_regret") is None else round(r["flown_regret"], 4),
                time.time() - t0), flush=True)
        except Exception as e:  # noqa: BLE001
            d.mkdir(parents=True, exist_ok=True)
            (d / "error.txt").write_text(repr(e))
            print("[{}/{}] {} #{} {} {}: FAILED {}".format(k + 1, len(jobs), src, i, model, cls, e),
                  flush=True)


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def report(a):
    run = SIM / a.run
    rows = [json.loads(p.read_text()) for p in sorted(run.glob("**/metrics.json"))]
    errors = list(run.glob("**/error.txt"))
    if not rows:
        raise SystemExit("no results in " + str(run))
    models = list(dict.fromkeys(r["model"] for r in rows))
    classes = [c for c in CLASSES if any(r["cls"] == c for r in rows)]
    keys = (("n", None), ("success %", "success"), ("collision %", "collision"),
            ("planned regret %", "planned_regret"), ("reference regret %", "reference_regret"),
            ("flown regret %", "flown_regret"), ("cross-track RMS m", "cross_track_rms"),
            ("min clearance m", "min_clearance_m"), ("cruise time s", "cruise_time_s"))
    lines = ["# Simulation results: {}".format(a.run), "",
             "{} flights, {} failed to run. Backend(s): {}.".format(
                 len(rows), len(errors), ", ".join(sorted({r["backend"] for r in rows}))), ""]
    summary = {}
    for model in models:
        lines += ["## " + model, "", "| class | " + " | ".join(k for k, _ in keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for cls in classes + ["ALL"]:
            rs = [r for r in rows if r["model"] == model and (cls == "ALL" or r["cls"] == cls)]
            if not rs:
                continue
            cells = []
            for name, k in keys:
                if k is None:
                    cells.append(str(len(rs)))
                    continue
                v = _mean([float(r.get(k)) if r.get(k) is not None else None for r in rs])
                if "%" in name:
                    v *= 100
                cells.append("{:.2f}".format(v))
                summary.setdefault(model, {}).setdefault(cls, {})[k] = v
            lines.append("| {} | {} |".format("**all**" if cls == "ALL" else cls, " | ".join(cells)))
        lines.append("")
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (run / "report.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def lim_args(p):
        p.add_argument("--v-xy", type=float, default=3.0)
        p.add_argument("--vz-up", type=float, default=1.5)
        p.add_argument("--vz-dn", type=float, default=1.0)
        p.add_argument("--a-tan", type=float, default=1.5)
        p.add_argument("--a-lat", type=float, default=2.0)
        p.add_argument("--robot-radius", type=float, default=0.38)

    def wind_args(p):
        p.add_argument("--wind", type=float, default=0.0, help="mean wind speed, m/s")
        p.add_argument("--wind-dir", type=float, default=0.0, help="direction it blows TOWARDS, deg from East")
        p.add_argument("--gust", type=float, default=0.3, help="relative std of the gusts")
        p.add_argument("--gust-period", type=float, default=2.0)

    p = sub.add_parser("mission")
    p.add_argument("--source", required=True, help="synthetic or a LiDAR city (e.g. seattle_downtown)")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--model", default="oracle", help="oracle, a model name, or a .pt path")
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--cls", choices=CLASSES)
    p.add_argument("--text", help="free-text instruction (nli / embedding models)")
    p.add_argument("--sentences", default="held", choices=["held", "train", "hard"])
    p.add_argument("--out", required=True)
    lim_args(p)

    p = sub.add_parser("fly")
    p.add_argument("mission")
    p.add_argument("--px4-dir")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--port", type=int, default=14540)
    wind_args(p)

    p = sub.add_parser("dryrun")
    p.add_argument("mission")
    wind_args(p)

    p = sub.add_parser("analyse")
    p.add_argument("mission")

    p = sub.add_parser("batch")
    p.add_argument("--run", required=True, help="name of the results folder under datasets/sim")
    p.add_argument("--sources", nargs="+", default=["heldout"],
                   help="heldout, synthetic, or city names")
    p.add_argument("--n", type=int, default=10, help="blocks per source")
    p.add_argument("--models", nargs="+", default=["oracle", "k3_onehot"])
    p.add_argument("--model-seed", type=int, default=0)
    p.add_argument("--classes", nargs="*", choices=CLASSES)
    p.add_argument("--sentences", default="held", choices=["held", "train", "hard"])
    p.add_argument("--backend", default="px4", choices=["px4", "dryrun"])
    p.add_argument("--px4-dir")
    p.add_argument("--port", type=int, default=14540)
    p.add_argument("--seed", type=int, default=0, help="which blocks are picked")
    p.add_argument("--blocks", nargs="*", help="explicit city:index blocks instead of --sources/--n")
    p.add_argument("--flight-timeout", type=float, default=900)
    lim_args(p)
    wind_args(p)

    p = sub.add_parser("report")
    p.add_argument("--run", required=True)

    p = sub.add_parser("climbing", help="list blocks whose true shortest path climbs (city:index)")
    p.add_argument("--sources", nargs="+", default=["heldout"])

    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "climbing":
        print(" ".join(climbing(a.sources)))
        return
    if a.cmd == "mission":
        from .mission import build_mission
        m = build_mission(a.out, a.source, a.index, a.model, a.cls, a.text, a.sentences,
                          a.model_seed, _limits(a))
        print(json.dumps(dict(out=a.out, sentence=m["sentence"], feasibility=m["feasibility"],
                              smoothing=m["smoothing"]), indent=1, default=float))
    elif a.cmd == "fly":
        from .px4 import fly
        print(json.dumps(fly(a.mission, a.px4_dir, a.gui, _wind(a), a.port), indent=1))
    elif a.cmd == "dryrun":
        from .dryrun import fly
        print(json.dumps(fly(a.mission, _wind(a)), indent=1))
    elif a.cmd == "analyse":
        from .analyse import analyse
        print(json.dumps(analyse(a.mission), indent=1, default=float))
    elif a.cmd == "batch":
        batch(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
