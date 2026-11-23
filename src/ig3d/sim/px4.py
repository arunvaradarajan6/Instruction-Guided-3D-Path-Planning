"""Fly a mission in PX4 SITL + Gazebo (gz Harmonic) with MAVSDK offboard control.

Needs a built PX4-Autopilot (`make px4_sitl`), Gazebo Harmonic, and `pip install mavsdk`.
PX4 is located through --px4-dir or $PX4_DIR.

The vehicle is the stock gz_x500. The reference is streamed as position + velocity
setpoints at the reference rate. Reference time only advances while the drone keeps up
(lag-based time scaling), so a slow simulator or strong wind stretches the flight instead
of letting the setpoint run away from the vehicle.

Wind: a horizontal drag force F = 0.5 rho CdA v^2 on the airframe through Gazebo's
ApplyLinkWrench system, re-drawn every `gust_period` seconds with relative std `gust`.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from .geometry import enu2ned, ned2enu
from .mission import load_mission

MODEL_INSTANCE = "x500_0"
RHO, CDA = 1.225, 0.10


def px4_dir(arg=None) -> Path:
    d = Path(arg or os.environ.get("PX4_DIR", Path.home() / "PX4-Autopilot")).expanduser()
    if not (d / "build" / "px4_sitl_default" / "bin" / "px4").exists():
        raise SystemExit("PX4 SITL not found under {} (build it with `make px4_sitl`, or pass "
                         "--px4-dir / set PX4_DIR)".format(d))
    return d


def _kill_stale(world_name):
    for pat in ("gz sim.*{}.sdf".format(world_name), "build/px4_sitl_default/bin/px4"):
        subprocess.run(["pkill", "-f", pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)


def launch(px4: Path, mission: dict, log_path: Path, gui: bool = False):
    """Start PX4 SITL, which starts Gazebo with our world and spawns gz_x500 at the spawn pose."""
    world_name = mission["world_name"]
    _kill_stale(world_name)
    worlds = px4 / "Tools" / "simulation" / "gz" / "worlds"
    shutil.copy(mission["world"], worlds / (world_name + ".sdf"))
    e, n, u = mission["spawn_enu"]
    env = dict(os.environ, PX4_SYS_AUTOSTART="4001", PX4_SIM_MODEL="gz_x500",
               PX4_GZ_WORLD=world_name, PX4_GZ_MODEL_POSE="{:.3f},{:.3f},{:.3f},0,0,0".format(
                   e, n, u + 0.05))
    if not gui:
        env["HEADLESS"] = "1"
    log = open(log_path, "w")
    proc = subprocess.Popen([str(px4 / "build" / "px4_sitl_default" / "bin" / "px4"), "-d"],
                            cwd=str(px4), env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    return proc, log


def shutdown(proc, log, world_name):
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    log.close()
    _kill_stale(world_name)


def gz_model_position(world_name, model=MODEL_INSTANCE, timeout=10.0):
    """The model's ground-truth position from Gazebo, or None."""
    try:
        out = subprocess.run(["gz", "topic", "-e", "-n", "1", "-t",
                              "/world/{}/pose/info".format(world_name)],
                             capture_output=True, text=True, timeout=timeout).stdout
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r'name: "{}".*?position \{{([^}}]*)\}}'.format(re.escape(model)), out, re.S)
    if not m:
        return None
    vals = dict(re.findall(r"([xyz]):\s*([-0-9.eE+]+)", m.group(1)))
    return np.array([float(vals.get(k, 0.0)) for k in "xyz"])


async def _gz_pub(topic, msgtype, payload):
    p = await asyncio.create_subprocess_exec("gz", "topic", "-t", topic, "-m", msgtype, "-p", payload,
                                             stdout=asyncio.subprocess.DEVNULL,
                                             stderr=asyncio.subprocess.DEVNULL)
    await p.wait()


async def wind_task(world_name, speed, direction_deg, gust, period, stop, log):
    """Persistent horizontal wind force on the airframe, re-drawn every `period` seconds."""
    rng = np.random.default_rng(0)
    ent = 'name: "{}", type: MODEL'.format(MODEL_INSTANCE)
    th = math.radians(direction_deg)
    while not stop.is_set():
        v = max(0.0, speed * (1.0 + gust * rng.standard_normal()))
        f = 0.5 * RHO * CDA * v * v
        fx, fy = f * math.cos(th), f * math.sin(th)
        await _gz_pub("/world/{}/wrench/clear".format(world_name), "gz.msgs.Entity", ent)
        await _gz_pub("/world/{}/wrench/persistent".format(world_name), "gz.msgs.EntityWrench",
                      "entity: {{{}}}, wrench: {{force: {{x: {:.4f}, y: {:.4f}, z: 0}}}}".format(
                          ent, fx, fy))
        log.append((time.monotonic(), v, fx, fy))
        try:
            await asyncio.wait_for(stop.wait(), timeout=period)
        except asyncio.TimeoutError:
            pass
    await _gz_pub("/world/{}/wrench/clear".format(world_name), "gz.msgs.Entity", ent)


async def fly_async(mission_dir, port=14540, wind=None, health_timeout=180.0,
                    lag_ok=1.0, lag_span=1.5):
    from mavsdk import System
    from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw

    m, arr, ref = load_mission(mission_dir)
    lim = m["limits"]
    drone = System()
    await drone.connect(system_address="udp://:{}".format(port))
    t0 = time.monotonic()
    async for st in drone.core.connection_state():
        if st.is_connected:
            break
    async for h in drone.telemetry.health():
        if h.is_local_position_ok and h.is_global_position_ok and h.is_home_position_ok:
            break
        if time.monotonic() - t0 > health_timeout:
            raise RuntimeError("PX4 never became healthy")
    for name, val in (("COM_RCL_EXCEPT", 4), ("NAV_RCL_ACT", 0), ("NAV_DLL_ACT", 0)):
        try:
            await drone.param.set_param_int(name, val)
        except Exception:  # noqa: BLE001
            pass
    for name, val in (("MPC_XY_VEL_MAX", lim["v_xy"] + 2.0), ("MPC_Z_VEL_MAX_UP", lim["vz_up"] + 1.0),
                      ("MPC_Z_VEL_MAX_DN", min(4.0, lim["vz_dn"] + 1.0))):
        try:
            await drone.param.set_param_float(name, float(val))
        except Exception:  # noqa: BLE001
            pass

    origin = gz_model_position(m["world_name"])
    origin_src = "gazebo"
    if origin is None:
        origin, origin_src = np.asarray(m["spawn_enu"], float) + [0, 0, 0.24], "spawn+0.24"

    state = dict(pos=None, vel=None, in_air=False)
    log = []

    async def watch_pv():
        await drone.telemetry.set_rate_position_velocity_ned(50.0)
        async for pv in drone.telemetry.position_velocity_ned():
            state["pos"] = np.array([pv.position.north_m, pv.position.east_m, pv.position.down_m])
            state["vel"] = np.array([pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s])

    async def watch_air():
        async for a in drone.telemetry.in_air():
            state["in_air"] = a

    tasks = [asyncio.ensure_future(watch_pv()), asyncio.ensure_future(watch_air())]
    while state["pos"] is None:
        await asyncio.sleep(0.05)

    T, P, V = ref["t"], ref["pos"], ref["vel"]
    rate = lim["rate_hz"]

    def ref_at(tau):
        i = min(len(T) - 1, int(np.searchsorted(T, tau)))
        return P[i], V[i], int(ref["phase"][i])

    def to_ned(p_enu):
        return enu2ned(np.asarray(p_enu) - origin)

    yaw = 0.0
    p0 = to_ned(P[0])
    await drone.offboard.set_position_ned(PositionNedYaw(*p0, yaw))
    await drone.action.arm()
    try:
        await drone.offboard.start()
    except OffboardError as e:
        await drone.action.disarm()
        raise RuntimeError("offboard start failed: {}".format(e._result.result))

    wind_log, stop = [], asyncio.Event()
    if wind and wind.get("speed", 0) > 0:
        tasks.append(asyncio.ensure_future(wind_task(m["world_name"], wind["speed"], wind["dir"],
                                                     wind["gust"], wind["period"], stop, wind_log)))

    tau, last = 0.0, time.monotonic()
    start_wall = last
    crashed = False
    while tau < T[-1]:
        now = time.monotonic()
        dt, last = now - last, now
        pos_enu = ned2enu(state["pos"]) + origin
        pr, vr, ph = ref_at(tau)
        lag = float(np.linalg.norm(pos_enu - pr))
        scale = float(np.clip(1.0 - (lag - lag_ok) / lag_span, 0.0, 1.0))
        tau += dt * scale
        pr, vr, ph = ref_at(tau)
        vr = vr * scale
        if np.hypot(vr[0], vr[1]) > 0.3:
            yaw = math.degrees(math.atan2(vr[0], vr[1]))  # NED yaw: atan2(east, north)
        pn, vn = to_ned(pr), enu2ned(vr)
        await drone.offboard.set_position_velocity_ned(PositionNedYaw(*pn, yaw),
                                                       VelocityNedYaw(*vn, yaw))
        log.append((now - start_wall, tau, ph, *pos_enu, *ned2enu(state["vel"]), lag))
        if ph >= 1 and not state["in_air"]:
            crashed = True
            break
        if now - start_wall > 4 * T[-1] + 120:
            break
        await asyncio.sleep(max(0.0, 1.0 / rate - (time.monotonic() - now)))

    stop.set()
    try:
        await drone.offboard.stop()
    except Exception:  # noqa: BLE001
        pass
    await drone.action.land()
    t_land = time.monotonic()
    while state["in_air"] and time.monotonic() - t_land < 60:
        await asyncio.sleep(0.2)
    for t in tasks:
        t.cancel()
    L = np.asarray(log, float)
    np.savez_compressed(Path(mission_dir) / "flight.npz", t=L[:, 0], tau=L[:, 1], phase=L[:, 2],
                        pos_enu=L[:, 3:6], vel_enu=L[:, 6:9], lag=L[:, 9],
                        wind=np.asarray(wind_log, float).reshape(-1, 4))
    info = dict(backend="px4", origin_enu=origin.tolist(), origin_source=origin_src,
                completed=bool(tau >= T[-1]), crashed=crashed, wall_s=float(L[-1, 0]),
                ref_s=float(T[-1]), wind=wind or None)
    (Path(mission_dir) / "flight.json").write_text(json.dumps(info, indent=1))
    return info


def fly(mission_dir, px4=None, gui=False, wind=None, port=14540):
    m, _, _ = load_mission(mission_dir)
    root = px4_dir(px4)
    proc, log = launch(root, m, Path(mission_dir) / "px4.log", gui)
    try:
        return asyncio.run(fly_async(mission_dir, port, wind))
    finally:
        shutdown(proc, log, m["world_name"])
