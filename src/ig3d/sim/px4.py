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
    """Clear what a killed flight leaves behind: gz sim can ignore SIGTERM, and an orphaned
    mavsdk_server keeps UDP 14540 bound, which fails every later flight with a bind error."""
    pats = ("gz sim.*{}.sdf".format(world_name), "build/px4_sitl_default/bin/px4",
            "mavsdk/bin/mavsdk_server")
    for sig in ("-TERM", "-KILL"):
        for pat in pats:
            subprocess.run(["pkill", sig, "-f", pat], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        time.sleep(1.0)


def launch(px4: Path, mission: dict, log_path: Path, gui: bool = False):
    """Start PX4 SITL, which starts Gazebo with our world and spawns gz_x500 at the spawn pose."""
    world_name = mission["world_name"]
    _kill_stale(world_name)
    worlds = px4 / "Tools" / "simulation" / "gz" / "worlds"
    world_sdf = worlds / (world_name + ".sdf")
    shutil.copy(mission["world"], world_sdf)
    # Start Gazebo ourselves and wait for its spawn service. PX4's startup script would start
    # it too, but then asks to spawn the drone once, after ~1 s: a city world with hundreds of
    # boxes is not loaded by then and PX4 gives up. Finding a running world, it attaches.
    models = px4 / "Tools" / "simulation" / "gz" / "models"
    gz_env = dict(os.environ, GZ_SIM_RESOURCE_PATH=":".join(
        p for p in (os.environ.get("GZ_SIM_RESOURCE_PATH", ""), str(models), str(worlds)) if p))
    gz_log = open(log_path.with_name("gz.log"), "w")
    subprocess.Popen(["gz", "sim", "--verbose=1", "-r", "-s", str(world_sdf)], env=gz_env,
                     stdout=gz_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
    t0 = time.monotonic()
    want = "/world/{}/create".format(world_name)
    while time.monotonic() - t0 < 180:
        out = subprocess.run(["gz", "service", "-l"], capture_output=True, text=True,
                             timeout=30).stdout
        if want in out.split():
            break
        time.sleep(1.0)
    time.sleep(2.0)
    if gui:
        subprocess.Popen(["gz", "sim", "-g"], env=gz_env, stdout=gz_log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
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

    async def connected():
        async for st in drone.core.connection_state():
            if st.is_connected:
                return

    await asyncio.wait_for(connected(), timeout=120.0)
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
    crashed = aborted = False
    was_air, stalled_since = False, None
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
        # PX4's in-air flag lags takeoff by a second or two: a crash is being airborne and then
        # down again before the end, or never getting airborne at all.
        was_air = was_air or state["in_air"]
        if (was_air and not state["in_air"]) or (not was_air and now - start_wall > 30.0):
            crashed = True
            break
        # the reference waits for a lagging drone; one that stays far off for 20 s (after a
        # collision, say) will not recover, so stop instead of drifting until the timeout
        if scale <= 0.0:
            stalled_since = stalled_since or now
            if now - stalled_since > 20.0:
                aborted = True
                break
        else:
            stalled_since = None
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
                completed=bool(tau >= T[-1]), crashed=crashed, aborted=aborted,
                wall_s=float(L[-1, 0]),
                ref_s=float(T[-1]), wind=wind or None)
    (Path(mission_dir) / "flight.json").write_text(json.dumps(info, indent=1))
    return info


def _wait_ready(proc, log_path: Path, timeout: float = 240.0) -> bool:
    """True once PX4 reports 'Ready for takeoff'; False if it exits or fails to spawn.

    Big worlds (hundreds of boxes) load slowly, and PX4 then gives up on spawning the model
    ('gz_bridge failed to start'); a fresh launch usually works."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        if "Ready for takeoff" in text:
            return True
        # (on success PX4 also prints "Startup script returned successfully")
        if ("gz_bridge failed" in text or "Startup script returned with return value" in text
                or proc.poll() is not None):
            return False
        time.sleep(1.0)
    return False


def fly(mission_dir, px4=None, gui=False, wind=None, port=14540, attempts=3):
    m, _, _ = load_mission(mission_dir)
    root = px4_dir(px4)
    log_path = Path(mission_dir) / "px4.log"
    for attempt in range(1, attempts + 1):
        proc, log = launch(root, m, log_path, gui)
        if _wait_ready(proc, log_path):
            break
        shutdown(proc, log, m["world_name"])
        print("PX4/Gazebo launch attempt {} failed; retrying".format(attempt), flush=True)
    else:
        raise RuntimeError("PX4/Gazebo did not start after {} attempts".format(attempts))
    try:
        return asyncio.run(fly_async(mission_dir, port, wind))
    finally:
        shutdown(proc, log, m["world_name"])
