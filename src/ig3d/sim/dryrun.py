"""PX4-free stand-in: a point mass with a PD tracking controller and acceleration limit.

Not a substitute for SITL. It exists so the whole pipeline (mission -> fly -> analyse) runs
and can be tested on any machine, and as a sanity baseline for the PX4 numbers.
Writes the same flight.npz / flight.json as `px4.fly`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mission import load_mission
from .px4 import CDA, RHO

MASS = 2.0


def fly(mission_dir, wind=None, kp=4.0, kd=3.0, a_max=8.0, dt=0.005, lag_ok=1.0, lag_span=1.5,
        seed=0):
    m, _, ref = load_mission(mission_dir)
    T, P, V, A = ref["t"], ref["pos"], ref["vel"], ref["acc"]
    rate = m["limits"]["rate_hz"]
    rng = np.random.default_rng(seed)
    p, v = P[0].copy(), np.zeros(3)
    tau, t, next_log, next_gust = 0.0, 0.0, 0.0, 0.0
    f_wind = np.zeros(3)
    log, wind_log = [], []
    while tau < T[-1] and t < 4 * T[-1] + 60:
        if wind and wind.get("speed", 0) > 0 and t >= next_gust:
            s = max(0.0, wind["speed"] * (1 + wind["gust"] * rng.standard_normal()))
            th = np.radians(wind["dir"])
            f = 0.5 * RHO * CDA * s * s
            f_wind = np.array([f * np.cos(th), f * np.sin(th), 0.0])
            wind_log.append((t, s, f_wind[0], f_wind[1]))
            next_gust += wind["period"]
        i = min(len(T) - 1, int(np.searchsorted(T, tau)))
        lag = float(np.linalg.norm(p - P[i]))
        scale = float(np.clip(1.0 - (lag - lag_ok) / lag_span, 0.0, 1.0))
        tau += dt * scale
        i = min(len(T) - 1, int(np.searchsorted(T, tau)))
        a = kp * (P[i] - p) + kd * (V[i] * scale - v) + A[i] * scale
        n = np.linalg.norm(a)
        if n > a_max:
            a *= a_max / n
        a = a + f_wind / MASS
        v = v + a * dt
        p = p + v * dt
        if t >= next_log:
            log.append((t, tau, ref["phase"][i], *p, *v, lag))
            next_log += 1.0 / rate
        t += dt
    L = np.asarray(log, float)
    np.savez_compressed(Path(mission_dir) / "flight.npz", t=L[:, 0], tau=L[:, 1], phase=L[:, 2],
                        pos_enu=L[:, 3:6], vel_enu=L[:, 6:9], lag=L[:, 9],
                        wind=np.asarray(wind_log, float).reshape(-1, 4))
    info = dict(backend="dryrun", origin_enu=P[0].tolist(), origin_source="reference",
                completed=bool(tau >= T[-1]), crashed=False, wall_s=float(t), ref_s=float(T[-1]),
                wind=wind or None)
    (Path(mission_dir) / "flight.json").write_text(json.dumps(info, indent=1))
    return info
