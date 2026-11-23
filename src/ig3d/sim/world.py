"""A voxel volume as a Gazebo (gz Harmonic) world that PX4 SITL can fly in.

The world carries the system plugins PX4's own worlds load (IMU, baro, magnetometer,
NavSat, ...) so the stock gz_x500 airframe works unchanged. Occupied voxels become a few
merged static boxes; the outer shell is drawn see-through so the GUI camera can look in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import merge_boxes

WORLD_NAME = "ig3d"

_HEADER = """<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="{name}">
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <grid>false</grid>
      <ambient>0.5 0.5 0.5 1</ambient>
      <background>0.75 0.8 0.85 1</background>
      <shadows>false</shadows>
    </scene>
    <plugin name="gz::sim::systems::Physics" filename="gz-sim-physics-system"/>
    <plugin name="gz::sim::systems::UserCommands" filename="gz-sim-user-commands-system"/>
    <plugin name="gz::sim::systems::SceneBroadcaster" filename="gz-sim-scene-broadcaster-system"/>
    <plugin name="gz::sim::systems::Contact" filename="gz-sim-contact-system"/>
    <plugin name="gz::sim::systems::Imu" filename="gz-sim-imu-system"/>
    <plugin name="gz::sim::systems::AirPressure" filename="gz-sim-air-pressure-system"/>
    <plugin name="gz::sim::systems::AirSpeed" filename="gz-sim-air-speed-system"/>
    <plugin name="gz::sim::systems::ApplyLinkWrench" filename="gz-sim-apply-link-wrench-system"/>
    <plugin name="gz::sim::systems::NavSat" filename="gz-sim-navsat-system"/>
    <plugin name="gz::sim::systems::Magnetometer" filename="gz-sim-magnetometer-system"/>
    <plugin name="gz::sim::systems::Sensors" filename="gz-sim-sensors-system">
      <render_engine>ogre2</render_engine>
    </plugin>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 200 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.9</direction>
    </light>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg>
      <longitude_deg>8.546163739800146</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>
"""

_FOOTER = """  </world>
</sdf>
"""


def _colour(kind: str, frac: float):
    if kind == "floor":
        return (0.35, 0.37, 0.33, 0.0)
    if kind == "shell":
        return (0.8, 0.85, 0.9, 0.88)
    g = 0.45 + 0.35 * frac
    return (g, g * 0.97, g * 0.92, 0.0)


def _box_xml(i, box, s, kind, frac):
    z0, z1, y0, y1, x0, x1 = box
    cx, cy, cz = 0.5 * (x0 + x1 + 1) * s, 0.5 * (y0 + y1 + 1) * s, 0.5 * (z0 + z1 + 1) * s
    sx, sy, sz = (x1 - x0 + 1) * s, (y1 - y0 + 1) * s, (z1 - z0 + 1) * s
    r, g, b, alpha = _colour(kind, frac)
    geom = "<geometry><box><size>{:.3f} {:.3f} {:.3f}</size></box></geometry>".format(sx, sy, sz)
    pose = "<pose>{:.3f} {:.3f} {:.3f} 0 0 0</pose>".format(cx, cy, cz)
    mat = ("<material><ambient>{0:.2f} {1:.2f} {2:.2f} 1</ambient>"
           "<diffuse>{0:.2f} {1:.2f} {2:.2f} 1</diffuse></material>").format(r, g, b)
    vis = ("<visual name='v{i}'>{pose}{geom}{mat}<cast_shadows>false</cast_shadows>"
           "{tr}</visual>").format(i=i, pose=pose, geom=geom, mat=mat,
                                   tr="<transparency>{:.2f}</transparency>".format(alpha)
                                   if alpha else "")
    col = "<collision name='c{i}'>{pose}{geom}</collision>".format(i=i, pose=pose, geom=geom)
    return "        " + col + "\n        " + vis + "\n"


def _classify(box, shape, border=1):
    Z, Y, X = shape
    z0, z1, y0, y1, x0, x1 = box
    if z1 < border:
        return "floor"
    if z0 >= Z - border or y1 < border or y0 >= Y - border or x1 < border or x0 >= X - border:
        return "shell"
    return "solid"


def _markers(name, pts_enu, rgb, radius):
    if pts_enu is None or len(pts_enu) == 0:
        return ""
    vis = []
    for i, p in enumerate(pts_enu):
        vis.append("        <visual name='m{}'><pose>{:.3f} {:.3f} {:.3f} 0 0 0</pose>"
                   "<geometry><sphere><radius>{}</radius></sphere></geometry>"
                   "<material><ambient>{} {} {} 1</ambient><diffuse>{} {} {} 1</diffuse>"
                   "<emissive>{} {} {} 1</emissive></material>"
                   "<cast_shadows>false</cast_shadows></visual>\n".format(
                       i, *p, radius, *rgb, *rgb, *rgb))
    return ("    <model name='{}'><static>true</static><link name='link'>\n{}"
            "    </link></model>\n").format(name, "".join(vis))


def write_world(occ: np.ndarray, voxel_m: float, path, ref_enu=None, start_enu=None,
                goal_enu=None, name: str = WORLD_NAME) -> Path:
    """Write the SDF. ref_enu: optional reference path drawn as small spheres (no collision)."""
    boxes = merge_boxes(occ)
    Z = occ.shape[0]
    body = ["    <model name='ig3d_env'><static>true</static><link name='link'>\n"]
    for i, b in enumerate(boxes):
        body.append(_box_xml(i, b, voxel_m, _classify(b, occ.shape), b[1] / max(1, Z - 1)))
    body.append("    </link></model>\n")
    if ref_enu is not None:
        ref = np.asarray(ref_enu)
        step = max(1, len(ref) // 400)
        body.append(_markers("ig3d_path", ref[::step], (0.1, 0.5, 1.0), 0.06 * voxel_m))
    if start_enu is not None:
        body.append(_markers("ig3d_start", [start_enu], (0.1, 0.8, 0.2), 0.25 * voxel_m))
    if goal_enu is not None:
        body.append(_markers("ig3d_goal", [goal_enu], (0.9, 0.15, 0.1), 0.25 * voxel_m))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER.format(name=name) + "".join(body) + _FOOTER)
    return path
