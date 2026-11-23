"""Frames, clearance and box merging.

Voxel index (z, y, x) has its centre at ((x+.5)s, (y+.5)s, (z+.5)s) in the Gazebo world
frame, which is ENU: E = x, N = y, U = z. PX4's local frame is NED about the drone's
rest pose.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


def vox2enu(p, s: float) -> np.ndarray:
    p = np.asarray(p, float)
    return np.stack([p[..., 2] + 0.5, p[..., 1] + 0.5, p[..., 0] + 0.5], axis=-1) * s


def enu2vox(q, s: float) -> np.ndarray:
    q = np.asarray(q, float) / s - 0.5
    return np.stack([q[..., 2], q[..., 1], q[..., 0]], axis=-1)


def enu2ned(d) -> np.ndarray:
    d = np.asarray(d, float)
    return np.stack([d[..., 1], d[..., 0], -d[..., 2]], axis=-1)


def ned2enu(d) -> np.ndarray:
    d = np.asarray(d, float)
    return np.stack([d[..., 1], d[..., 0], -d[..., 2]], axis=-1)


class Clearance:
    """Distance in metres from a point (voxel coords) to the nearest occupied voxel face.

    Linear interpolation of the voxel-centre EDT minus half a voxel: 0 on the face of an
    occupied voxel, negative inside one.
    """

    def __init__(self, occ: np.ndarray, voxel_m: float):
        self.edt = distance_transform_edt(occ == 0).astype(np.float32)
        self.s = float(voxel_m)

    def __call__(self, pts) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 3)
        v = map_coordinates(self.edt, pts.T, order=1, mode="nearest")
        return (v - 0.5) * self.s

    def along(self, A, B, step: float = 0.2) -> np.ndarray:
        """Minimum clearance along each straight segment A[i] -> B[i] (voxel coords)."""
        A = np.atleast_2d(np.asarray(A, float))
        B = np.atleast_2d(np.asarray(B, float))
        L = np.linalg.norm(B - A, axis=1)
        S = max(2, int(np.ceil(L.max() / step)) + 1) if len(L) else 2
        t = np.linspace(0.0, 1.0, S)
        P = A[:, None, :] + t[None, :, None] * (B - A)[:, None, :]
        return self(P.reshape(-1, 3)).reshape(len(A), S).min(axis=1)

    def polyline(self, P, step: float = 0.2) -> float:
        P = np.asarray(P, float)
        if len(P) < 2:
            return float(self(P).min())
        return float(self.along(P[:-1], P[1:], step).min())


def merge_boxes(occ: np.ndarray):
    """Greedy cover of the occupied voxels by axis-aligned boxes (z0, z1, y0, y1, x0, x1)."""
    occ = occ.astype(bool)
    used = np.zeros_like(occ)
    Z, Y, X = occ.shape
    boxes = []
    todo = occ.copy()
    for z, y, x in zip(*np.nonzero(todo)):
        if used[z, y, x]:
            continue
        x1 = x
        while x1 + 1 < X and occ[z, y, x1 + 1] and not used[z, y, x1 + 1]:
            x1 += 1
        y1 = y
        while y1 + 1 < Y and (occ[z, y1 + 1, x:x1 + 1] & ~used[z, y1 + 1, x:x1 + 1]).all():
            y1 += 1
        z1 = z
        while z1 + 1 < Z and (occ[z1 + 1, y:y1 + 1, x:x1 + 1]
                              & ~used[z1 + 1, y:y1 + 1, x:x1 + 1]).all():
            z1 += 1
        used[z:z1 + 1, y:y1 + 1, x:x1 + 1] = True
        boxes.append((int(z), int(z1), int(y), int(y1), int(x), int(x1)))
    return boxes


def ground_below(occ: np.ndarray, p) -> int:
    """Index of the first free voxel above the solid directly below voxel p."""
    z, y, x = (int(round(v)) for v in p)
    col = occ[:z + 1, y, x]
    solid = np.flatnonzero(col)
    return int(solid.max()) + 1 if solid.size else 0
