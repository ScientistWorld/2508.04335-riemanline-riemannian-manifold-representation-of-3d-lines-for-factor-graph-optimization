from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-12


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("cannot normalize near-zero vector")
    return v / n


def orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = normalize(axis)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, ref))) > 0.85:
        ref = np.array([0.0, 1.0, 0.0])
    b1 = normalize(ref - np.dot(ref, axis) * axis)
    b2 = normalize(np.cross(axis, b1))
    return b1, b2


def unit_from_angles(theta: float, phi: float) -> np.ndarray:
    c = math.cos(theta)
    return np.array([c * math.cos(phi), c * math.sin(phi), math.sin(theta)])


def angles_from_unit(v: np.ndarray) -> tuple[float, float]:
    v = normalize(v)
    theta = math.asin(float(np.clip(v[2], -1.0, 1.0)))
    phi = math.atan2(float(v[1]), float(v[0]))
    return theta, phi


def plucker_from_direction_normal(direction: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = normalize(direction)
    normal = np.asarray(normal, dtype=float)
    normal = normal - np.dot(normal, direction) * direction
    return normal, direction


def camera_line(normal_w: np.ndarray, direction_w: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    r_cw = np.asarray(camera["R_cw"], dtype=float)
    t_cw = np.asarray(camera["t_cw"], dtype=float)
    d_c = r_cw @ direction_w
    n_c = r_cw @ normal_w + np.cross(t_cw, d_c)
    return normalize(np.asarray(camera["K_inv_T"], dtype=float) @ n_c)


def point_line_distance(point_xy: np.ndarray, line_abc: np.ndarray) -> float:
    a, b, c = line_abc
    return float((a * point_xy[0] + b * point_xy[1] + c) / math.sqrt(a * a + b * b + EPS))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    # A 3D line has no signed direction, so compare up to sign.
    dot = abs(float(np.dot(a, b)))
    return math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))


def project_point(point_w: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    r_cw = np.asarray(camera["R_cw"], dtype=float)
    t_cw = np.asarray(camera["t_cw"], dtype=float)
    k = np.asarray(camera["K"], dtype=float)
    p_c = r_cw @ point_w + t_cw
    if p_c[2] <= 0.05:
        raise ValueError("point behind camera")
    uvw = k @ (p_c / p_c[2])
    return uvw[:2]


def look_at_r_cw(camera_center: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = normalize(target - camera_center)
    down_hint = np.array([0.0, 0.0, -1.0])
    right = normalize(np.cross(down_hint, forward))
    down = normalize(np.cross(forward, right))
    # Rows map world vectors into camera x, y, z axes.
    return np.vstack([right, down, forward])


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
