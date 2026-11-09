"""Line bundle-adjustment building blocks used by the OrthLine baselines (copy kept
independent of method/ so that baselines never change with the method).

* ``Observations``: vectorised view of a scene's 2D segment observations.
* ``reprojection_residuals``: paper Eq. 10, the signed distances (pixels) of
  both observed endpoints to the projection of a 3D Plucker line.
* ``triangulate_lines``: linear multi-view line triangulation (intersection of
  back-projected planes) used as the initial factor graph.
* ``levenberg_marquardt``: a Ceres-style LM solver for states that live on a
  manifold.  Every step is applied through the caller's ``plus`` retraction
  (the paper's on-manifold update), the Jacobian is taken with respect to the
  local (tangent-space) coordinates by finite differences, and columns that
  touch disjoint lines are perturbed together (graph colouring).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

EPS = 1e-12


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def tangent_basis(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal basis (b_x, b_y) of the tangent plane of S^2 at v."""
    v = unit(v)
    ref = np.zeros(3)
    ref[int(np.argmin(np.abs(v)))] = 1.0
    b1 = unit(ref - np.dot(ref, v) * v)
    b2 = np.cross(v, b1)
    return b1, b2


class Observations:
    """Vectorised observations of one scene (optionally restricted to some lines)."""

    def __init__(self, scene: dict[str, Any], keep_lines: Sequence[int] | None = None):
        cams = {int(c["id"]): c for c in scene["cameras"]}
        self.line_ids = [int(line["id"]) for line in scene["lines"]]
        self.index = {lid: i for i, lid in enumerate(self.line_ids)}
        obs = scene["observations"]
        if keep_lines is not None:
            keep = {self.line_ids[i] for i in keep_lines}
            obs = [o for o in obs if int(o["line_id"]) in keep]
        self.num_lines = len(self.line_ids)
        self.line = np.array([self.index[int(o["line_id"])] for o in obs], dtype=int)
        self.R = np.array([cams[int(o["camera_id"])]["R_cw"] for o in obs], dtype=float).reshape(-1, 3, 3)
        self.t = np.array([cams[int(o["camera_id"])]["t_cw"] for o in obs], dtype=float).reshape(-1, 3)
        k = np.array([cams[int(o["camera_id"])]["K"] for o in obs], dtype=float).reshape(-1, 3, 3)
        self.K = k
        self.KinvT = np.transpose(np.linalg.inv(k), (0, 2, 1)) if len(obs) else k
        self.p0 = np.array([o["p0"] for o in obs], dtype=float).reshape(-1, 2)
        self.p1 = np.array([o["p1"] for o in obs], dtype=float).reshape(-1, 2)
        self.row_line = np.repeat(self.line, 2)
        self.fx = float(np.median(k[:, 0, 0])) if len(obs) else 1.0


def reprojection_residuals(obs: Observations, normals: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Eq. 10: endpoint-to-projected-line distances in pixels, two rows per observation."""
    d = directions[obs.line]
    n = normals[obs.line]
    dc = np.einsum("mij,mj->mi", obs.R, d)
    nc = np.einsum("mij,mj->mi", obs.R, n) + np.cross(obs.t, dc)
    l = np.einsum("mij,mj->mi", obs.KinvT, nc)
    den = np.sqrt(l[:, 0] ** 2 + l[:, 1] ** 2) + EPS
    r0 = (l[:, 0] * obs.p0[:, 0] + l[:, 1] * obs.p0[:, 1] + l[:, 2]) / den
    r1 = (l[:, 0] * obs.p1[:, 0] + l[:, 1] * obs.p1[:, 1] + l[:, 2]) / den
    return np.stack([r0, r1], axis=1).ravel()


def triangulate_lines(obs: Observations) -> tuple[np.ndarray, np.ndarray]:
    """Linear triangulation: each observed segment back-projects to a plane
    pi = P^T l; the 3D line spans the 2D null space of the stacked planes."""
    normals = np.zeros((obs.num_lines, 3))
    directions = np.zeros((obs.num_lines, 3))
    h0 = np.concatenate([obs.p0, np.ones((len(obs.p0), 1))], axis=1)
    h1 = np.concatenate([obs.p1, np.ones((len(obs.p1), 1))], axis=1)
    img_lines = np.cross(h0, h1)
    kt_l = np.einsum("mji,mj->mi", obs.K, img_lines)  # K^T l
    planes = np.concatenate(
        [np.einsum("mji,mj->mi", obs.R, kt_l), np.einsum("mi,mi->m", obs.t, kt_l)[:, None]], axis=1
    )
    planes /= np.linalg.norm(planes[:, :3], axis=1, keepdims=True) + EPS
    for i in range(obs.num_lines):
        rows = planes[obs.line == i]
        if len(rows) < 2:
            directions[i] = np.array([0.0, 0.0, 1.0])
            normals[i] = np.array([1.0, 0.0, 0.0])
            continue
        _, _, vt = np.linalg.svd(rows)
        x1, x2 = vt[-1], vt[-2]
        d = x1[3] * x2[:3] - x2[3] * x1[:3]
        n = np.cross(x1[:3], x2[:3])
        s = float(np.linalg.norm(d))
        if s < 1e-12:
            d = unit(np.cross(rows[0, :3], rows[1, :3]))
            n = np.cross(np.zeros(3), d)
        else:
            d, n = d / s, n / s
        n = n - np.dot(n, d) * d
        directions[i] = d
        normals[i] = n
    return normals, directions


# ----------------------------------------------------------------- solver

def robust_cost(r: np.ndarray, loss: str, scale: float) -> tuple[float, np.ndarray]:
    """Return (sum rho(r), IRLS weights rho'(r^2)) for the chosen loss."""
    s = r * r
    if loss == "cauchy":
        c2 = scale * scale
        return float(0.5 * c2 * np.sum(np.log1p(s / c2))), 1.0 / (1.0 + s / c2)
    if loss == "huber":
        a = np.abs(r)
        inlier = a <= scale
        rho = np.where(inlier, 0.5 * s, scale * a - 0.5 * scale * scale)
        return float(np.sum(rho)), np.where(inlier, 1.0, scale / np.maximum(a, EPS))
    if loss == "linear":
        return float(0.5 * np.sum(s)), np.ones_like(r)
    raise ValueError(f"unknown loss {loss}")


def _colouring(col_lines: Sequence[np.ndarray], row_lines: Sequence[Sequence[int]], num_lines: int):
    rows_of_line: list[list[int]] = [[] for _ in range(num_lines)]
    for r, lines in enumerate(row_lines):
        for l in lines:
            rows_of_line[int(l)].append(r)
    n_rows = len(row_lines)
    col_rows = []
    for lines in col_lines:
        parts = [np.asarray(rows_of_line[int(l)], dtype=int) for l in lines]
        col_rows.append(np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=int))
    colours: list[dict[str, Any]] = []
    for c, rows in enumerate(col_rows):
        for colour in colours:
            if not colour["used"][rows].any():
                colour["cols"].append(c)
                colour["used"][rows] = True
                break
        else:
            used = np.zeros(n_rows, dtype=bool)
            used[rows] = True
            colours.append({"cols": [c], "used": used})
    packed = []
    for colour in colours:
        cols = np.asarray(colour["cols"], dtype=int)
        rows_idx = np.concatenate([col_rows[c] for c in cols])
        cols_idx = np.concatenate([np.full(len(col_rows[c]), c, dtype=int) for c in cols])
        packed.append((cols, rows_idx, cols_idx))
    return packed


def levenberg_marquardt(
    state: Any,
    plus: Callable[[Any, np.ndarray], Any],
    residual: Callable[[Any], np.ndarray],
    col_lines: Sequence[np.ndarray],
    row_lines: Sequence[Sequence[int]],
    num_lines: int,
    loss: str = "cauchy",
    scale: float = 3.0,
    max_iterations: int = 100,
    step_eps: float = 1e-6,
    function_tolerance: float = 1e-10,
    gradient_tolerance: float = 1e-10,
) -> tuple[Any, dict[str, Any]]:
    num_params = len(col_lines)
    r = residual(state)
    cost, w = robust_cost(r, loss, scale)
    info: dict[str, Any] = {"initial_cost": cost, "iterations": 0, "num_local_parameters": num_params}
    if num_params == 0 or len(r) == 0:
        info["final_cost"] = cost
        return state, info
    colours = _colouring(col_lines, row_lines, num_lines)
    mu = 1e-4
    for it in range(max_iterations):
        jac = np.zeros((len(r), num_params))
        for cols, rows_idx, cols_idx in colours:
            delta = np.zeros(num_params)
            delta[cols] = step_eps
            r1 = residual(plus(state, delta))
            jac[rows_idx, cols_idx] = (r1[rows_idx] - r[rows_idx]) / step_eps
        jw = jac * w[:, None]
        hess = jac.T @ jw
        grad = jw.T @ r
        if float(np.max(np.abs(grad))) < gradient_tolerance:
            break
        diag = np.maximum(np.diag(hess), 1e-9)
        accepted = False
        while mu < 1e12:
            try:
                step = -np.linalg.solve(hess + mu * np.diag(diag), grad)
            except np.linalg.LinAlgError:
                mu *= 4.0
                continue
            cand = plus(state, step)
            r_new = residual(cand)
            if np.all(np.isfinite(r_new)):
                cost_new, w_new = robust_cost(r_new, loss, scale)
                if cost_new < cost:
                    accepted = True
                    break
            mu *= 4.0
        if not accepted:
            break
        decrease = cost - cost_new
        state, r, w, cost = cand, r_new, w_new, cost_new
        info["iterations"] = it + 1
        mu = max(mu / 3.0, 1e-12)
        if decrease <= function_tolerance * max(cost, 1e-300) or float(np.linalg.norm(step)) < 1e-12:
            break
    info["final_cost"] = cost
    return state, info
