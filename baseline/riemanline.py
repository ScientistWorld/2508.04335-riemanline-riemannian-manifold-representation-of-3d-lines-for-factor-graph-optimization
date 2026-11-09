"""RiemanLine baseline: the paper's independent-line representation (Sec. 3).

Copy of the representation kept in baseline/ so the baseline never changes when
method/ is edited.  The CLI always runs with independent lines (no shared
directions, 4 DoF per line); the group code is kept only to share the update.

Original description:

Representation (Eq. 8/9).  A parallel-line group is stored as one unit
direction u2 on S^2 plus, for every member line, a unit normal direction u1 on
the circle orthogonal to u2 and a scale lambda > 0 (the line moment is
lambda * u1).  A group of k lines has 2 + 2k degrees of freedom; an individual
line is the k = 1 case (4 DoF), i.e. RiemanLine.

On-manifold update (Eq. 4-7), local coordinates [d_theta(2), (d_gamma, d_lambda) per line]:
    dm  = [b_x b_y] d_theta                                  (Eq. 4)
    v2  = u2 cos|dm| + dm/|dm| sin|dm|                         (Eq. 5)
    u3  = v2 x u1                                             (u1 re-projected onto the plane of v2)
    v1  = cos(d_gamma) u1 + sin(d_gamma) u3,   lambda += d_lambda   (Eq. 7)

Factors: the line reprojection factor of Eq. 10 for every observation, with
the authors' Cauchy loss (3 px).  Parallel groups are taken from the scene's
declared associations after the authors' inlier check (seeded consensus with
|cos| > 0.9; a group is used only with more than two inliers), exactly as in
the released ``CovisPointParaRiemanLine`` optimiser.  The initial factor graph
is a linear multi-view triangulation; a group's direction is initialised to
the principal direction of its members.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baseline.line_ba import (
    Observations,
    dump_json,
    levenberg_marquardt,
    load_json,
    reprojection_residuals,
    tangent_basis,
    triangulate_lines,
    unit,
)

INLIER_COS = 0.9
CAUCHY_PX = 3.0
MAX_ITERATIONS = 100


@dataclass
class LineGroup:
    """A shared direction u2 and per-line scaled normals lambda_i * u1_i."""

    lines: np.ndarray  # (k,) line indices
    u2: np.ndarray  # (3,)
    u1: np.ndarray  # (k, 3)
    lam: np.ndarray  # (k,)

    def __post_init__(self) -> None:
        self.basis = tangent_basis(self.u2)

    @property
    def dof(self) -> int:
        return 2 + 2 * len(self.lines)


def plus_group(g: LineGroup, delta: np.ndarray) -> LineGroup:
    """Eq. 4-7 retraction of one group by its local coordinates ``delta``."""
    b1, b2 = g.basis
    dm = delta[0] * b1 + delta[1] * b2
    norm = float(np.linalg.norm(dm))
    v2 = g.u2 if norm < 1e-15 else g.u2 * math.cos(norm) + dm / norm * math.sin(norm)
    v2 = unit(v2)
    u1 = g.u1 - (g.u1 @ v2)[:, None] * v2[None, :]
    u1 = u1 / np.linalg.norm(u1, axis=1, keepdims=True)
    u3 = np.cross(v2[None, :], u1)
    d_gamma = delta[2::2]
    d_lambda = delta[3::2]
    v1 = np.cos(d_gamma)[:, None] * u1 + np.sin(d_gamma)[:, None] * u3
    lam = np.maximum(g.lam + d_lambda, 1e-6)
    return LineGroup(g.lines, v2, v1, lam)


class GroupedLines:
    """State of all lines as a list of groups (singletons for individual lines)."""

    def __init__(self, groups: list[LineGroup], num_lines: int):
        self.groups = groups
        self.num_lines = num_lines
        self.offsets = np.cumsum([0] + [g.dof for g in groups])

    def plus(self, delta: np.ndarray) -> "GroupedLines":
        new = [plus_group(g, delta[self.offsets[i] : self.offsets[i + 1]]) for i, g in enumerate(self.groups)]
        return GroupedLines(new, self.num_lines)

    def decode(self) -> tuple[np.ndarray, np.ndarray]:
        normals = np.zeros((self.num_lines, 3))
        directions = np.zeros((self.num_lines, 3))
        for g in self.groups:
            directions[g.lines] = g.u2
            normals[g.lines] = g.lam[:, None] * g.u1
        return normals, directions

    def column_lines(self) -> list[np.ndarray]:
        cols: list[np.ndarray] = []
        for g in self.groups:
            cols.extend([g.lines, g.lines])
            for line in g.lines:
                cols.extend([np.array([line]), np.array([line])])
        return cols

    @property
    def dof(self) -> int:
        return int(self.offsets[-1])


def group_from_lines(lines: list[int], normals: np.ndarray, directions: np.ndarray, u2: np.ndarray | None = None) -> LineGroup:
    """Build a group with shared direction u2 from per-line Plucker estimates."""
    idx = np.asarray(lines, dtype=int)
    if u2 is None:
        d = directions[idx]
        scatter = d.T @ d
        u2 = np.linalg.eigh(scatter)[1][:, -1]
    u2 = unit(u2)
    u1 = np.zeros((len(idx), 3))
    lam = np.zeros(len(idx))
    for k, i in enumerate(idx):
        di = unit(directions[i])
        point = np.cross(di, normals[i])  # closest point of the line to the origin
        point = point - np.dot(point, u2) * u2
        n = np.cross(point, u2)
        s = float(np.linalg.norm(n))
        if s < 1e-9:
            u1[k] = tangent_basis(u2)[0]
            lam[k] = 1e-3
        else:
            u1[k] = n / s
            lam[k] = s
    return LineGroup(idx, u2, u1, lam)


def parallel_inliers(members: list[int], directions: np.ndarray, cos_threshold: float = INLIER_COS) -> list[int]:
    """Deterministic version of the authors' seeded consensus (every member as seed)."""
    best: list[int] = []
    for seed in members:
        ds = unit(directions[seed])
        inl = [j for j in members if abs(float(np.dot(unit(directions[j]), ds))) > cos_threshold]
        if len(inl) > len(best):
            best = inl
    return best


def initial_state(scene: dict[str, Any], obs: Observations, use_groups: bool = True) -> GroupedLines:
    normals, directions = triangulate_lines(obs)
    grouped: set[int] = set()
    groups: list[LineGroup] = []
    if use_groups:
        for members_ids in scene.get("parallel_groups", []):
            members = [obs.index[int(m)] for m in members_ids if int(m) in obs.index and obs.index[int(m)] not in grouped]
            inl = parallel_inliers(members, directions)
            if len(inl) > 2:
                groups.append(group_from_lines(inl, normals, directions))
                grouped.update(inl)
    for i in range(obs.num_lines):
        if i not in grouped:
            groups.append(group_from_lines([i], normals, directions, u2=directions[i]))
    return GroupedLines(groups, obs.num_lines)


def optimize(state: GroupedLines, obs: Observations, loss: str = "cauchy", scale: float = CAUCHY_PX,
             max_iterations: int = MAX_ITERATIONS) -> tuple[GroupedLines, dict[str, Any]]:
    def residual(s: GroupedLines) -> np.ndarray:
        return reprojection_residuals(obs, *s.decode())

    row_lines = [(int(l),) for l in obs.row_line]
    return levenberg_marquardt(
        state,
        lambda s, d: s.plus(d),
        residual,
        state.column_lines(),
        row_lines,
        obs.num_lines,
        loss=loss,
        scale=scale,
        max_iterations=max_iterations,
    )


def payload(scene: dict[str, Any], obs: Observations, state: GroupedLines, method: str, info: dict[str, Any]) -> dict[str, Any]:
    normals, directions = state.decode()
    return {
        "scene_id": scene.get("scene_id"),
        "method": method,
        "num_parameters": state.dof,
        "solver": info,
        "lines": [
            {"id": obs.line_ids[i], "normal": normals[i].tolist(), "direction": directions[i].tolist()}
            for i in range(obs.num_lines)
        ],
    }


def solve_scene(scene: dict[str, Any], use_groups: bool = True, loss: str = "cauchy",
                scale: float = CAUCHY_PX, max_iterations: int = MAX_ITERATIONS) -> dict[str, Any]:
    obs = Observations(scene)
    state = initial_state(scene, obs, use_groups=use_groups)
    state, info = optimize(state, obs, loss=loss, scale=scale, max_iterations=max_iterations)
    return payload(scene, obs, state, "struct_riemanline" if use_groups else "riemanline", info)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent-line RiemanLine on every scene JSON in a directory.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    args.no_groups = True
    args.loss = "cauchy"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.input_dir).glob("*.json")):
        scene = load_json(path)
        result = solve_scene(scene, use_groups=not args.no_groups, loss=args.loss)
        dump_json(out / path.name, result)
        print(f"{path.name}: cost {result['solver']['initial_cost']:.1f} -> {result['solver']['final_cost']:.1f} "
              f"in {result['solver']['iterations']} it, {result['num_parameters']} params", flush=True)


if __name__ == "__main__":
    main()
