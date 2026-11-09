"""OrthLine baselines (paper Sec. 3 "Preliminary" and Sec. 4 constraints).

Every line is an independent landmark in the orthonormal representation of
Bartoli & Sturm (2005): (U, W) in SO(3) x SO(2) with
    U = [n/|n|, d/|d|, (n x d)/|n x d|],  W = [[w1, -w2], [w2, w1]],
    w1 = |n| / lambda, w2 = |d| / lambda,  lambda = sqrt(|n|^2 + |d|^2)   (Eq. 1-2)
and the 4-DoF update U <- U Rx(d0) Ry(d1) Rz(d2), W <- W R(d3) used by the
authors' ``LineOrthParameterization``.  Reprojection factors are Eq. 10 with
the authors' Cauchy loss (3 px).

``--constrained`` adds the paper's parallelism residual (Eq. 11) for every
line of a declared parallel group,
    r_i = w / (N_i) * sum_j (1 - |u2_i . u2_j|),
over the group members whose initial direction satisfies |cos| > 0.92 (the
authors' gating).  w = f_x converts the authors' normalised-image-coordinate
residual units into the pixel units used here.  The state stays 4n-dimensional.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from baseline.line_ba import (
    Observations,
    dump_json,
    levenberg_marquardt,
    load_json,
    reprojection_residuals,
    triangulate_lines,
)

CAUCHY_PX = 3.0
MAX_ITERATIONS = 100
PAIR_COS = 0.92


class OrthLines:
    def __init__(self, u: np.ndarray, phi: np.ndarray):
        self.u = u  # (L, 3, 3) columns u1 (normal), u2 (direction), u3
        self.phi = phi  # (L,)

    @classmethod
    def from_plucker(cls, normals: np.ndarray, directions: np.ndarray) -> "OrthLines":
        num = len(normals)
        u = np.zeros((num, 3, 3))
        phi = np.zeros(num)
        for i in range(num):
            d = directions[i] / np.linalg.norm(directions[i])
            n = normals[i] - np.dot(normals[i], d) * d
            nn = float(np.linalg.norm(n))
            if nn < 1e-9:
                ref = np.zeros(3)
                ref[int(np.argmin(np.abs(d)))] = 1.0
                n = np.cross(d, ref)
                n = 1e-3 * n / np.linalg.norm(n)
                nn = 1e-3
            u1 = n / nn
            u3 = np.cross(u1, d)
            u[i] = np.column_stack([u1, d, u3 / np.linalg.norm(u3)])
            phi[i] = np.arctan2(1.0, nn)  # w1 = cos(phi) ~ |n|, w2 = sin(phi) ~ |d| = 1
        return cls(u, phi)

    def plus(self, delta: np.ndarray) -> "OrthLines":
        d = delta.reshape(-1, 4)
        ca, sa = np.cos(d[:, 0]), np.sin(d[:, 0])
        cb, sb = np.cos(d[:, 1]), np.sin(d[:, 1])
        cg, sg = np.cos(d[:, 2]), np.sin(d[:, 2])
        num = len(d)
        rx = np.zeros((num, 3, 3))
        rx[:, 0, 0] = 1.0
        rx[:, 1, 1], rx[:, 1, 2], rx[:, 2, 1], rx[:, 2, 2] = ca, -sa, sa, ca
        ry = np.zeros((num, 3, 3))
        ry[:, 1, 1] = 1.0
        ry[:, 0, 0], ry[:, 0, 2], ry[:, 2, 0], ry[:, 2, 2] = cb, sb, -sb, cb
        rz = np.zeros((num, 3, 3))
        rz[:, 2, 2] = 1.0
        rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1] = cg, -sg, sg, cg
        return OrthLines(self.u @ rx @ ry @ rz, self.phi + d[:, 3])

    def decode(self) -> tuple[np.ndarray, np.ndarray]:
        w1 = np.cos(self.phi)
        w2 = np.sin(self.phi)
        w2 = np.where(np.abs(w2) < 1e-9, 1e-9, w2)
        directions = self.u[:, :, 1] * np.sign(w2)[:, None]
        normals = self.u[:, :, 0] * (w1 / np.abs(w2))[:, None]
        return normals, directions


def parallel_pairs(scene: dict[str, Any], obs: Observations, directions: np.ndarray) -> list[tuple[int, list[int]]]:
    rows = []
    for members_ids in scene.get("parallel_groups", []):
        members = [obs.index[int(m)] for m in members_ids if int(m) in obs.index]
        for i in members:
            di = directions[i] / np.linalg.norm(directions[i])
            partners = [
                j for j in members
                if j != i and abs(float(np.dot(di, directions[j] / np.linalg.norm(directions[j])))) > PAIR_COS
            ]
            if partners:
                rows.append((i, partners))
    return rows


def solve_scene(scene: dict[str, Any], constrained: bool = False, loss: str = "cauchy") -> dict[str, Any]:
    obs = Observations(scene)
    normals, directions = triangulate_lines(obs)
    state = OrthLines.from_plucker(normals, directions)
    pairs = parallel_pairs(scene, obs, directions) if constrained else []
    weight = obs.fx

    def residual(s: OrthLines) -> np.ndarray:
        n, d = s.decode()
        reproj = reprojection_residuals(obs, n, d)
        if not pairs:
            return reproj
        par = np.empty(len(pairs))
        for k, (i, partners) in enumerate(pairs):
            par[k] = weight * float(np.mean(1.0 - np.abs(d[partners] @ d[i])))
        return np.concatenate([reproj, par])

    row_lines = [(int(l),) for l in obs.row_line] + [tuple([i] + partners) for i, partners in pairs]
    col_lines = [np.array([i]) for i in range(obs.num_lines) for _ in range(4)]
    state, info = levenberg_marquardt(
        state,
        lambda s, delta: s.plus(delta),
        residual,
        col_lines,
        row_lines,
        obs.num_lines,
        loss=loss,
        scale=CAUCHY_PX,
        max_iterations=MAX_ITERATIONS,
    )
    n, d = state.decode()
    return {
        "scene_id": scene.get("scene_id"),
        "method": "orthline_constr" if constrained else "orthline",
        "num_parameters": 4 * obs.num_lines,
        "solver": info,
        "lines": [
            {"id": obs.line_ids[i], "normal": n[i].tolist(), "direction": d[i].tolist()}
            for i in range(obs.num_lines)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OrthLine on every scene JSON in a directory.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--constrained", action="store_true", help="add the Eq. 11 parallelism residuals")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.input_dir).glob("*.json")):
        result = solve_scene(load_json(path), constrained=args.constrained)
        dump_json(out / path.name, result)
        print(f"{path.name}: cost {result['solver']['initial_cost']:.1f} -> {result['solver']['final_cost']:.1f} "
              f"in {result['solver']['iterations']} it", flush=True)


if __name__ == "__main__":
    main()
