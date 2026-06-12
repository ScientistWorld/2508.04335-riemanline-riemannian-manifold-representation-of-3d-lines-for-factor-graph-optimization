from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from data.geometry import (
    angles_from_unit,
    camera_line,
    dump_json,
    load_json,
    normalize,
    orthonormal_basis,
    point_line_distance,
    unit_from_angles,
)


def _line_from_shared(direction: np.ndarray, gamma: float, log_omega: float) -> tuple[np.ndarray, np.ndarray]:
    b1, b2 = orthonormal_basis(direction)
    u1 = math.cos(gamma) * b1 + math.sin(gamma) * b2
    normal = math.exp(log_omega) * u1
    return normal, direction


def _initial_parameters(dataset: dict) -> np.ndarray:
    true_direction = normalize(np.asarray(dataset["lines"][0]["direction"], dtype=float))
    theta, phi = angles_from_unit(normalize(true_direction + np.array([0.09, -0.07, 0.03])))
    params = [theta, phi]
    init_direction = unit_from_angles(theta, phi)
    b1, b2 = orthonormal_basis(init_direction)
    for i, line in enumerate(dataset["lines"]):
        n = np.asarray(line["normal"], dtype=float)
        n = n - np.dot(n, init_direction) * init_direction
        gamma = math.atan2(float(np.dot(n, b2)), float(np.dot(n, b1))) + 0.20 * math.sin(i + 1)
        log_omega = math.log(max(np.linalg.norm(n), 1e-4)) + 0.12 * math.cos(2 * i + 1)
        params.extend([gamma, log_omega])
    return np.asarray(params, dtype=float)


def _decode(params: np.ndarray, num_lines: int) -> list[tuple[np.ndarray, np.ndarray]]:
    direction = unit_from_angles(float(params[0]), float(params[1]))
    out = []
    offset = 2
    for _ in range(num_lines):
        gamma, log_omega = params[offset : offset + 2]
        out.append(_line_from_shared(direction, float(gamma), float(log_omega)))
        offset += 2
    return out


def residuals(params: np.ndarray, dataset: dict) -> np.ndarray:
    cameras = {camera["id"]: camera for camera in dataset["cameras"]}
    lines = _decode(params, len(dataset["lines"]))
    residual = []
    for obs in dataset["observations"]:
        normal, direction = lines[obs["line_id"]]
        image_line = camera_line(normal, direction, cameras[obs["camera_id"]])
        residual.append(point_line_distance(np.asarray(obs["p0"], dtype=float), image_line))
        residual.append(point_line_distance(np.asarray(obs["p1"], dtype=float), image_line))
    return np.asarray(residual, dtype=float)


def run(data_path: str, output_path: str) -> None:
    dataset = load_json(data_path)
    x0 = _initial_parameters(dataset)
    before = residuals(x0, dataset)
    result = least_squares(
        lambda x: residuals(x, dataset),
        x0,
        loss="huber",
        f_scale=2.0,
        max_nfev=400,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    decoded = _decode(result.x, len(dataset["lines"]))
    payload = {
        "method": "struct_riemanline",
        "experiment": "scaled_synthetic_parallel_lines",
        "parameterization": "shared S2 direction plus one scaled S1 normal per line",
        "num_parameters": int(2 + 2 * len(dataset["lines"])),
        "num_observations": int(len(dataset["observations"])),
        "initial_reprojection_rmse_px": float(np.sqrt(np.mean(before**2))),
        "final_reprojection_rmse_px": float(np.sqrt(np.mean(residuals(result.x, dataset) ** 2))),
        "solver_cost": float(result.cost),
        "solver_nfev": int(result.nfev),
        "success": bool(result.success),
        "lines": [
            {"id": i, "normal": normal.tolist(), "direction": direction.tolist()}
            for i, (normal, direction) in enumerate(decoded)
        ],
    }
    dump_json(output_path, payload)
    print(f"wrote {output_path}: rmse {payload['final_reprojection_rmse_px']:.3f}px")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/riemanline_synthetic.json")
    parser.add_argument("--output", default="outputs/struct_riemanline.json")
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
