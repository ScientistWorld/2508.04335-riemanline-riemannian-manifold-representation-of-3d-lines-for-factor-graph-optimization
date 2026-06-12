from __future__ import annotations

import argparse
import math

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


def _line_from_params(theta: float, phi: float, gamma: float, log_omega: float) -> tuple[np.ndarray, np.ndarray]:
    direction = unit_from_angles(theta, phi)
    b1, b2 = orthonormal_basis(direction)
    normal = math.exp(log_omega) * (math.cos(gamma) * b1 + math.sin(gamma) * b2)
    return normal, direction


def _initial_parameters(dataset: dict) -> np.ndarray:
    params = []
    for i, line in enumerate(dataset["lines"]):
        d = normalize(np.asarray(line["direction"], dtype=float) + np.array([0.08, -0.05, 0.02]) * math.sin(i + 1))
        theta, phi = angles_from_unit(d)
        b1, b2 = orthonormal_basis(d)
        n = np.asarray(line["normal"], dtype=float)
        n = n - np.dot(n, d) * d
        gamma = math.atan2(float(np.dot(n, b2)), float(np.dot(n, b1))) + 0.24 * math.cos(i + 2)
        log_omega = math.log(max(np.linalg.norm(n), 1e-4)) + 0.15 * math.sin(2 * i + 1)
        params.extend([theta, phi, gamma, log_omega])
    return np.asarray(params, dtype=float)


def _decode(params: np.ndarray, num_lines: int) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for i in range(num_lines):
        theta, phi, gamma, log_omega = params[4 * i : 4 * i + 4]
        out.append(_line_from_params(float(theta), float(phi), float(gamma), float(log_omega)))
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
        "method": "orthline",
        "experiment": "scaled_synthetic_parallel_lines",
        "parameterization": "independent S2 direction plus scaled S1 normal per line",
        "num_parameters": int(4 * len(dataset["lines"])),
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
    parser.add_argument("--output", default="outputs/orthline.json")
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
