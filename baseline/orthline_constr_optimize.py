from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.optimize import least_squares

from baseline.orthline_optimize import _decode, _initial_parameters, residuals as reprojection_residuals
from data.geometry import dump_json, load_json, normalize


def _parallel_residuals(params: np.ndarray, num_lines: int, weight: float) -> np.ndarray:
    lines = _decode(params, num_lines)
    ref = normalize(lines[0][1])
    residual = []
    for _, direction in lines[1:]:
        direction = normalize(direction)
        if float(np.dot(ref, direction)) < 0.0:
            direction = -direction
        # Eq. 11 minimizes angular deviation between line directions.  The
        # cross product gives a smooth small-angle residual with the same zero.
        residual.extend((weight * np.cross(ref, direction)).tolist())
    return np.asarray(residual, dtype=float)


def residuals(params: np.ndarray, dataset: dict, parallel_weight: float) -> np.ndarray:
    return np.concatenate(
        [
            reprojection_residuals(params, dataset),
            _parallel_residuals(params, len(dataset["lines"]), parallel_weight),
        ]
    )


def run(data_path: str, output_path: str, parallel_weight: float) -> None:
    dataset = load_json(data_path)
    x0 = _initial_parameters(dataset)
    before = reprojection_residuals(x0, dataset)
    result = least_squares(
        lambda x: residuals(x, dataset, parallel_weight),
        x0,
        loss="huber",
        f_scale=2.0,
        max_nfev=400,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    decoded = _decode(result.x, len(dataset["lines"]))
    final_reprojection = reprojection_residuals(result.x, dataset)
    payload = {
        "method": "orthline_constr",
        "experiment": dataset.get("name", "scaled_synthetic_parallel_lines"),
        "parameterization": "independent S2/S1 lines with explicit parallel direction residuals",
        "num_parameters": int(4 * len(dataset["lines"])),
        "num_observations": int(len(dataset["observations"])),
        "parallel_weight": float(parallel_weight),
        "initial_reprojection_rmse_px": float(math.sqrt(np.mean(before**2))),
        "final_reprojection_rmse_px": float(math.sqrt(np.mean(final_reprojection**2))),
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
    parser.add_argument("--output", default="outputs/orthline_constr.json")
    parser.add_argument("--parallel-weight", type=float, default=35.0)
    args = parser.parse_args()
    run(args.data, args.output, args.parallel_weight)


if __name__ == "__main__":
    main()
