from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np

from data.geometry import (
    angle_deg,
    camera_line,
    dump_json,
    load_json,
    normalize,
    point_line_distance,
)


def _parallel_spread_deg(directions: list[np.ndarray]) -> float:
    if len(directions) < 2:
        return 0.0
    angles = [angle_deg(a, b) for a, b in itertools.combinations(directions, 2)]
    return float(np.mean(angles))


def _subset_reprojection_errors(
    dataset: dict[str, Any],
    lines_by_id: dict[int, dict[str, Any]],
) -> tuple[float, float]:
    cameras = {int(camera["id"]): camera for camera in dataset["cameras"]}
    squared_errors: list[float] = []
    endpoint_errors: list[float] = []
    for obs in dataset["observations"]:
        line = lines_by_id[int(obs["line_id"])]
        normal = np.asarray(line["normal"], dtype=float)
        direction = normalize(np.asarray(line["direction"], dtype=float))
        image_line = camera_line(normal, direction, cameras[int(obs["camera_id"])])
        for key in ("p0", "p1"):
            distance = point_line_distance(np.asarray(obs[key], dtype=float), image_line)
            squared_errors.append(distance * distance)
            endpoint_errors.append(abs(distance))
    if not squared_errors:
        raise ValueError("dataset has no scored line observations")
    return float(np.sqrt(np.mean(squared_errors))), float(np.percentile(endpoint_errors, 90))


def _score_one(dataset: dict[str, Any], output: dict[str, Any]) -> dict[str, float]:
    gt_lines = {int(line["id"]): line for line in dataset["lines"]}
    dir_errors = []
    normal_errors = []
    directions = []
    out_lines: dict[int, dict[str, Any]] = {}
    for line in output["lines"]:
        line_id = int(line["id"])
        gt = gt_lines[line_id]
        est_d = normalize(np.asarray(line["direction"], dtype=float))
        est_n = normalize(np.asarray(line["normal"], dtype=float))
        gt_d = normalize(np.asarray(gt["direction"], dtype=float))
        gt_n = normalize(np.asarray(gt["normal"], dtype=float))
        dir_errors.append(angle_deg(est_d, gt_d))
        normal_errors.append(angle_deg(est_n, gt_n))
        directions.append(est_d)
        out_lines[line_id] = line

    missing = sorted(set(gt_lines) - set(out_lines))
    if missing:
        raise ValueError(f"{output.get('method', '<unknown>')} is missing line ids {missing}")

    reprojection_rmse, p90_reprojection = _subset_reprojection_errors(dataset, out_lines)
    return {
        "median_direction_error_deg": float(np.median(dir_errors)),
        "p90_direction_error_deg": float(np.percentile(dir_errors, 90)),
        "median_normal_error_deg": float(np.median(normal_errors)),
        "reprojection_rmse_px": reprojection_rmse,
        "p90_reprojection_error_px": p90_reprojection,
        "effective_params": float(output["num_parameters"]),
        "parallel_spread_deg": _parallel_spread_deg(directions),
    }


def _score_dataset(data_path: str, outputs_dir: str) -> tuple[str, dict[str, Any]]:
    dataset = load_json(data_path)
    outputs = {}
    for path in sorted(Path(outputs_dir).glob("*.json")):
        payload = load_json(path)
        outputs[payload["method"]] = _score_one(dataset, payload)
    if not outputs:
        raise FileNotFoundError(f"no method output json files found in {outputs_dir}")
    experiment = str(dataset.get("name", Path(data_path).stem))
    return experiment, {"results": outputs}


def evaluate(pairs: list[tuple[str, str]], scores_path: str) -> None:
    scores: dict[str, Any] = {"experiments": {}}
    for data_path, outputs_dir in pairs:
        experiment, payload = _score_dataset(data_path, outputs_dir)
        scores["experiments"][experiment] = payload
    dump_json(scores_path, scores)
    print(f"wrote {scores_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("DATA_PATH", "OUTPUTS_DIR"),
        action="append",
        help="Dataset JSON and directory of method output JSON files.",
    )
    parser.add_argument("--scores", default="scoring/scores.json")
    args = parser.parse_args()
    pairs = args.pair
    if pairs is None:
        data_path = args.data or "data/riemanline_synthetic.json"
        outputs_dir = args.outputs_dir or "outputs"
        pairs = [(data_path, outputs_dir)]
    evaluate([(data, outputs) for data, outputs in pairs], args.scores)


if __name__ == "__main__":
    main()
