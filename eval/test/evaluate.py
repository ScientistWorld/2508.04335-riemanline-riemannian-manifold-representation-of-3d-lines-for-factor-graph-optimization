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


SLICE = "test"


def _selected_line_ids(dataset: dict[str, Any]) -> list[int]:
    ids = sorted(int(line["id"]) for line in dataset["lines"])
    return [line_id for idx, line_id in enumerate(ids) if idx % 2 == 1]


def _parallel_spread_deg(directions: list[np.ndarray]) -> float:
    if len(directions) < 2:
        return 0.0
    angles = [angle_deg(a, b) for a, b in itertools.combinations(directions, 2)]
    return float(np.mean(angles))


def _subset_reprojection_rmse_px(
    dataset: dict[str, Any],
    lines_by_id: dict[int, dict[str, Any]],
    selected_ids: set[int],
) -> float:
    cameras = {int(camera["id"]): camera for camera in dataset["cameras"]}
    squared_errors: list[float] = []
    for obs in dataset["observations"]:
        line_id = int(obs["line_id"])
        if line_id not in selected_ids:
            continue
        line = lines_by_id[line_id]
        normal = np.asarray(line["normal"], dtype=float)
        direction = normalize(np.asarray(line["direction"], dtype=float))
        image_line = camera_line(normal, direction, cameras[int(obs["camera_id"])])
        for key in ("p0", "p1"):
            distance = point_line_distance(np.asarray(obs[key], dtype=float), image_line)
            squared_errors.append(distance * distance)
    if not squared_errors:
        raise ValueError("selected test slice has no observations")
    return float(np.sqrt(np.mean(squared_errors)))


def _score_one(dataset: dict[str, Any], output: dict[str, Any], selected_ids: list[int]) -> dict[str, float]:
    gt_lines = {int(line["id"]): line for line in dataset["lines"]}
    out_lines = {int(line["id"]): line for line in output["lines"]}
    missing = sorted(set(selected_ids) - set(out_lines))
    if missing:
        raise ValueError(f"{output.get('method', '<unknown>')} is missing line ids {missing}")

    dir_errors = []
    normal_errors = []
    directions = []
    for line_id in selected_ids:
        line = out_lines[line_id]
        gt = gt_lines[line_id]
        est_d = normalize(np.asarray(line["direction"], dtype=float))
        est_n = normalize(np.asarray(line["normal"], dtype=float))
        gt_d = normalize(np.asarray(gt["direction"], dtype=float))
        gt_n = normalize(np.asarray(gt["normal"], dtype=float))
        dir_errors.append(angle_deg(est_d, gt_d))
        normal_errors.append(angle_deg(est_n, gt_n))
        directions.append(est_d)

    return {
        "median_direction_error_deg": float(np.median(dir_errors)),
        "median_normal_error_deg": float(np.median(normal_errors)),
        "reprojection_rmse_px": _subset_reprojection_rmse_px(dataset, out_lines, set(selected_ids)),
        "effective_params": float(output["num_parameters"]),
        "parallel_spread_deg": _parallel_spread_deg(directions),
    }


def _load_outputs(outputs_dir: str | Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(outputs_dir).glob("*.json")):
        payload = load_json(path)
        method = payload.get("method")
        if not method:
            raise ValueError(f"{path} does not contain a method field")
        outputs[str(method)] = payload
    if not outputs:
        raise FileNotFoundError(f"no method output json files found in {outputs_dir}")
    return outputs


def _required_methods_by_experiment(path: str | None) -> dict[str, set[str]]:
    if not path:
        return {}
    scores_path = Path(path)
    if not scores_path.exists():
        return {}
    payload = load_json(scores_path)
    experiments = payload.get("experiments", payload)
    required: dict[str, set[str]] = {}
    for exp_name, exp_payload in experiments.items():
        if str(exp_name).startswith("_") or not isinstance(exp_payload, dict):
            continue
        results = exp_payload.get("results", exp_payload)
        if not isinstance(results, dict):
            continue
        methods = {str(name) for name in results if not str(name).startswith("_")}
        if methods:
            required[str(exp_name)] = methods
    return required


def _score_dataset(
    data_path: str,
    outputs_dir: str,
    required_methods: set[str] | None,
) -> tuple[str, dict[str, Any]]:
    dataset = load_json(data_path)
    selected_ids = _selected_line_ids(dataset)
    outputs = _load_outputs(outputs_dir)
    experiment = str(dataset.get("name", Path(data_path).stem))

    if required_methods:
        missing = sorted(required_methods - set(outputs))
        if missing:
            raise FileNotFoundError(
                f"{outputs_dir} is missing output JSON for train-scored methods: {missing}"
            )
        method_names = sorted(required_methods)
    else:
        method_names = sorted(outputs)

    results = {
        method: _score_one(dataset, outputs[method], selected_ids)
        for method in method_names
    }
    return experiment, {
        "results": results,
        "_metadata": {
            "slice": SLICE,
            "split_rule": "sorted line ids at odd zero-based positions",
            "line_ids": selected_ids,
            "data": data_path,
            "outputs_dir": outputs_dir,
            "required_methods_from_train": bool(required_methods),
        },
    }


def evaluate(pairs: list[tuple[str, str]], scores_path: str, train_scores_path: str | None) -> None:
    required = _required_methods_by_experiment(train_scores_path)
    scores: dict[str, Any] = {
        "experiments": {},
        "_metadata": {
            "slice": SLICE,
            "split_style": "deterministic 50/50 split over sorted line ids",
            "required_methods_source": train_scores_path or "",
        },
    }
    for data_path, outputs_dir in pairs:
        dataset = load_json(data_path)
        experiment_name = str(dataset.get("name", Path(data_path).stem))
        experiment, payload = _score_dataset(
            data_path,
            outputs_dir,
            required.get(experiment_name),
        )
        scores["experiments"][experiment] = payload
    dump_json(scores_path, scores)
    print(f"wrote {scores_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("DATA_PATH", "OUTPUTS_DIR"),
        action="append",
        required=True,
        help="Dataset JSON and directory of method output JSON files.",
    )
    parser.add_argument("--scores", default="scoring/scores_test.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    args = parser.parse_args()
    evaluate([(data, outputs) for data, outputs in args.pair], args.scores, args.train_scores)


if __name__ == "__main__":
    main()
