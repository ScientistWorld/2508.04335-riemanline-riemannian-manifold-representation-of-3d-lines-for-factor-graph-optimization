from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from data.geometry import angle_deg, dump_json, load_json, normalize


def _parallel_spread_deg(directions: list[np.ndarray]) -> float:
    if len(directions) < 2:
        return 0.0
    angles = [angle_deg(a, b) for a, b in itertools.combinations(directions, 2)]
    return float(np.mean(angles))


def _score_one(dataset: dict, output: dict) -> dict[str, float]:
    gt_lines = {line["id"]: line for line in dataset["lines"]}
    dir_errors = []
    normal_errors = []
    directions = []
    for line in output["lines"]:
        gt = gt_lines[line["id"]]
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
        "reprojection_rmse_px": float(output["final_reprojection_rmse_px"]),
        "effective_params": float(output["num_parameters"]),
        "parallel_spread_deg": _parallel_spread_deg(directions),
    }


def evaluate(data_path: str, outputs_dir: str, scores_path: str) -> None:
    dataset = load_json(data_path)
    outputs = {}
    for path in sorted(Path(outputs_dir).glob("*.json")):
        payload = load_json(path)
        outputs[payload["method"]] = _score_one(dataset, payload)
    if not outputs:
        raise FileNotFoundError(f"no method output json files found in {outputs_dir}")
    scores = {"experiments": {dataset.get("name", "scaled_synthetic_parallel_lines"): {"results": outputs}}}
    dump_json(scores_path, scores)
    print(f"wrote {scores_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/riemanline_synthetic.json")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--scores", default="scoring/scores.json")
    args = parser.parse_args()
    evaluate(args.data, args.outputs_dir, args.scores)


if __name__ == "__main__":
    main()
