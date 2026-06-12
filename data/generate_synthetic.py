from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.geometry import (
    dump_json,
    look_at_r_cw,
    normalize,
    plucker_from_direction_normal,
    project_point,
)


def generate(seed: int = 7, num_lines: int = 12, num_cameras: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    k = np.array([[520.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]])
    direction = normalize(np.array([0.28, 0.08, 0.96]))

    cameras = []
    for i, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, num_cameras, endpoint=False)):
        center = np.array([5.0 * np.cos(angle), 5.0 * np.sin(angle), 2.2 + 0.25 * np.sin(2 * angle)])
        r_cw = look_at_r_cw(center, np.array([0.0, 0.0, 1.2]))
        cameras.append(
            {
                "id": i,
                "R_cw": r_cw.tolist(),
                "t_cw": (-r_cw @ center).tolist(),
                "K": k.tolist(),
                "K_inv_T": np.linalg.inv(k).T.tolist(),
            }
        )

    lines = []
    observations = []
    b1 = normalize(np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), direction) * direction)
    b2 = normalize(np.cross(direction, b1))
    for line_id in range(num_lines):
        radius = 0.45 + 0.17 * line_id
        angle = 2.0 * np.pi * line_id / num_lines + 0.18 * rng.normal()
        closest = radius * (np.cos(angle) * b1 + np.sin(angle) * b2) + np.array([0.0, 0.0, 1.15])
        closest = closest - np.dot(closest, direction) * direction
        normal, d = plucker_from_direction_normal(direction, np.cross(closest, direction))
        segment_half = 1.25 + 0.2 * rng.random()
        lines.append(
            {
                "id": line_id,
                "direction": d.tolist(),
                "normal": normal.tolist(),
                "closest_point": closest.tolist(),
                "segment_half": segment_half,
            }
        )
        for camera in cameras:
            start = closest - segment_half * d
            end = closest + segment_half * d
            try:
                p0 = project_point(start, camera)
                p1 = project_point(end, camera)
            except ValueError:
                continue
            if not (-80 <= p0[0] <= 720 and -80 <= p0[1] <= 560 and -80 <= p1[0] <= 720 and -80 <= p1[1] <= 560):
                continue
            noise = rng.normal(0.0, 1.2, size=(2, 2))
            observations.append(
                {
                    "line_id": line_id,
                    "camera_id": camera["id"],
                    "p0": (p0 + noise[0]).tolist(),
                    "p1": (p1 + noise[1]).tolist(),
                }
            )

    return {
        "name": "scaled_synthetic_parallel_lines",
        "seed": seed,
        "cameras": cameras,
        "lines": lines,
        "parallel_groups": [list(range(num_lines))],
        "observations": observations,
        "notes": "Procedural factor graph with known camera poses and one group of parallel 3D line landmarks.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/riemanline_synthetic.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-lines", type=int, default=12)
    parser.add_argument("--num-cameras", type=int, default=8)
    args = parser.parse_args()
    payload = generate(args.seed, args.num_lines, args.num_cameras)
    dump_json(args.output, payload)
    print(f"wrote {args.output} with {len(payload['lines'])} lines and {len(payload['observations'])} observations")


if __name__ == "__main__":
    main()
