"""Procedural line-landmark scenes for the hidden test split.

Self-contained on purpose: the hidden test must not depend on anything outside
``eval/test/`` (``data/``, ``method/`` and ``baseline/`` are editable by the
agents being ranked).  The scene distribution is identical to the train split;
only the seeds differ, and these seeds never leave this directory.

A scene is a factor graph with known camera poses and noisy 2D line segment
observations of 3D line landmarks.  Lines come in declared parallel groups plus
individual lines.  The evaluator keeps the 3D ground truth, held-out cameras
used for novel-view scoring, and (for the mislabeled experiment) the true
direction classes in memory; only the public part is shown to a method.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

SPLIT = "test"
NUM_SCENES = 64
# 128-bit root seed: scenes cannot be rebuilt without it, and it cannot be
# recovered by enumerating seeds.
ROOT_SEED = 0x9E3779B97F4A7C15F39CC0605CEDC834

IMAGE_W = 640
IMAGE_H = 480
FOCAL = 520.0

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "parallel_lines": {
        "code": 101,
        "groups": (2, 3),
        "lines_per_group": (5, 8),
        "singles": (2, 4),
        "subtle_mislabels_per_group": (0, 0),
        "gross_mislabel_prob": 0.0,
        "num_cameras": 8,
        "arc_deg": 360.0,
        "radius": (4.5, 5.5),
        "noise_px": 1.0,
        "visibility": 0.85,
        "truncation": 0.2,
        "half_length": (0.6, 1.4),
        "heldout_cameras": 4,
    },
    "parallel_lines_harder": {
        "code": 202,
        "groups": (3, 3),
        "lines_per_group": (6, 9),
        "singles": (3, 5),
        "subtle_mislabels_per_group": (0, 0),
        "gross_mislabel_prob": 0.0,
        "num_cameras": 6,
        "arc_deg": 110.0,
        "radius": (5.5, 6.5),
        "noise_px": 1.5,
        "visibility": 0.7,
        "truncation": 0.3,
        "half_length": (0.4, 1.0),
        "heldout_cameras": 4,
    },
    "parallel_lines_mislabeled": {
        "code": 303,
        "groups": (2, 3),
        "lines_per_group": (6, 9),
        "singles": (2, 3),
        "subtle_mislabels_per_group": (1, 2),
        "subtle_mislabel_deg": (3.0, 10.0),
        "gross_mislabel_prob": 0.5,
        "num_cameras": 8,
        "arc_deg": 360.0,
        "radius": (4.5, 5.5),
        "noise_px": 1.2,
        "visibility": 0.85,
        "truncation": 0.2,
        "half_length": (0.6, 1.4),
        "heldout_cameras": 4,
    },
}

SCENE_CENTER = np.array([0.0, 0.0, 1.2])
PLACEMENT_BOX = np.array([2.0, 2.0, 0.9])
MIN_GROUP_ANGLE_DEG = 30.0
MIN_SEGMENT_PX = 25.0
MIN_OBSERVATIONS = 3


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    while True:
        v = rng.normal(size=3)
        n = np.linalg.norm(v)
        if n > 1e-6:
            return v / n


def _perpendicular(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    while True:
        w = rng.normal(size=3)
        w = w - np.dot(w, v) * v
        n = np.linalg.norm(w)
        if n > 1e-6:
            return w / n


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    return (
        v * math.cos(angle)
        + np.cross(axis, v) * math.sin(angle)
        + axis * np.dot(axis, v) * (1.0 - math.cos(angle))
    )


def _look_at(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = _unit(target - center)
    right = _unit(np.cross(np.array([0.0, 0.0, -1.0]), forward))
    down = _unit(np.cross(forward, right))
    return np.vstack([right, down, forward])  # rows: world -> camera axes


def _project(point: np.ndarray, r_cw: np.ndarray, center: np.ndarray, k: np.ndarray):
    pc = r_cw @ (point - center)
    if pc[2] <= 0.2:
        return None
    uvw = k @ (pc / pc[2])
    return uvw[:2]


def _inside(p: np.ndarray) -> bool:
    return 0.0 <= p[0] <= IMAGE_W and 0.0 <= p[1] <= IMAGE_H


def _scene_rng(experiment: str, index: int) -> np.random.Generator:
    code = int(EXPERIMENTS[experiment]["code"])
    seq = np.random.SeedSequence([ROOT_SEED, code, int(index)])
    return np.random.Generator(np.random.PCG64(seq))


def _cameras(rng, cfg, k):
    n = int(cfg["num_cameras"])
    arc = math.radians(float(cfg["arc_deg"]))
    az0 = rng.uniform(0.0, 2.0 * math.pi)
    if arc >= 2.0 * math.pi - 1e-9:
        azimuths = az0 + 2.0 * math.pi * np.arange(n) / n + rng.normal(0.0, 0.08, n)
    else:
        azimuths = az0 + np.linspace(-arc / 2.0, arc / 2.0, n) + rng.normal(0.0, 0.03, n)
    cams = []
    for az in azimuths:
        radius = rng.uniform(*cfg["radius"])
        height = rng.uniform(0.8, 2.4)
        center = np.array([radius * math.cos(az), radius * math.sin(az), height])
        target = SCENE_CENTER + rng.normal(0.0, 0.3, 3)
        cams.append((_look_at(center, target), center))
    held = []
    for _ in range(int(cfg["heldout_cameras"])):
        az = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(4.0, 6.5)
        height = rng.uniform(0.5, 2.8)
        center = np.array([radius * math.cos(az), radius * math.sin(az), height])
        target = SCENE_CENTER + rng.normal(0.0, 0.3, 3)
        held.append((_look_at(center, target), center))
    return cams, held


def _group_directions(rng, count):
    dirs: list[np.ndarray] = []
    cos_min = math.cos(math.radians(MIN_GROUP_ANGLE_DEG))
    while len(dirs) < count:
        d = _random_unit(rng)
        if all(abs(float(np.dot(d, e))) < cos_min for e in dirs):
            dirs.append(d)
    return dirs


def _observe(rng, cfg, cams, k, start, end):
    obs = []
    trunc = float(cfg["truncation"])
    for cam_idx, (r_cw, center) in enumerate(cams):
        keep = rng.random() < float(cfg["visibility"])
        t0 = rng.uniform(0.0, trunc)
        t1 = rng.uniform(1.0 - trunc, 1.0)
        noise = rng.normal(0.0, float(cfg["noise_px"]), size=(2, 2))
        if not keep:
            continue
        a = start + t0 * (end - start)
        b = start + t1 * (end - start)
        pa = _project(a, r_cw, center, k)
        pb = _project(b, r_cw, center, k)
        if pa is None or pb is None or not (_inside(pa) and _inside(pb)):
            continue
        if np.linalg.norm(pb - pa) < MIN_SEGMENT_PX:
            continue
        obs.append((cam_idx, pa + noise[0], pb + noise[1]))
    return obs


def generate_scene(experiment: str, index: int) -> dict[str, Any]:
    """Return the full scene (public inputs plus hidden ground truth)."""
    cfg = EXPERIMENTS[experiment]
    rng = _scene_rng(experiment, index)
    k = np.array([[FOCAL, 0.0, IMAGE_W / 2.0], [0.0, FOCAL, IMAGE_H / 2.0], [0.0, 0.0, 1.0]])
    cams, held = _cameras(rng, cfg, k)

    num_groups = int(rng.integers(cfg["groups"][0], cfg["groups"][1] + 1))
    group_dirs = _group_directions(rng, num_groups)

    # (true direction, declared group or None, true direction class)
    specs: list[tuple[np.ndarray, int | None, int]] = []
    next_class = num_groups
    for g, d in enumerate(group_dirs):
        for _ in range(int(rng.integers(cfg["lines_per_group"][0], cfg["lines_per_group"][1] + 1))):
            specs.append((d, g, g))
        lo, hi = cfg["subtle_mislabels_per_group"]
        for _ in range(int(rng.integers(lo, hi + 1))):
            angle = math.radians(rng.uniform(*cfg.get("subtle_mislabel_deg", (0.0, 0.0))))
            specs.append((_unit(_rotate(d, _perpendicular(d, rng), angle)), g, next_class))
            next_class += 1
    if num_groups and rng.random() < float(cfg["gross_mislabel_prob"]):
        g = int(rng.integers(0, num_groups))
        while True:
            d = _random_unit(rng)
            if abs(float(np.dot(d, group_dirs[g]))) < math.cos(math.radians(35.0)):
                break
        specs.append((d, g, next_class))
        next_class += 1
    for _ in range(int(rng.integers(cfg["singles"][0], cfg["singles"][1] + 1))):
        specs.append((_random_unit(rng), None, next_class))
        next_class += 1

    origin = cams[0][1].copy()  # world frame = first input camera centre
    raw_lines = []
    for direction, declared, true_class in specs:
        for _attempt in range(200):
            mid = SCENE_CENTER + rng.uniform(-1.0, 1.0, 3) * PLACEMENT_BOX
            half = rng.uniform(*cfg["half_length"])
            start = mid - half * direction
            end = mid + half * direction
            obs = _observe(rng, cfg, cams, k, start, end)
            closest = (mid - origin) - np.dot(mid - origin, direction) * direction
            if len(obs) >= MIN_OBSERVATIONS and np.linalg.norm(closest) > 0.5:
                raw_lines.append((direction, declared, true_class, start, end, obs))
                break

    order = rng.permutation(len(raw_lines))  # ids carry no group information
    lines_gt = []
    observations = []
    groups: dict[int, list[int]] = {}
    for new_id, old in enumerate(order):
        direction, declared, true_class, start, end, obs = raw_lines[int(old)]
        s = start - origin
        e = end - origin
        point = s - np.dot(s, direction) * direction
        normal = np.cross(point, direction)
        lines_gt.append(
            {
                "id": new_id,
                "direction": direction.tolist(),
                "normal": normal.tolist(),
                "start": s.tolist(),
                "end": e.tolist(),
                "true_class": int(true_class),
                "declared_group": None if declared is None else int(declared),
            }
        )
        if declared is not None:
            groups.setdefault(int(declared), []).append(new_id)
        for cam_idx, p0, p1 in obs:
            observations.append(
                {"line_id": new_id, "camera_id": int(cam_idx), "p0": p0.tolist(), "p1": p1.tolist()}
            )
    observations.sort(key=lambda o: (o["camera_id"], o["line_id"]))

    def cam_payload(idx, r_cw, center):
        return {
            "id": idx,
            "K": k.tolist(),
            "R_cw": r_cw.tolist(),
            "t_cw": (-r_cw @ (center - origin)).tolist(),
        }

    scene_id = f"{experiment}_{index:03d}"
    public = {
        "scene_id": scene_id,
        "experiment": experiment,
        "image_size": [IMAGE_W, IMAGE_H],
        "cameras": [cam_payload(i, r, c) for i, (r, c) in enumerate(cams)],
        "lines": [{"id": i} for i in range(len(lines_gt))],
        "parallel_groups": [sorted(groups[g]) for g in sorted(groups)],
        "observations": observations,
    }
    hidden = {
        "lines": lines_gt,
        "heldout_cameras": [cam_payload(1000 + i, r, c) for i, (r, c) in enumerate(held)],
    }
    return {"public": public, "hidden": hidden}


def all_scenes(experiment: str) -> list[dict[str, Any]]:
    return [generate_scene(experiment, i) for i in range(NUM_SCENES)]
