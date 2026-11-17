"""Open-Structure joint factor-graph instances (ICL-NUIM and TartanAir).

Each sequence of the Open-Structure benchmark (Li et al. 2024; the factor
graphs used by the paper's Tables 2-4) provides an *initial* co-visibility
factor graph built by a VO front end (drifting camera poses, point and line
landmarks), the 2D point / line-segment measurements, declared parallel-line
groups, and the ground-truth trajectory and line landmarks.

An instance is one sequence restricted to keyframes: every ``stride``-th
frame (at most ``MAX_KEYFRAMES``), starting at one of ``INSTANCES_PER_SEQUENCE``
phase offsets, so each sequence yields several factor graphs over the whole
trajectory with disjoint keyframes and measurements.  As in the paper (Sec. 5,
"Landmarks observed by fewer than three cameras are pruned prior to
optimization"), landmarks with fewer than three keyframe observations are
pruned; so are degenerate initial landmarks.

ICL-NUIM measurements come from real feature detection (about 1 px error).
Open-Structure's TartanAir measurements are exact projections of the
ground-truth landmarks; we add zero-mean Gaussian pixel noise
(``TARTAN_PIXEL_NOISE`` px, a fresh draw per instance) to their point and
segment-endpoint measurements, so that the problem stays an estimation
problem instead of one with an exact zero-error solution.  The public scene expresses everything in a world frame moved by
a rigid transform drawn from this split's seed and uses relabelled ids; ground
truth (trajectory, line landmarks) stays in the evaluator.

This file belongs to the TRAIN split: its sequences and ground truth are
public (``data/downloads/open_structure``).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "downloads" / "open_structure"
SPLIT = "train"
SEED = 20250806
SEQUENCES = ["livingroom0", "livingroom1", "office0", "office", "japanesealley"]
EXPERIMENT = "open_structure_joint_ba"

ICL = {"livingroom0", "livingroom1", "livingroom2", "livingroom3", "office0", "office1", "office2", "office3"}
INTRINSICS = {
    "icl": {"fx": 481.20, "fy": -480.0, "cx": 319.5, "cy": 239.5},
    "tartan": {"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0},
}
IMAGE_SIZE = [640, 480]
MAX_KEYFRAMES = 150
INSTANCES_PER_SEQUENCE = 4
TARTAN_PIXEL_NOISE = 1.0
MIN_OBS = 3


# ------------------------------------------------------------------ parsing

def quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _find(base: Path, name: str) -> Path | None:
    hits = sorted(base.rglob(name))
    return hits[0] if hits else None


def parse_venom(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"V": {}, "P": {}, "L": {}, "PA": [], "LA": [], "G": []}
    with path.open() as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            k = t[0]
            if k == "Vertex:":
                out["V"][int(t[1])] = np.array(t[2:9], float)
            elif k == "Mappoint:":
                out["P"][int(t[1])] = np.array(t[2:5], float)
            elif k == "Mapline:":
                out["L"][int(t[1])] = np.array(t[2:8], float)
            elif k == "MappointFrameAsso:":
                out["PA"].append((int(t[1]), int(t[2]), float(t[3]), float(t[4])))
            elif k == "MaplineFrameAsso:":
                # ml_id frame_id u_s v_s d_s u_e v_e d_e  (depths are not used)
                out["LA"].append((int(t[1]), int(t[2]), float(t[3]), float(t[4]), float(t[6]), float(t[7])))
            elif k == "ParalineMaplineAsso:":
                n = int(t[2])
                out["G"].append([int(x) for x in t[3:3 + n]])
    return out


def read_traj(path: Path) -> dict[int, np.ndarray]:
    d = {}
    with path.open() as f:
        for line in f:
            t = line.split()
            if len(t) >= 8 and not t[0].startswith("#"):
                d[int(float(t[0]))] = np.array(t[1:8], float)
    return d


def read_gt_lines(base: Path) -> dict[int, np.ndarray]:
    path = _find(base, "structline_landmarks.txt") or _find(base, "line_landmarks_refined.txt")
    out: dict[int, np.ndarray] = {}
    if path is None:
        return out
    with path.open() as f:
        for line in f:
            t = line.split()
            if len(t) >= 7 and not t[0].startswith("#"):
                try:
                    out[int(t[0])] = np.array(t[1:7], float)
                except ValueError:
                    continue
    return out


def sequence_dir(seq: str) -> Path:
    return DATA_DIR / seq


# ------------------------------------------------------------------ instances

def _rng(seq: str, k: int) -> np.random.Generator:
    h = hashlib.sha256(f"{SEED}:{seq}:{k}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:16], "little"))


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return quat_to_R(q)


def _scene_id(seq: str, k: int) -> str:
    return hashlib.sha256(f"{SPLIT}:{SEED}:{seq}:{k}:joint".encode()).hexdigest()[:16]


_SEQ_CACHE: dict[str, tuple] = {}


def load_sequence(seq: str) -> tuple[dict[str, Any], dict[int, np.ndarray], dict[int, np.ndarray]]:
    if seq not in _SEQ_CACHE:
        base = sequence_dir(seq)
        venom = _find(base, "venom_sequence.txt")
        gt_path = _find(base, "ground_truth.txt")
        if venom is None or gt_path is None:
            raise FileNotFoundError(f"Open-Structure sequence {seq} missing under {base}; run scripts/download.sh")
        _SEQ_CACHE[seq] = (parse_venom(venom), read_traj(gt_path), read_gt_lines(base))
    return _SEQ_CACHE[seq]


def build_instance(seq: str, k: int = 0) -> dict[str, Any]:
    g, gt, gt_lines = load_sequence(seq)
    intr = INTRINSICS["icl" if seq in ICL else "tartan"]
    rng = _rng(seq, k)

    frames_all = sorted(i for i in g["V"] if i in gt)
    stride = max(1, math.ceil(len(frames_all) / MAX_KEYFRAMES))
    offset = (k * stride) // INSTANCES_PER_SEQUENCE
    kfs = frames_all[offset::stride]
    kf_set = set(kfs)

    # landmarks: non-degenerate, observed in >= MIN_OBS distinct keyframes
    pobs = [a for a in g["PA"] if a[1] in kf_set and a[0] in g["P"]]
    lobs = [a for a in g["LA"] if a[1] in kf_set and a[0] in g["L"]]
    p_frames: dict[int, set] = {}
    for a in pobs:
        p_frames.setdefault(a[0], set()).add(a[1])
    l_frames: dict[int, set] = {}
    for a in lobs:
        l_frames.setdefault(a[0], set()).add(a[1])
    keep_p = {p for p, fs in p_frames.items() if len(fs) >= MIN_OBS and np.linalg.norm(g["P"][p]) > 1e-4}
    keep_l = set()
    for l, fs in l_frames.items():
        e = g["L"][l]
        if len(fs) >= MIN_OBS and np.linalg.norm(e[:3] - e[3:]) >= 0.01:
            keep_l.add(l)
    pobs = [a for a in pobs if a[0] in keep_p]
    lobs = [a for a in lobs if a[0] in keep_l]

    # world-frame change and id relabelling
    r_s = _random_rotation(rng)
    t_s = rng.uniform(-5.0, 5.0, size=3)
    frame_new = dict(zip(kfs, rng.permutation(len(kfs)).tolist()))
    p_list = sorted(keep_p)
    l_list = sorted(keep_l)
    point_new = dict(zip(p_list, rng.permutation(len(p_list)).tolist()))
    line_new = dict(zip(l_list, rng.permutation(len(l_list)).tolist()))

    def tf(x: np.ndarray) -> np.ndarray:
        return r_s @ x + t_s

    frames = []
    for f in kfs:  # temporal order; the first listed frame is the gauge anchor
        v = g["V"][f]
        r_wc = r_s @ quat_to_R(v[3:7])
        t_wc = tf(v[:3])
        r_cw = r_wc.T
        frames.append({"id": frame_new[f], "R_cw": r_cw.tolist(), "t_cw": (-r_cw @ t_wc).tolist()})
    points = [{"id": point_new[p], "xyz": tf(g["P"][p]).tolist()} for p in p_list]
    lines = [{"id": line_new[l], "start": tf(g["L"][l][:3]).tolist(), "end": tf(g["L"][l][3:]).tolist()} for l in l_list]
    order_p = rng.permutation(len(points))
    order_l = rng.permutation(len(lines))
    points = [points[i] for i in order_p]
    lines = [lines[i] for i in order_l]
    sigma = 0.0 if seq in ICL else TARTAN_PIXEL_NOISE
    pn = rng.normal(0.0, sigma, size=(len(pobs), 2)) if sigma > 0 else np.zeros((len(pobs), 2))
    ln = rng.normal(0.0, sigma, size=(len(lobs), 4)) if sigma > 0 else np.zeros((len(lobs), 4))
    point_obs = [[point_new[a[0]], frame_new[a[1]], round(a[2] + e[0], 4), round(a[3] + e[1], 4)]
                 for a, e in zip(pobs, pn)]
    line_obs = [[line_new[a[0]], frame_new[a[1]], round(a[2] + e[0], 4), round(a[3] + e[1], 4),
                 round(a[4] + e[2], 4), round(a[5] + e[3], 4)] for a, e in zip(lobs, ln)]
    groups = []
    for grp in g["G"]:
        members = [line_new[m] for m in grp if m in keep_l]
        if len(members) >= 2:
            groups.append(sorted(members))

    sid = _scene_id(seq, k)
    public = {
        "scene_id": sid,
        "task": "joint_ba",
        "intrinsics": dict(intr),
        "image_size": list(IMAGE_SIZE),
        "frames": frames,
        "points": points,
        "lines": lines,
        "point_observations": point_obs,
        "line_observations": line_obs,
        "parallel_groups": groups,
    }
    gt_frames = {}
    for f in kfs:
        q = gt[f]
        gt_frames[frame_new[f]] = {"center": q[:3].tolist(), "R_wc": quat_to_R(q[3:7]).tolist()}
    gt_l = {}
    for l in l_list:
        if l in gt_lines:
            e = gt_lines[l]
            if np.linalg.norm(e[:3] - e[3:]) > 1e-6:
                gt_l[line_new[l]] = {"start": e[:3].tolist(), "end": e[3:].tolist()}
    hidden = {
        "sequence": seq,
        "instance": k,
        "stride": stride,
        "offset": offset,
        "frames": gt_frames,
        "lines": gt_l,
        "initial_frames": {frame_new[f]: g["V"][f][:3].tolist() for f in kfs},
    }
    return {"public": public, "hidden": hidden}


def all_instances() -> list[dict[str, Any]]:
    return [build_instance(seq, k) for seq in SEQUENCES for k in range(INSTANCES_PER_SEQUENCE)]


# ------------------------------------------------------------------ scoring helpers

def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising sum |s R src + t - dst|^2 (Umeyama 1991; evo's Sim(3) alignment).
    The factor graph has 2D measurements only, so its scale is a free gauge."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    h = a.T @ b / len(src)
    u, sv, vt = np.linalg.svd(h)
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1.0
    r = vt.T @ d @ u.T
    var = float(np.sum(a * a)) / len(src)
    scale = float(np.sum(sv * np.diag(d))) / var if var > 1e-18 else 1.0
    if not (math.isfinite(scale) and scale > 1e-9):
        scale = 1.0
    return scale, r, md - scale * r @ ms


# ------------------------------------------------------------------ scoring

GATE_MEDIAN_LINE_RESIDUAL_PX = 10.0
ROTATION_TOL = 1e-3
PLUCKER_TOL = 1e-6
WORST = {
    "ate_rmse_cm": 1000.0,
    "rotation_rmse_deg": 180.0,
    "median_direction_error_deg": 90.0,
    "p90_direction_error_deg": 90.0,
    "median_normal_error_deg": 90.0,
    "param_count_error_per_line": 2.0,
}
METRICS = tuple(WORST)
SAME_DIRECTION_RAD = 1e-7  # estimated directions count as shared only when (numerically) identical
GT_SAME_DIRECTION_DEG = 0.01  # ground-truth parallel classes (declared groups are exactly parallel)
ATE_FLOOR_CM = 0.01
ROT_FLOOR_DEG = 1e-3


class InvalidJointOutput(Exception):
    pass


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not (math.isfinite(na) and math.isfinite(nb)) or na < 1e-12 or nb < 1e-12:
        return 90.0
    c = abs(float(np.dot(a, b))) / (na * nb)
    if not math.isfinite(c):
        return 90.0
    return math.degrees(math.acos(min(1.0, max(0.0, c))))


def _read_output(path: Path, frame_ids: list[int], line_ids: list[int]):
    import json

    if not path.is_file():
        raise InvalidJointOutput(f"missing output {path.name}")
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise InvalidJointOutput(f"{path.name}: unreadable JSON ({exc})") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list) or not isinstance(payload.get("lines"), list):
        raise InvalidJointOutput(f"{path.name}: needs 'frames' and 'lines' lists")
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for item in payload["frames"]:
        try:
            fid = int(item["id"])
            r = np.asarray(item["R_cw"], dtype=float).reshape(3, 3)
            t = np.asarray(item["t_cw"], dtype=float).reshape(3)
        except Exception as exc:  # noqa: BLE001
            raise InvalidJointOutput(f"{path.name}: malformed frame entry ({exc})") from exc
        if fid in frames:
            raise InvalidJointOutput(f"{path.name}: duplicate frame id {fid}")
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(t))):
            raise InvalidJointOutput(f"{path.name}: non-finite pose for frame {fid}")
        if np.max(np.abs(r @ r.T - np.eye(3))) > ROTATION_TOL or np.linalg.det(r) < 0:
            raise InvalidJointOutput(f"{path.name}: R_cw of frame {fid} is not a rotation")
        frames[fid] = (r, t)
    if set(frames) != set(frame_ids):
        raise InvalidJointOutput(f"{path.name}: frame ids mismatch")
    lines: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for item in payload["lines"]:
        try:
            lid = int(item["id"])
            n = np.asarray(item["normal"], dtype=float).reshape(3)
            d = np.asarray(item["direction"], dtype=float).reshape(3)
        except Exception as exc:  # noqa: BLE001
            raise InvalidJointOutput(f"{path.name}: malformed line entry ({exc})") from exc
        if lid in lines:
            raise InvalidJointOutput(f"{path.name}: duplicate line id {lid}")
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(d))):
            raise InvalidJointOutput(f"{path.name}: non-finite line {lid}")
        dn = float(np.linalg.norm(d))
        if dn < 1e-9:
            raise InvalidJointOutput(f"{path.name}: zero direction for line {lid}")
        d = d / dn
        m = n / dn
        # A far-away (poorly constrained) line is a legitimate, if bad, estimate: its
        # errors are scored like any other line, so no distance limit is imposed.
        if abs(float(np.dot(m, d))) > PLUCKER_TOL * max(1.0, float(np.linalg.norm(m))):
            raise InvalidJointOutput(f"{path.name}: line {lid} violates the Plucker constraint n.d = 0")
        lines[lid] = (m, d)
    if set(lines) != set(line_ids):
        raise InvalidJointOutput(f"{path.name}: line ids mismatch")
    return frames, lines


def _count_classes(directions: list[np.ndarray], tol_rad: float) -> int:
    reps: list[np.ndarray] = []
    for d in directions:
        d = d / np.linalg.norm(d)
        if not any(float(np.linalg.norm(np.cross(d, r))) < tol_rad for r in reps):
            reps.append(d)
    return len(reps)


def _count_shared(directions: list[np.ndarray]) -> int:
    """Distinct estimated directions: union of lines whose directions coincide."""
    n = len(directions)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(np.cross(directions[i], directions[j]))) < SAME_DIRECTION_RAD:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def score_instance(instance: dict[str, Any], output_path: Path) -> dict[str, Any]:
    public, hidden = instance["public"], instance["hidden"]
    frame_ids = [int(f["id"]) for f in public["frames"]]
    line_ids = [int(l["id"]) for l in public["lines"]]
    frames, lines = _read_output(output_path, frame_ids, line_ids)

    # Task gate: the output poses and lines must explain the scene's own line
    # measurements (evaluator copy).
    intr = public["intrinsics"]
    k = np.array([[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]])
    kinv_t = np.linalg.inv(k).T
    lo = np.asarray(public["line_observations"], float).reshape(-1, 6)
    dists = []
    for row in lo:
        r_cw, t_cw = frames[int(row[1])]
        m, d = lines[int(row[0])]
        dc = r_cw @ d
        nc = r_cw @ m + np.cross(t_cw, dc)
        l = kinv_t @ nc
        den = math.hypot(l[0], l[1])
        for u, v in ((row[2], row[3]), (row[4], row[5])):
            dist = abs(l[0] * u + l[1] * v + l[2]) / den if den > 1e-12 else 1e3
            dists.append(min(dist, 1e3) if math.isfinite(dist) else 1e3)
    gate = float(np.median(dists)) if dists else 0.0
    if not gate <= GATE_MEDIAN_LINE_RESIDUAL_PX:
        raise InvalidJointOutput(
            f"{public['scene_id']}: median line residual {gate:.2f}px on its own measurements exceeds "
            f"{GATE_MEDIAN_LINE_RESIDUAL_PX}px (task not solved)")

    # Trajectory: Sim(3) alignment of camera centres to ground truth (evo -as).
    ids = sorted(hidden["frames"], key=int)
    c_est = np.array([-frames[int(i)][0].T @ frames[int(i)][1] for i in ids])
    c_gt = np.array([hidden["frames"][i]["center"] for i in ids], float)
    s_a, r_a, t_a = umeyama_sim3(c_est, c_gt)
    err = np.linalg.norm(s_a * c_est @ r_a.T + t_a - c_gt, axis=1)
    rot = []
    for i in ids:
        r_wc = r_a @ frames[int(i)][0].T
        r_gt = np.asarray(hidden["frames"][i]["R_wc"], float)
        c = (np.trace(r_gt.T @ r_wc) - 1.0) / 2.0
        rot.append(math.degrees(math.acos(min(1.0, max(-1.0, c)))))
    rot = np.asarray(rot)

    # Lines in the ground-truth frame (same alignment).
    dir_err, nrm_err = [], []
    est_dirs, gt_dirs = [], []
    for lid, gl in sorted(hidden["lines"].items(), key=lambda kv: int(kv[0])):
        m, d = lines[int(lid)]
        est_dirs.append(d)
        d2 = r_a @ d
        m2 = s_a * (r_a @ m) + np.cross(t_a, d2)  # moment of the line under X -> s R X + t
        s, e = np.asarray(gl["start"], float), np.asarray(gl["end"], float)
        dg = e - s
        gt_dirs.append(dg)
        mg = np.cross(s, e) / np.linalg.norm(dg)
        dir_err.append(_angle_deg(d2, dg))
        nrm_err.append(_angle_deg(m2, mg))

    init = np.array([hidden["initial_frames"][i] for i in ids], float)
    s0, r0, t0 = umeyama_sim3(init, c_gt)
    init_err = np.linalg.norm(s0 * init @ r0.T + t0 - c_gt, axis=1)
    return {
        "scene_id": public["scene_id"],
        "sequence": hidden["sequence"],
        "num_frames": len(ids),
        "num_lines_scored": len(dir_err),
        "sim3_scale": s_a,
        "ate_rmse_cm": float(100.0 * math.sqrt(float(np.mean(err**2)))),
        "ate_median_cm": float(100.0 * np.median(err)),
        "rotation_rmse_deg": float(math.sqrt(float(np.mean(rot**2)))),
        "initial_ate_rmse_cm": float(100.0 * math.sqrt(float(np.mean(init_err**2)))),
        "direction_errors_deg": [round(v, 6) for v in dir_err],
        "normal_errors_deg": [round(v, 6) for v in nrm_err],
        "line_gate_median_px": gate,
        # Table 6 compactness on the scored lines: effective 2 x distinct directions + 2 x lines
        "distinct_directions": _count_shared(est_dirs),
        "true_direction_classes": _count_classes(gt_dirs, math.radians(GT_SAME_DIRECTION_DEG)),
    }


def _geomean(values: list[float], floor: float, ceil: float) -> float:
    v = np.clip(np.asarray(values, float), floor, ceil)
    v = np.where(np.isfinite(v), v, ceil)
    return float(math.exp(float(np.mean(np.log(v))))) if len(v) else ceil


def pooled_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    dirs = np.concatenate([np.asarray(r["direction_errors_deg"], float) for r in rows]) if rows else np.zeros(0)
    nrms = np.concatenate([np.asarray(r["normal_errors_deg"], float) for r in rows]) if rows else np.zeros(0)
    return {
        "ate_rmse_cm": _geomean([r["ate_rmse_cm"] for r in rows], ATE_FLOOR_CM, WORST["ate_rmse_cm"]),
        "rotation_rmse_deg": _geomean([r["rotation_rmse_deg"] for r in rows], ROT_FLOOR_DEG, WORST["rotation_rmse_deg"]),
        "median_direction_error_deg": float(np.median(dirs)) if len(dirs) else 90.0,
        "p90_direction_error_deg": float(np.percentile(dirs, 90)) if len(dirs) else 90.0,
        "median_normal_error_deg": float(np.median(nrms)) if len(nrms) else 90.0,
        "param_count_error_per_line": float(
            sum(2 * abs(r["distinct_directions"] - r["true_direction_classes"]) for r in rows)
            / max(1, sum(r["num_lines_scored"] for r in rows))) if any(r["num_lines_scored"] for r in rows) else 2.0,
    }
