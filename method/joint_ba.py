"""Joint pose / point / line factor-graph optimisation (paper Sec. 4).

The co-visibility factor graph of an Open-Structure sequence: camera poses
T_wc (the first frame is held fixed, as in the authors' optimisers), 3D point
landmarks, and line landmarks in a pluggable minimal representation.

Factors, all in normalised image coordinates scaled by f_x (so residuals are
in pixel-like units and the authors' CauchyLoss(3 / f_x) becomes Cauchy(3)):
  * point re-projection factor r_p(p_obs, P_w, T_wc)               (Sec. 4)
  * line re-projection factor (Eq. 10): distances of both observed segment
    endpoints to the projected 3D line
  * optional extra residuals supplied by the line representation (e.g. the
    Eq. 11 parallelism residual of the OrthLine_Constr baseline).

Pose update (authors' ``PoseLocalParameterization``): t <- t + dp,
R <- R Exp(dtheta).  Line updates are delegated to the representation's
``plus`` (Eq. 4-7 for RiemanLine / StructRiemanLine).

Solver: Ceres-style Levenberg-Marquardt (trust-region radius update, default
Ceres tolerances, at most 100 iterations as in the authors' code) with the
point landmarks eliminated by the Schur complement.  Point Jacobians are
analytic; line-factor Jacobians are numeric in the local coordinates (as in
the authors' line factors), with columns that touch disjoint rows perturbed
together.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

EPS = 1e-12
FD_EPS = 1e-6


# ------------------------------------------------------------------ geometry

def hat(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] = v[..., 1]
    out[..., 1, 0] = v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] = v[..., 0]
    return out


def so3_exp(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=-1)
    k = hat(w)
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1.0 - th**2 / 6.0, np.sin(ths) / ths)
    b = np.where(small, 0.5 - th**2 / 24.0, (1.0 - np.cos(ths)) / ths**2)
    eye = np.broadcast_to(np.eye(3), k.shape)
    return eye + a[..., None, None] * k + b[..., None, None] * (k @ k)


def orthonormalize(r: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(r)
    out = u @ vt
    det = np.linalg.det(out)
    if np.any(det < 0):
        u = u.copy()
        u[det < 0, :, -1] *= -1
        out = u @ vt
    return out


# ------------------------------------------------------------------ problem

class JointProblem:
    """Vectorised view of one joint factor graph (public scene JSON)."""

    def __init__(self, scene: dict[str, Any]):
        intr = scene["intrinsics"]
        self.fx, self.fy = float(intr["fx"]), float(intr["fy"])
        self.cx, self.cy = float(intr["cx"]), float(intr["cy"])
        self.scale = abs(self.fx)

        frames = scene["frames"]
        self.frame_ids = [int(f["id"]) for f in frames]
        self.frame_index = {fid: i for i, fid in enumerate(self.frame_ids)}
        r_cw = np.array([f["R_cw"] for f in frames], float).reshape(-1, 3, 3)
        t_cw = np.array([f["t_cw"] for f in frames], float).reshape(-1, 3)
        r_cw = orthonormalize(r_cw)
        self.R_wc0 = np.transpose(r_cw, (0, 2, 1))
        self.t_wc0 = -np.einsum("nij,nj->ni", self.R_wc0, t_cw)
        self.num_frames = len(frames)
        self.anchor = 0  # first listed frame is held fixed (gauge), as in the authors' code

        pts = scene.get("points", [])
        self.point_ids = [int(p["id"]) for p in pts]
        self.point_index = {pid: i for i, pid in enumerate(self.point_ids)}
        self.X0 = np.array([p["xyz"] for p in pts], float).reshape(-1, 3)
        self.num_points = len(pts)

        lines = scene["lines"]
        self.line_ids = [int(l["id"]) for l in lines]
        self.line_index = {lid: i for i, lid in enumerate(self.line_ids)}
        self.line_start = np.array([l["start"] for l in lines], float).reshape(-1, 3)
        self.line_end = np.array([l["end"] for l in lines], float).reshape(-1, 3)
        self.num_lines = len(lines)

        po = np.asarray(scene.get("point_observations", []), float).reshape(-1, 4)
        keep = np.array([int(a) in self.point_index and int(b) in self.frame_index for a, b in po[:, :2]], bool) \
            if len(po) else np.zeros(0, bool)
        po = po[keep]
        self.p_pt = np.array([self.point_index[int(a)] for a in po[:, 0]], int)
        self.p_fr = np.array([self.frame_index[int(b)] for b in po[:, 1]], int)
        self.p_xn = np.stack([(po[:, 2] - self.cx) / self.fx, (po[:, 3] - self.cy) / self.fy], axis=1) \
            if len(po) else np.zeros((0, 2))

        lo = np.asarray(scene.get("line_observations", []), float).reshape(-1, 6)
        keep = np.array([int(a) in self.line_index and int(b) in self.frame_index for a, b in lo[:, :2]], bool) \
            if len(lo) else np.zeros(0, bool)
        lo = lo[keep]
        self.l_ln = np.array([self.line_index[int(a)] for a in lo[:, 0]], int)
        self.l_fr = np.array([self.frame_index[int(b)] for b in lo[:, 1]], int)
        self.l_s = np.stack([(lo[:, 2] - self.cx) / self.fx, (lo[:, 3] - self.cy) / self.fy], axis=1) \
            if len(lo) else np.zeros((0, 2))
        self.l_e = np.stack([(lo[:, 4] - self.cx) / self.fx, (lo[:, 5] - self.cy) / self.fy], axis=1) \
            if len(lo) else np.zeros((0, 2))

        self.groups = [
            [self.line_index[int(m)] for m in g if int(m) in self.line_index]
            for g in scene.get("parallel_groups", [])
        ]

        # pose column offsets (anchor frame has none)
        self.pose_col = np.full(self.num_frames, -1, int)
        k = 0
        for i in range(self.num_frames):
            if i != self.anchor:
                self.pose_col[i] = 6 * k
                k += 1
        self.num_pose_params = 6 * k

    def initial_plucker(self) -> tuple[np.ndarray, np.ndarray]:
        """Plucker (moment, unit direction) of the initial line landmarks."""
        d = self.line_end - self.line_start
        dn = np.linalg.norm(d, axis=1, keepdims=True)
        dn = np.where(dn < EPS, 1.0, dn)
        n = np.cross(self.line_start, self.line_end) / dn
        d = d / dn
        n = n - np.sum(n * d, axis=1, keepdims=True) * d
        return n, d


class Poses:
    def __init__(self, r_wc: np.ndarray, t_wc: np.ndarray):
        self.R = r_wc
        self.t = t_wc

    def plus(self, problem: JointProblem, delta: np.ndarray) -> "Poses":
        d = np.zeros((problem.num_frames, 6))
        free = problem.pose_col >= 0
        idx = problem.pose_col[free]
        d[free] = delta[idx[:, None] + np.arange(6)[None, :]]
        return Poses(self.R @ so3_exp(d[:, 3:]), self.t + d[:, :3])

    def plus_all(self, k: int, eps: float) -> "Poses":
        """Perturb local coordinate k of every pose by eps (for numeric Jacobians)."""
        if k < 3:
            t = self.t.copy()
            t[:, k] += eps
            return Poses(self.R, t)
        w = np.zeros(3)
        w[k - 3] = eps
        return Poses(self.R @ so3_exp(w)[None], self.t)


# ------------------------------------------------------------------ residuals

def point_residuals(pb: JointProblem, poses: Poses, X: np.ndarray, with_jac: bool = False):
    R = poses.R[pb.p_fr]
    xc = np.einsum("kji,kj->ki", R, X[pb.p_pt] - poses.t[pb.p_fr])  # R^T (X - t)
    z = xc[:, 2]
    zs = np.where(np.abs(z) < 1e-6, np.where(z < 0, -1e-6, 1e-6), z)
    proj = xc[:, :2] / zs[:, None]
    r = pb.scale * (proj - pb.p_xn)
    if not with_jac:
        return r.ravel()
    dproj = np.zeros((len(z), 2, 3))
    dproj[:, 0, 0] = 1.0 / zs
    dproj[:, 1, 1] = 1.0 / zs
    dproj[:, 0, 2] = -xc[:, 0] / zs**2
    dproj[:, 1, 2] = -xc[:, 1] / zs**2
    dproj *= pb.scale
    rt = np.transpose(R, (0, 2, 1))
    j_pt = dproj @ rt
    j_pose = np.concatenate([-j_pt, dproj @ hat(xc)], axis=2)
    return r.ravel(), j_pose, j_pt


def line_residuals(pb: JointProblem, poses: Poses, normals: np.ndarray, directions: np.ndarray) -> np.ndarray:
    R = poses.R[pb.l_fr]
    t = poses.t[pb.l_fr]
    n = normals[pb.l_ln]
    d = directions[pb.l_ln]
    nc = np.einsum("kji,kj->ki", R, n - np.cross(t, d))  # R_cw (n_w - t_wc x d_w)
    den = np.sqrt(nc[:, 0] ** 2 + nc[:, 1] ** 2) + EPS
    r0 = (nc[:, 0] * pb.l_s[:, 0] + nc[:, 1] * pb.l_s[:, 1] + nc[:, 2]) / den
    r1 = (nc[:, 0] * pb.l_e[:, 0] + nc[:, 1] * pb.l_e[:, 1] + nc[:, 2]) / den
    return pb.scale * np.stack([r0, r1], axis=1).ravel()


# ------------------------------------------------------------------ robust loss

def robust(s: np.ndarray, loss: str, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-block rho(s) and rho'(s) for squared block norms s (Ceres convention: cost = 1/2 sum rho)."""
    if loss == "cauchy":
        c2 = c * c
        return c2 * np.log1p(s / c2), 1.0 / (1.0 + s / c2)
    if loss == "huber":
        c2 = c * c
        root = np.sqrt(np.maximum(s, 0.0))
        inl = s <= c2
        return np.where(inl, s, 2.0 * c * root - c2), np.where(inl, 1.0, c / np.maximum(root, EPS))
    if loss in ("linear", "trivial", "none"):
        return s.copy(), np.ones_like(s)
    raise ValueError(f"unknown loss {loss}")


# ------------------------------------------------------------------ colouring

def colouring(col_lines: Sequence[np.ndarray], row_lines: Sequence[Sequence[int]], num_lines: int):
    """Group local line columns whose residual rows are disjoint (one numeric
    perturbation per group).  ``row_lines[r]`` lists the lines residual row r touches."""
    rows_of_line: list[list[int]] = [[] for _ in range(num_lines)]
    for r, lines in enumerate(row_lines):
        for l in lines:
            rows_of_line[int(l)].append(r)
    n_rows = len(row_lines)
    col_rows = []
    for lines in col_lines:
        parts = [np.asarray(rows_of_line[int(l)], dtype=int) for l in lines]
        col_rows.append(np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=int))
    colours: list[dict[str, Any]] = []
    for c, rows in enumerate(col_rows):
        for colour in colours:
            if not colour["used"][rows].any():
                colour["cols"].append(c)
                colour["used"][rows] = True
                break
        else:
            used = np.zeros(n_rows, dtype=bool)
            used[rows] = True
            colours.append({"cols": [c], "used": used})
    packed = []
    for colour in colours:
        cols = np.asarray(colour["cols"], dtype=int)
        rows_idx = np.concatenate([col_rows[c] for c in cols]) if len(cols) else np.zeros(0, int)
        cols_idx = np.concatenate([np.full(len(col_rows[c]), c, dtype=int) for c in cols]) if len(cols) else np.zeros(0, int)
        packed.append((cols, rows_idx, cols_idx))
    return packed


# ------------------------------------------------------------------ solver

class LineModel:
    """Interface of a line representation used by :func:`solve_joint`.

    state.plus(delta) -> new state; state.decode() -> (normals, directions)
    col_lines: for every local column, the line indices it moves.
    extra(state) -> (E,) extra residual rows; extra_row_lines: lines per extra row.
    """

    def __init__(self, state: Any, col_lines: list[np.ndarray],
                 extra: Callable[[Any], np.ndarray] | None = None,
                 extra_row_lines: list[Sequence[int]] | None = None):
        self.state = state
        self.col_lines = col_lines
        self.extra = extra
        self.extra_row_lines = extra_row_lines or []


def _evaluate(pb: JointProblem, poses: Poses, X: np.ndarray, lstate: Any, model: LineModel, loss: str, c: float):
    """Residual vector, per-row IRLS weights (one robust weight per residual block) and cost."""
    K = len(pb.p_pt)
    KL = len(pb.l_ln)
    rp = point_residuals(pb, poses, X)
    normals, directions = lstate.decode()
    rl = line_residuals(pb, poses, normals, directions)
    re = model.extra(lstate) if model.extra is not None else np.zeros(0)
    r = np.concatenate([rp, rl, re])
    s_blocks = np.concatenate([np.sum(rp.reshape(-1, 2) ** 2, axis=1), np.sum(rl.reshape(-1, 2) ** 2, axis=1), re**2])
    rho, drho = robust(s_blocks, loss, c)
    cost = 0.5 * float(np.sum(rho))
    w = np.concatenate([np.repeat(drho[:K], 2), np.repeat(drho[K:K + KL], 2), drho[K + KL:]])
    return r, w, cost


def _jacobian(pb: JointProblem, poses: Poses, X: np.ndarray, lstate: Any, model: LineModel, colours, r: np.ndarray):
    """Sparse Jacobian of all rows w.r.t. [poses, line local params] and the
    per-observation point Jacobian blocks."""
    num_lp = len(model.col_lines)
    C = pb.num_pose_params + num_lp
    K = len(pb.p_pt)
    KL = len(pb.l_ln)
    _, j_pose, j_pt = point_residuals(pb, poses, X, with_jac=True)
    normals, directions = lstate.decode()
    rl = r[2 * K: 2 * K + 2 * KL]
    re = r[2 * K + 2 * KL:]

    rows, cols, vals = [], [], []
    # point rows: pose columns (analytic)
    pc = pb.pose_col[pb.p_fr]
    m = pc >= 0
    if np.any(m):
        k_idx = np.nonzero(m)[0]
        rr = (2 * k_idx[:, None, None] + np.arange(2)[None, :, None]) * np.ones((1, 1, 6), int)
        cc = (pc[k_idx][:, None, None] + np.arange(6)[None, None, :]) * np.ones((1, 2, 1), int)
        rows.append(rr.ravel())
        cols.append(cc.ravel())
        vals.append(j_pose[k_idx].ravel())
    # line rows: pose columns (numeric, all poses perturbed together)
    off_l = 2 * K
    if KL:
        lpc = pb.pose_col[pb.l_fr]
        lm = lpc >= 0
        lk = np.nonzero(lm)[0]
        for k in range(6):
            r1 = line_residuals(pb, poses.plus_all(k, FD_EPS), normals, directions)
            dj = ((r1 - rl) / FD_EPS).reshape(-1, 2)
            rows.append(np.concatenate([off_l + 2 * lk, off_l + 2 * lk + 1]))
            cols.append(np.concatenate([lpc[lk] + k, lpc[lk] + k]))
            vals.append(np.concatenate([dj[lk, 0], dj[lk, 1]]))
    # line rows + extra rows: line columns (numeric, coloured)
    base_l = np.concatenate([rl, re])
    for col_set, rows_idx, cols_idx in colours:
        delta = np.zeros(num_lp)
        delta[col_set] = FD_EPS
        s2 = lstate.plus(delta)
        n2, d2 = s2.decode()
        r2 = np.concatenate([line_residuals(pb, poses, n2, d2), model.extra(s2) if model.extra is not None else np.zeros(0)])
        rows.append(off_l + rows_idx)
        cols.append(pb.num_pose_params + cols_idx)
        vals.append((r2[rows_idx] - base_l[rows_idx]) / FD_EPS)

    n_rows = len(r)
    if rows:
        jc = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_rows, C))
    else:
        jc = sp.csr_matrix((n_rows, C))
    return jc, j_pt


def solve_joint(
    pb: JointProblem,
    model: LineModel,
    loss: str = "cauchy",
    scale: float = 3.0,
    max_iterations: int = 100,
    function_tolerance: float = 1e-6,
    gradient_tolerance: float = 1e-10,
    parameter_tolerance: float = 1e-8,
    initial_radius: float = 1e4,
    optimize_points: bool = True,
    verbose: bool = False,
) -> tuple[Poses, np.ndarray, Any, dict[str, Any]]:
    poses = Poses(pb.R_wc0.copy(), pb.t_wc0.copy())
    X = pb.X0.copy()
    lstate = model.state
    num_lp = len(model.col_lines)
    C = pb.num_pose_params + num_lp
    K = len(pb.p_pt)
    M = pb.num_points
    KL = len(pb.l_ln)

    row_lines = [(int(l),) for l in np.repeat(pb.l_ln, 2)] + [tuple(int(x) for x in rl) for rl in model.extra_row_lines]
    colours = colouring(model.col_lines, row_lines, pb.num_lines)

    # sparse layout of the point Jacobian (fixed)
    jp_rows = (2 * np.arange(K)[:, None, None] + np.arange(2)[None, :, None]) * np.ones((1, 1, 3), int)
    jp_cols = (3 * pb.p_pt[:, None, None] + np.arange(3)[None, None, :]) * np.ones((1, 2, 1), int)
    # index layout of the block-diagonal inverse point Hessian
    bd_rows = ((3 * np.arange(M)[:, None, None] + np.arange(3)[None, :, None]) * np.ones((1, 1, 3), int)).ravel()
    bd_cols = ((3 * np.arange(M)[:, None, None] + np.arange(3)[None, None, :]) * np.ones((1, 3, 1), int)).ravel()

    r, w, cost = _evaluate(pb, poses, X, lstate, model, loss, scale)
    info: dict[str, Any] = {"initial_cost": cost, "iterations": 0, "successful_steps": 0,
                            "num_pose_params": pb.num_pose_params, "num_line_params": num_lp,
                            "num_point_params": 3 * M, "termination": "max_iterations"}
    radius = initial_radius
    nu = 2.0
    need_build = True
    for it in range(max_iterations):
        if need_build:
            jc, j_pt = _jacobian(pb, poses, X, lstate, model, colours, r)
            wr = w * r
            jcw = (sp.diags(w) @ jc).tocsr()
            h_cc = (jc.T @ jcw).toarray()
            g_c = jc.T @ wr
            if optimize_points and M:
                wp = w[: 2 * K]
                jpw = j_pt * wp.reshape(-1, 2)[:, :, None]
                vblk = np.einsum("kai,kaj->kij", j_pt, jpw)
                V = np.zeros((M, 3, 3))
                np.add.at(V, pb.p_pt, vblk)
                gp_k = np.einsum("kai,ka->ki", j_pt, wr[: 2 * K].reshape(-1, 2))
                g_p = np.zeros((M, 3))
                np.add.at(g_p, pb.p_pt, gp_k)
                jp = sp.csr_matrix((j_pt.ravel(), (jp_rows.ravel(), jp_cols.ravel())), shape=(len(r), 3 * M))
                h_cp = (jcw.T @ jp).tocsr()  # C x 3M
            else:
                V = np.zeros((0, 3, 3))
                g_p = np.zeros((0, 3))
                jp = None
                h_cp = None
            grad_max = max(float(np.max(np.abs(g_c))) if C else 0.0, float(np.max(np.abs(g_p))) if g_p.size else 0.0)
            if grad_max < gradient_tolerance:
                info["termination"] = "gradient_tolerance"
                break
            d_c = np.clip(np.diag(h_cc), 1e-6, 1e32)
            d_p = np.clip(np.diagonal(V, axis1=1, axis2=2), 1e-6, 1e32) if M else np.zeros((0, 3))
            need_build = False

        mu = 1.0 / radius
        info["iterations"] = it + 1
        a = h_cc + np.diag(mu * d_c)
        try:
            if optimize_points and M:
                vd = V + mu * d_p[:, :, None] * np.eye(3)[None]
                vinv = np.linalg.inv(vd)
                vinv_sp = sp.csr_matrix((vinv.ravel(), (bd_rows, bd_cols)), shape=(3 * M, 3 * M))
                t_mat = (h_cp @ vinv_sp).tocsr()
                s_mat = a - (t_mat @ h_cp.T).toarray()
                rhs = -g_c + t_mat @ g_p.ravel()
                if C:
                    dc = sla.cho_solve(sla.cho_factor(s_mat, check_finite=False), rhs, check_finite=False)
                else:
                    dc = np.zeros(0)
                dp = -np.einsum("mij,mj->mi", vinv, g_p + (h_cp.T @ dc).reshape(-1, 3)).ravel()
            else:
                dc = sla.cho_solve(sla.cho_factor(a, check_finite=False), -g_c, check_finite=False) if C else np.zeros(0)
                dp = np.zeros(3 * M)
        except (np.linalg.LinAlgError, ValueError):
            radius /= nu
            nu *= 2.0
            if radius < 1e-32:
                info["termination"] = "linear_solver_failure"
                break
            continue
        if not (np.all(np.isfinite(dc)) and np.all(np.isfinite(dp))):
            radius /= nu
            nu *= 2.0
            continue

        # model decrease (undamped Gauss-Newton model of the IRLS-weighted cost)
        jd = jc @ dc
        if jp is not None:
            jd = jd + jp @ dp
        pred = -(float(g_c @ dc) + (float(g_p.ravel() @ dp) if g_p.size else 0.0)) - 0.5 * float(np.sum(w * jd * jd))

        new_poses = poses.plus(pb, dc[: pb.num_pose_params])
        new_X = X + dp.reshape(-1, 3) if M else X
        new_l = lstate.plus(dc[pb.num_pose_params:]) if num_lp else lstate
        r_new, w_new, cost_new = _evaluate(pb, new_poses, new_X, new_l, model, loss, scale)
        actual = cost - cost_new
        rho = actual / pred if pred > 0 else -1.0
        if np.isfinite(cost_new) and rho > 1e-3:
            step_norm = math.sqrt(float(dc @ dc) + float(dp @ dp))
            x_norm = math.sqrt(float(np.sum(poses.t**2)) + float(np.sum(X**2)))
            poses, X, lstate = new_poses, new_X, new_l
            r, w = r_new, w_new
            old_cost, cost = cost, cost_new
            info["successful_steps"] += 1
            radius = min(1e16, radius / max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3))
            nu = 2.0
            need_build = True
            if verbose:
                print(f"    it {it:3d} cost {cost:.6e} rho {rho:.3f} radius {radius:.2e}", flush=True)
            if abs(old_cost - cost) <= function_tolerance * old_cost:
                info["termination"] = "function_tolerance"
                break
            if step_norm <= parameter_tolerance * (x_norm + parameter_tolerance):
                info["termination"] = "parameter_tolerance"
                break
        else:
            radius /= nu
            nu *= 2.0
            if radius < 1e-32:
                info["termination"] = "trust_region_collapse"
                break
    info["final_cost"] = cost
    return poses, X, lstate, info


def frames_payload(pb: JointProblem, poses: Poses) -> list[dict[str, Any]]:
    out = []
    for i, fid in enumerate(pb.frame_ids):
        r_cw = poses.R[i].T
        t_cw = -r_cw @ poses.t[i]
        out.append({"id": fid, "R_cw": r_cw.tolist(), "t_cw": t_cw.tolist()})
    return out
