# Part of stereo-slam-kitti, a GPL-3.0 derivative of
# https://github.com/FoamoftheSea/KITTI_visual_odometry. This module is original work.
"""Trajectory error metrics.

Three families, all standard in the odometry/SLAM literature:

* :func:`kitti_odometry_error` - the official KITTI benchmark metric.  Errors are
  measured over sub-sequences of fixed length (100..800 m) and reported as
  translation error in percent of distance travelled and rotation error in
  degrees per metre.  This is the number papers quote, and the only one that is
  comparable across sequences of different length.
* :func:`absolute_trajectory_error` - RMSE of position difference over the whole
  trajectory, optionally after a rigid (or similarity) alignment.
* :func:`relative_pose_error` - drift measured over a fixed frame gap, which
  separates local accuracy from accumulated global drift.

Poses are ``(N, 3, 4)`` or ``(N, 4, 4)`` camera-to-world transforms, in the KITTI
convention (row-major 3x4 written out per line in the pose files).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

KITTI_LENGTHS = (100, 200, 300, 400, 500, 600, 700, 800)


# ---------------------------------------------------------------------- helpers


def to_4x4(poses: np.ndarray) -> np.ndarray:
    """Normalise an ``(N, 3, 4)`` or ``(N, 4, 4)`` pose array to ``(N, 4, 4)``."""
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1] not in (3, 4) or poses.shape[2] != 4:
        raise ValueError(f"Expected poses of shape (N, 3, 4) or (N, 4, 4), got {poses.shape}.")
    if poses.shape[1] == 4:
        return poses
    out = np.tile(np.eye(4), (poses.shape[0], 1, 1))
    out[:, :3, :] = poses
    return out


def rotation_angle(R: np.ndarray) -> float:
    """Geodesic rotation angle of a 3x3 rotation matrix, in radians."""
    trace = np.clip((np.trace(R[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(trace))


def trajectory_distances(poses: np.ndarray) -> np.ndarray:
    """Cumulative path length at each pose."""
    positions = to_4x4(poses)[:, :3, 3]
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _last_frame_from_distance(distances: np.ndarray, first: int, length: float) -> int:
    target = distances[first] + length
    index = np.searchsorted(distances, target, side="left")
    return int(index) if index < len(distances) else -1


# ----------------------------------------------------------------- KITTI metric


def kitti_odometry_error(
    gt: np.ndarray,
    estimated: np.ndarray,
    lengths: Iterable[float] = KITTI_LENGTHS,
    step_size: int = 10,
) -> Dict:
    """Official KITTI translation/rotation error.

    For every start frame (every ``step_size`` frames) and every sub-sequence
    length, compare the relative pose the estimate predicts against the relative
    pose the ground truth gives, and normalise by the length travelled.

    Returns a dict with the overall ``translation_error_percent`` and
    ``rotation_error_deg_per_m``, plus a ``per_length`` breakdown.
    """
    gt = to_4x4(gt)
    estimated = to_4x4(estimated)
    n = min(len(gt), len(estimated))
    if n < 2:
        raise ValueError("Need at least two poses to evaluate.")
    gt, estimated = gt[:n], estimated[:n]

    distances = trajectory_distances(gt)
    errors = []

    for first in range(0, n, step_size):
        for length in lengths:
            last = _last_frame_from_distance(distances, first, length)
            if last == -1 or last >= n:
                continue

            gt_delta = np.linalg.inv(gt[first]) @ gt[last]
            est_delta = np.linalg.inv(estimated[first]) @ estimated[last]
            error = np.linalg.inv(est_delta) @ gt_delta

            t_err = float(np.linalg.norm(error[:3, 3]))
            r_err = rotation_angle(error)
            errors.append(
                {
                    "first_frame": first,
                    "length": float(length),
                    "translation_error": t_err / length,
                    "rotation_error": r_err / length,
                }
            )

    if not errors:
        raise ValueError(
            "No sub-sequence of the requested lengths fits in this trajectory; "
            "use shorter lengths (e.g. lengths=(10, 20, 50))."
        )

    t_all = np.array([e["translation_error"] for e in errors])
    r_all = np.array([e["rotation_error"] for e in errors])

    per_length = {}
    for length in lengths:
        subset = [e for e in errors if e["length"] == float(length)]
        if subset:
            per_length[float(length)] = {
                "translation_error_percent": float(
                    np.mean([e["translation_error"] for e in subset]) * 100.0
                ),
                "rotation_error_deg_per_m": float(
                    np.degrees(np.mean([e["rotation_error"] for e in subset]))
                ),
                "num_samples": len(subset),
            }

    return {
        "translation_error_percent": float(t_all.mean() * 100.0),
        "rotation_error_deg_per_m": float(np.degrees(r_all.mean())),
        "num_samples": len(errors),
        "per_length": per_length,
    }


# -------------------------------------------------------------------- alignment


def umeyama_alignment(source: np.ndarray, target: np.ndarray, with_scale: bool = False):
    """Least-squares rigid (or similarity) alignment of two point sets.

    ``source`` and ``target`` are ``(N, 3)``.  Returns ``(R, t, s)`` such that
    ``s * R @ source_i + t`` best matches ``target_i``.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape:
        raise ValueError("Point sets must have the same shape.")

    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    sc, tc = source - mu_s, target - mu_t

    cov = tc.T @ sc / len(source)
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (sc ** 2).sum() * len(source)) if with_scale else 1.0
    t = mu_t - s * R @ mu_s
    return R, t, s


def absolute_trajectory_error(
    gt: np.ndarray,
    estimated: np.ndarray,
    align: bool = False,
    with_scale: bool = False,
) -> Dict:
    """RMSE / mean / median of the position error over the whole trajectory.

    KITTI trajectories are metric and start at the origin, so ``align=False`` is
    the honest default.  Alignment is useful when comparing against a monocular
    method that has no absolute scale.
    """
    gt = to_4x4(gt)
    estimated = to_4x4(estimated)
    n = min(len(gt), len(estimated))
    gt_xyz = gt[:n, :3, 3]
    est_xyz = estimated[:n, :3, 3]

    if align:
        R, t, s = umeyama_alignment(est_xyz, gt_xyz, with_scale=with_scale)
        est_xyz = (s * (R @ est_xyz.T)).T + t

    errors = np.linalg.norm(gt_xyz - est_xyz, axis=1)
    return {
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "std": float(errors.std()),
        "max": float(errors.max()),
        "final_drift": float(errors[-1]),
        "path_length": float(trajectory_distances(gt[:n])[-1]),
        "aligned": bool(align),
    }


def relative_pose_error(gt: np.ndarray, estimated: np.ndarray, delta: int = 1) -> Dict:
    """Translation and rotation drift over a fixed gap of ``delta`` frames."""
    gt = to_4x4(gt)
    estimated = to_4x4(estimated)
    n = min(len(gt), len(estimated))
    if n <= delta:
        raise ValueError(f"Trajectory of {n} poses is too short for delta={delta}.")

    t_errors, r_errors = [], []
    for i in range(n - delta):
        gt_delta = np.linalg.inv(gt[i]) @ gt[i + delta]
        est_delta = np.linalg.inv(estimated[i]) @ estimated[i + delta]
        error = np.linalg.inv(est_delta) @ gt_delta
        t_errors.append(np.linalg.norm(error[:3, 3]))
        r_errors.append(rotation_angle(error))

    t_errors = np.array(t_errors)
    r_errors = np.array(r_errors)
    return {
        "delta": delta,
        "translation_rmse": float(np.sqrt((t_errors ** 2).mean())),
        "translation_mean": float(t_errors.mean()),
        "rotation_rmse_deg": float(np.degrees(np.sqrt((r_errors ** 2).mean()))),
        "rotation_mean_deg": float(np.degrees(r_errors.mean())),
    }


def evaluate(
    gt: np.ndarray,
    estimated: np.ndarray,
    lengths: Iterable[float] = KITTI_LENGTHS,
    align: bool = False,
) -> Dict:
    """All three metric families in one call."""
    result = {
        "ate": absolute_trajectory_error(gt, estimated, align=align),
        "rpe_1": relative_pose_error(gt, estimated, delta=1),
    }
    try:
        result["kitti"] = kitti_odometry_error(gt, estimated, lengths=lengths)
    except ValueError as exc:
        result["kitti"] = {"error": str(exc)}
    return result


def format_report(result: Dict, title: Optional[str] = None) -> str:
    """Render :func:`evaluate` output as a readable block of text."""
    lines = []
    if title:
        lines += [title, "=" * len(title)]

    kitti = result.get("kitti", {})
    if "error" in kitti:
        lines.append(f"KITTI metric: unavailable ({kitti['error']})")
    else:
        lines += [
            f"KITTI translation error : {kitti['translation_error_percent']:.2f} %",
            f"KITTI rotation error    : {kitti['rotation_error_deg_per_m']:.5f} deg/m",
        ]

    ate = result["ate"]
    rpe = result["rpe_1"]
    lines += [
        f"ATE RMSE                : {ate['rmse']:.2f} m"
        + ("  (aligned)" if ate["aligned"] else ""),
        f"ATE median / max        : {ate['median']:.2f} m / {ate['max']:.2f} m",
        f"Final drift             : {ate['final_drift']:.2f} m over {ate['path_length']:.0f} m",
        f"RPE (1 frame) trans     : {rpe['translation_rmse']:.4f} m",
        f"RPE (1 frame) rot       : {rpe['rotation_rmse_deg']:.4f} deg",
    ]
    return "\n".join(lines)
