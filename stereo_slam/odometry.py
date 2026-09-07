# Derived from https://github.com/FoamoftheSea/KITTI_visual_odometry (Nate Cibik),
# licensed under GPL-3.0. Refactored and extended; see README "Attribution".
"""Frame-to-frame motion estimation by 3D-2D PnP with RANSAC."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

MAX_DEPTH = 3000.0


def estimate_motion(
    matches: Sequence,
    kp1: Sequence,
    kp2: Sequence,
    k: np.ndarray,
    depth1: Optional[np.ndarray] = None,
    max_depth: float = MAX_DEPTH,
    reprojection_error: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the camera motion between two frames.

    Keypoints in frame 1 are back-projected to 3D using the stereo depth map of
    that frame, then matched against their 2D observations in frame 2 and solved
    with PnP + RANSAC.  This keeps the metric scale that monocular essential-matrix
    decomposition cannot recover.

    Returns ``(rmat, tvec, points1, points2)`` where ``rmat``/``tvec`` map points
    from the frame-1 camera into the frame-2 camera.
    """
    if len(matches) < 6:
        raise ValueError(f"Need at least 6 matches for PnP, got {len(matches)}.")
    if depth1 is None:
        raise ValueError("estimate_motion requires a depth map for the first frame.")

    image1_points = np.float32([kp1[m.queryIdx].pt for m in matches])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in matches])

    cx, cy = k[0, 2], k[1, 2]
    fx, fy = k[0, 0], k[1, 1]

    height, width = depth1.shape[:2]
    u = np.clip(np.round(image1_points[:, 0]).astype(int), 0, width - 1)
    v = np.clip(np.round(image1_points[:, 1]).astype(int), 0, height - 1)
    z = depth1[v, u]

    # Points beyond max_depth are where the stereo match failed; their depth is
    # meaningless and would drag the PnP solution off.
    keep = (z > 0) & (z < max_depth) & np.isfinite(z)
    if keep.sum() < 6:
        raise ValueError(f"Only {int(keep.sum())} matches had valid depth; need 6.")

    z = z[keep]
    image1_points = image1_points[keep]
    image2_points = image2_points[keep]

    x = z * (image1_points[:, 0] - cx) / fx
    y = z * (image1_points[:, 1] - cy) / fy
    object_points = np.column_stack([x, y, z]).astype(np.float64)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image2_points.astype(np.float64),
        k,
        None,
        reprojectionError=reprojection_error,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnPRansac failed to find a pose.")

    rmat = cv2.Rodrigues(rvec)[0]
    return rmat, tvec, image1_points, image2_points


def to_homogeneous(rmat: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Pack a rotation matrix and translation vector into a 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = rmat
    T[:3, 3] = tvec.ravel()
    return T
