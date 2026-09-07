# Derived from https://github.com/FoamoftheSea/KITTI_visual_odometry (Nate Cibik),
# licensed under GPL-3.0. Refactored and extended; see README "Attribution".
"""Stereo disparity and depth estimation."""

from __future__ import annotations

import numpy as np
import cv2

from .dataset import decompose_projection_matrix

SAD_WINDOW = 6
NUM_DISPARITIES = SAD_WINDOW * 16
BLOCK_SIZE = 11


def compute_disparity(
    img_left: np.ndarray,
    img_right: np.ndarray,
    matcher: str = "sgbm",
    rgb: bool = False,
) -> np.ndarray:
    """Left-camera disparity map, in pixels.

    ``matcher`` is ``"bm"`` (block matching, fast) or ``"sgbm"`` (semi-global,
    slower but much cleaner on low-texture road surfaces).
    """
    if rgb:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    name = matcher.lower()
    if name == "bm":
        stereo = cv2.StereoBM_create(numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE)
    elif name == "sgbm":
        stereo = cv2.StereoSGBM_create(
            numDisparities=NUM_DISPARITIES,
            minDisparity=0,
            blockSize=BLOCK_SIZE,
            P1=8 * 1 * SAD_WINDOW ** 2,
            P2=32 * 1 * SAD_WINDOW ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
    else:
        raise ValueError(f"Unknown stereo matcher {matcher!r}; expected 'bm' or 'sgbm'.")

    # OpenCV returns fixed-point disparities scaled by 16.
    return stereo.compute(img_left, img_right).astype(np.float32) / 16.0


def disparity_to_depth(
    disparity: np.ndarray,
    k_left: np.ndarray,
    t_left: np.ndarray,
    t_right: np.ndarray,
    rectified: bool = True,
) -> np.ndarray:
    """Convert disparity to metric depth via ``Z = f * b / d``."""
    focal_length = k_left[0][0]
    baseline = t_right[0] - t_left[0] if rectified else t_left[0] - t_right[0]

    disparity = disparity.copy()
    # Invalid disparities (0 and -1) would blow up the division; clamp them to a
    # small value so they come out as very large depths and get masked later.
    disparity[disparity == 0.0] = 0.1
    disparity[disparity == -1.0] = 0.1

    return focal_length * baseline / disparity


def stereo_to_depth(
    img_left: np.ndarray,
    img_right: np.ndarray,
    P0: np.ndarray,
    P1: np.ndarray,
    matcher: str = "sgbm",
    rgb: bool = False,
    rectified: bool = True,
) -> np.ndarray:
    """Stereo pair to depth map in one call."""
    disparity = compute_disparity(img_left, img_right, matcher=matcher, rgb=rgb)
    k_left, _, t_left = decompose_projection_matrix(P0)
    _, _, t_right = decompose_projection_matrix(P1)
    return disparity_to_depth(disparity, k_left, t_left, t_right, rectified=rectified)


def left_border_mask(imheight: int, imwidth: int, border: int = NUM_DISPARITIES) -> np.ndarray:
    """Mask out the left band where no disparity can be computed.

    The first ``numDisparities`` columns of the left image have no counterpart
    inside the search range, so depth there is meaningless.  Feeding this mask to
    the feature detector both avoids bad 3D points and saves detection time.
    """
    mask = np.zeros((imheight, imwidth), dtype=np.uint8)
    cv2.rectangle(mask, (border, 0), (imwidth, imheight), 255, thickness=-1)
    return mask
