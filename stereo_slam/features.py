# Derived from https://github.com/FoamoftheSea/KITTI_visual_odometry (Nate Cibik),
# licensed under GPL-3.0. Refactored and extended; see README "Attribution".
"""Feature detection, description and matching."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


#: ORB features per frame.  This interacts with ``KeyframeLoopCloser`` more than it
#: looks: loop detection fires on a fixed *count* of surviving matches, so raising
#: this silently loosens the loop detector.  At 3000 features the same threshold of
#: 50 matches is crossed by chance and produces ~200 m false closures on sequence
#: 00; at 500 it does not.  See README "Results".
ORB_FEATURES = 500

#: Distance used to compare ORB descriptors.  ``NORM_HAMMING2`` compares *pairs* of
#: bits, which is what OpenCV documents for ORB built with ``WTA_K=3`` or ``4``; with
#: the default ``WTA_K=2`` the textbook choice is ``NORM_HAMMING``.  The published
#: results use ``NORM_HAMMING2`` because that is what the original implementation
#: used.  Measured, the two are close on sequence 00 (500 features, loop closure on):
#: HAMMING2 gives 3.73 % / ATE 22.22 m, HAMMING gives 4.63 % / ATE 20.90 m.  Switch
#: this to ``cv2.NORM_HAMMING`` and the numbers in the README shift accordingly.
ORB_NORM = cv2.NORM_HAMMING2


def _create_detector(detector: str, nfeatures: int = ORB_FEATURES):
    name = detector.lower()
    if name == "sift":
        return cv2.SIFT_create()
    if name == "orb":
        return cv2.ORB_create(nfeatures=nfeatures)
    if name == "akaze":
        return cv2.AKAZE_create()
    if name == "surf":  # patented; only present in opencv-contrib non-free builds
        return cv2.xfeatures2d.SURF_create()
    raise ValueError(f"Unknown detector {detector!r}; expected sift, orb, akaze or surf.")


def extract_features(
    image: np.ndarray,
    detector: str = "orb",
    mask: Optional[np.ndarray] = None,
    nfeatures: int = ORB_FEATURES,
) -> Tuple[Sequence, np.ndarray]:
    """Detect keypoints and compute descriptors."""
    return _create_detector(detector, nfeatures).detectAndCompute(image, mask)


def match_features(
    des1: np.ndarray,
    des2: np.ndarray,
    matching: str = "BF",
    detector: str = "orb",
    sort: bool = True,
    k: int = 2,
) -> List:
    """k-NN match two descriptor sets.

    Binary descriptors (ORB, AKAZE) use Hamming distance, float descriptors
    (SIFT, SURF) use L2.  ``crossCheck`` stays off because the ratio test below
    needs the two nearest neighbours.
    """
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []

    binary = detector.lower() in {"orb", "akaze"}
    if matching.upper() == "BF":
        norm = ORB_NORM if detector.lower() == "orb" else (
            cv2.NORM_HAMMING if binary else cv2.NORM_L2
        )
        matcher = cv2.BFMatcher_create(norm, crossCheck=False)
    elif matching.upper() == "FLANN":
        if binary:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        else:
            index_params = dict(algorithm=1, trees=5)
        matcher = cv2.FlannBasedMatcher(index_params, dict(checks=50))
    else:
        raise ValueError(f"Unknown matching strategy {matching!r}; expected BF or FLANN.")

    matches = matcher.knnMatch(des1, des2, k=k)
    matches = [m for m in matches if len(m) == k]
    if sort and matches:
        matches = sorted(matches, key=lambda pair: pair[0].distance)
    return matches


def filter_matches(matches: Sequence, ratio: float = 0.5) -> List:
    """Lowe's ratio test: keep matches whose best neighbour clearly beats the second."""
    return [m for m, n in matches if m.distance <= ratio * n.distance]


def visualize_matches(image1, kp1, image2, kp2, matches, figsize=(16, 6)):
    """Draw matched keypoints across two images (returns a matplotlib figure)."""
    import matplotlib.pyplot as plt

    drawn = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=2)
    fig = plt.figure(figsize=figsize, dpi=100)
    plt.imshow(drawn)
    plt.axis("off")
    return fig
