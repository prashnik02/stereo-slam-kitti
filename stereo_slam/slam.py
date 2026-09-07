# Derived from https://github.com/FoamoftheSea/KITTI_visual_odometry (Nate Cibik),
# licensed under GPL-3.0. Refactored and extended; see README "Attribution".
"""The pipeline: stereo depth -> features -> PnP motion -> pose accumulation."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .dataset import KittiSequence, decompose_projection_matrix
from .features import extract_features, filter_matches, match_features
from .loop_closure import KeyframeLoopCloser
from .odometry import estimate_motion, to_homogeneous
from .stereo import left_border_mask, stereo_to_depth


def run_slam(
    sequence: KittiSequence,
    detector: str = "orb",
    matching: str = "BF",
    stereo_matcher: str = "sgbm",
    match_ratio: Optional[float] = 0.5,
    loop_closure: bool = True,
    num_frames: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    max_depth: float = 3000.0,
    verbose: bool = True,
    closer_kwargs: Optional[dict] = None,
):
    """Run visual odometry (optionally with loop closure) over a sequence.

    Returns a dict with the estimated ``trajectory`` as an ``(N, 3, 4)`` array of
    camera-to-world poses, the ``KeyframeLoopCloser`` (or ``None``), per-frame
    timings, and the indices of any frames where motion estimation failed and the
    previous motion had to be carried forward.
    """
    total = sequence.num_frames if num_frames is None else min(num_frames, sequence.num_frames)
    if mask is None:
        mask = left_border_mask(sequence.imheight, sequence.imwidth)

    k_left, _, _ = decompose_projection_matrix(sequence.P0)

    closer = None
    if loop_closure:
        kwargs = dict(detector=detector, matching=matching, match_ratio=match_ratio or 0.5)
        kwargs.update(closer_kwargs or {})
        closer = KeyframeLoopCloser(**kwargs)

    T_total = np.eye(4)
    trajectory = np.zeros((total, 3, 4))
    trajectory[0] = T_total[:3, :]

    last_transform = np.eye(4)
    failures = []
    frame_times = []

    if verbose:
        print(
            f"sequence {sequence.sequence}: {total} frames | stereo={stereo_matcher} "
            f"| detector={detector} | matching={matching} "
            f"| loop_closure={'on' if loop_closure else 'off'}"
        )

    for i, image_left, image_right, image_next in sequence.frames(total):
        start = time.perf_counter()

        depth = stereo_to_depth(
            image_left, image_right, sequence.P0, sequence.P1, matcher=stereo_matcher
        )

        kp0, des0 = extract_features(image_left, detector, mask)
        kp1, des1 = extract_features(image_next, detector, mask)

        raw_matches = match_features(des0, des1, matching=matching, detector=detector, sort=True)
        matches = filter_matches(raw_matches, match_ratio) if match_ratio else [
            pair[0] for pair in raw_matches
        ]

        try:
            rmat, tvec, _, _ = estimate_motion(
                matches, kp0, kp1, k_left, depth, max_depth=max_depth
            )
            transform = to_homogeneous(rmat, tvec)
            last_transform = transform
        except (ValueError, RuntimeError) as exc:
            # Too few matches or a degenerate configuration: assume the motion
            # continued as it did in the previous frame rather than dropping a pose.
            failures.append(i)
            transform = last_transform
            if verbose:
                print(f"  frame {i}: motion estimation failed ({exc}); reusing last transform")

        # The PnP transform maps frame i into frame i+1, so the camera-to-world
        # pose accumulates with its inverse.
        T_total = T_total.dot(np.linalg.inv(transform))

        if closer is not None:
            T_total = closer.update(i, T_total, kp0, des0)

        trajectory[i + 1, :, :] = T_total[:3, :]
        frame_times.append(time.perf_counter() - start)

        if verbose and (i + 1) % 100 == 0:
            fps = 1.0 / np.mean(frame_times[-100:])
            print(f"  frame {i + 1}/{total - 1}  ({fps:.1f} fps)")

    if verbose:
        print(
            f"done: {np.mean(frame_times):.3f}s/frame, {len(failures)} failed frames"
            + (f", {len(closer.loop_events)} loop closures" if closer else "")
        )

    return {
        "trajectory": trajectory,
        "loop_closer": closer,
        "frame_times": np.array(frame_times),
        "failed_frames": failures,
    }
