# Part of stereo-slam-kitti, a GPL-3.0 derivative of
# https://github.com/FoamoftheSea/KITTI_visual_odometry. This module is original work.
"""Trajectory plots."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .metrics import to_4x4


def plot_trajectory_2d(
    trajectories: dict,
    title: str = "Trajectory (bird's-eye view)",
    loop_events: Optional[Sequence[dict]] = None,
    figsize=(8, 8),
):
    """Top-down x-z plot of one or more trajectories.

    ``trajectories`` maps a label to an ``(N, 3, 4)`` pose array, e.g.
    ``{"Ground truth": gt, "VO": traj}``.  KITTI's camera frame has y pointing
    down, so x-z is the ground plane.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for label, poses in trajectories.items():
        xyz = to_4x4(poses)[:, :3, 3]
        style = dict(lw=2, c="k") if "truth" in label.lower() else dict(lw=1.6)
        ax.plot(xyz[:, 0], xyz[:, 2], label=label, **style)

    if loop_events:
        estimate = next(
            (p for lbl, p in trajectories.items() if "truth" not in lbl.lower()), None
        )
        if estimate is not None:
            xyz = to_4x4(estimate)[:, :3, 3]
            idx = [e["frame"] for e in loop_events if e["frame"] < len(xyz)]
            if idx:
                ax.scatter(
                    xyz[idx, 0], xyz[idx, 2], marker="o", s=45, facecolors="none",
                    edgecolors="crimson", label="loop closure", zorder=5,
                )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_trajectory_3d(trajectories: dict, title: str = "Trajectory", figsize=(10, 10)):
    """3D view, mostly useful for sequences with real elevation change."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=-20, azim=270)
    for label, poses in trajectories.items():
        xyz = to_4x4(poses)[:, :3, 3]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=label)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_error_over_distance(gt: np.ndarray, estimated: np.ndarray, figsize=(8, 4)):
    """Position error as a function of distance travelled - shows drift growth."""
    import matplotlib.pyplot as plt

    from .metrics import trajectory_distances

    gt4, est4 = to_4x4(gt), to_4x4(estimated)
    n = min(len(gt4), len(est4))
    distance = trajectory_distances(gt4[:n])
    error = np.linalg.norm(gt4[:n, :3, 3] - est4[:n, :3, 3], axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(distance, error, lw=1.5)
    ax.set_xlabel("distance travelled [m]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Drift accumulation")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
