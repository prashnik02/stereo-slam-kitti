"""Stereo visual odometry and SLAM on the KITTI odometry benchmark."""

from .dataset import KittiSequence
from .stereo import compute_disparity, disparity_to_depth, stereo_to_depth
from .features import extract_features, match_features, filter_matches, visualize_matches
from .odometry import estimate_motion
from .loop_closure import Keyframe, KeyframeLoopCloser
from .slam import run_slam
from .metrics import kitti_odometry_error, absolute_trajectory_error, relative_pose_error, evaluate

__all__ = [
    "KittiSequence",
    "compute_disparity",
    "disparity_to_depth",
    "stereo_to_depth",
    "extract_features",
    "match_features",
    "filter_matches",
    "visualize_matches",
    "estimate_motion",
    "Keyframe",
    "KeyframeLoopCloser",
    "run_slam",
    "kitti_odometry_error",
    "absolute_trajectory_error",
    "relative_pose_error",
    "evaluate",
]

__version__ = "0.1.0"
