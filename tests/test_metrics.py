"""Tests for the trajectory metrics, checked against analytically known answers."""

import numpy as np
import pytest

from stereo_slam.metrics import (
    absolute_trajectory_error,
    kitti_odometry_error,
    relative_pose_error,
    rotation_angle,
    to_4x4,
    trajectory_distances,
    umeyama_alignment,
)


def straight_line(n=200, step=1.0):
    """A trajectory driving straight along +z at `step` metres per frame."""
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 2, 3] = np.arange(n) * step
    return poses


def test_to_4x4_from_3x4():
    poses = np.random.rand(5, 3, 4)
    out = to_4x4(poses)
    assert out.shape == (5, 4, 4)
    assert np.allclose(out[:, 3, :], [0, 0, 0, 1])
    assert np.allclose(out[:, :3, :], poses)


def test_trajectory_distances():
    poses = straight_line(11, step=2.0)
    distances = trajectory_distances(poses)
    assert distances[0] == 0.0
    assert np.isclose(distances[-1], 20.0)


def test_rotation_angle():
    theta = np.pi / 6
    R = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    assert np.isclose(rotation_angle(R), theta)
    assert np.isclose(rotation_angle(np.eye(3)), 0.0)


def test_identical_trajectories_have_zero_error():
    gt = straight_line(300)
    ate = absolute_trajectory_error(gt, gt)
    assert ate["rmse"] == pytest.approx(0.0)

    rpe = relative_pose_error(gt, gt, delta=1)
    assert rpe["translation_rmse"] == pytest.approx(0.0)

    kitti = kitti_odometry_error(gt, gt, lengths=(100, 200))
    assert kitti["translation_error_percent"] == pytest.approx(0.0)
    assert kitti["rotation_error_deg_per_m"] == pytest.approx(0.0)


def test_constant_scale_error_gives_expected_kitti_percentage():
    """An estimate that under-travels by 10% must report ~10% translation error."""
    gt = straight_line(1000)
    est = straight_line(1000, step=0.9)
    kitti = kitti_odometry_error(gt, est, lengths=(100, 200, 400))
    assert kitti["translation_error_percent"] == pytest.approx(10.0, abs=0.5)
    assert kitti["rotation_error_deg_per_m"] == pytest.approx(0.0, abs=1e-9)


def test_constant_offset_does_not_affect_relative_metrics():
    """A rigid offset inflates ATE but leaves relative errors untouched."""
    gt = straight_line(500)
    est = gt.copy()
    est[:, 0, 3] += 5.0

    assert absolute_trajectory_error(gt, est)["rmse"] == pytest.approx(5.0)
    assert relative_pose_error(gt, est, delta=1)["translation_rmse"] == pytest.approx(0.0)
    assert kitti_odometry_error(gt, est, lengths=(100,))[
        "translation_error_percent"
    ] == pytest.approx(0.0, abs=1e-6)


def test_alignment_removes_a_rigid_transform():
    gt = straight_line(200)
    theta = 0.3
    R = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    est = gt.copy()
    est[:, :3, 3] = (R @ gt[:, :3, 3].T).T + np.array([3.0, -1.0, 2.0])

    assert absolute_trajectory_error(gt, est, align=False)["rmse"] > 1.0
    assert absolute_trajectory_error(gt, est, align=True)["rmse"] == pytest.approx(0.0, abs=1e-8)


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 3))
    theta = 0.7
    R_true = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    t_true = np.array([1.0, 2.0, 3.0])
    s_true = 2.5
    target = (s_true * (R_true @ source.T)).T + t_true

    R, t, s = umeyama_alignment(source, target, with_scale=True)
    assert np.allclose(R, R_true, atol=1e-8)
    assert np.allclose(t, t_true, atol=1e-8)
    assert s == pytest.approx(s_true)


def test_short_trajectory_raises_a_helpful_error():
    gt = straight_line(20)
    with pytest.raises(ValueError, match="No sub-sequence"):
        kitti_odometry_error(gt, gt, lengths=(100, 200))


def test_final_drift_matches_last_position_error():
    gt = straight_line(100)
    est = straight_line(100, step=0.95)
    ate = absolute_trajectory_error(gt, est)
    expected = np.linalg.norm(gt[-1, :3, 3] - est[-1, :3, 3])
    assert ate["final_drift"] == pytest.approx(expected)
