"""End-to-end smoke test on a synthetic KITTI-format sequence.

The images are random texture warped by a known disparity and a known forward
shift, so they are not photometrically realistic - the point is to check that the
dataset loader, stereo, feature, PnP and pose-accumulation stages fit together and
produce a finite trajectory of the right shape without any KITTI download.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from stereo_slam import KittiSequence, run_slam
from stereo_slam.stereo import left_border_mask

NUM_FRAMES = 6
HEIGHT, WIDTH = 200, 640
DISPARITY = 40
SHIFT = 3


def _write_synthetic_sequence(root, sequence="00"):
    rng = np.random.default_rng(7)
    seq_dir = root / "data_odometry_gray" / "dataset" / "sequences" / sequence
    (seq_dir / "image_0").mkdir(parents=True)
    (seq_dir / "image_1").mkdir(parents=True)

    world = rng.integers(0, 255, size=(HEIGHT, WIDTH + 200), dtype=np.uint8)
    world = cv2.GaussianBlur(world, (3, 3), 0)

    for i in range(NUM_FRAMES):
        left = world[:, i * SHIFT : i * SHIFT + WIDTH]
        right = world[:, i * SHIFT + DISPARITY : i * SHIFT + DISPARITY + WIDTH]
        cv2.imwrite(str(seq_dir / "image_0" / f"{i:06d}.png"), left)
        cv2.imwrite(str(seq_dir / "image_1" / f"{i:06d}.png"), right)

    calib_dir = root / "data_odometry_calib" / "dataset" / "sequences" / sequence
    calib_dir.mkdir(parents=True)
    fx, cx, cy = 718.856, WIDTH / 2, HEIGHT / 2
    P0 = f"P0: {fx} 0 {cx} 0 0 {fx} {cy} 0 0 0 1 0"
    P1 = f"P1: {fx} 0 {cx} {-fx * 0.54} 0 {fx} {cy} 0 0 0 1 0"
    (calib_dir / "calib.txt").write_text("\n".join([P0, P1, P0.replace("P0", "P2"),
                                                    P1.replace("P1", "P3")]) + "\n")

    poses_dir = root / "data_odometry_poses" / "dataset" / "poses"
    poses_dir.mkdir(parents=True)
    poses = []
    for i in range(NUM_FRAMES):
        T = np.eye(4)
        T[2, 3] = i * 0.5
        poses.append(" ".join(f"{v}" for v in T[:3, :].ravel()))
    (poses_dir / f"{sequence}.txt").write_text("\n".join(poses) + "\n")
    return root


def test_sequence_loader_reads_synthetic_data(tmp_path):
    root = _write_synthetic_sequence(tmp_path)
    sequence = KittiSequence("00", data_root=str(root), low_memory=True)

    assert sequence.num_frames == NUM_FRAMES
    assert (sequence.imheight, sequence.imwidth) == (HEIGHT, WIDTH)
    assert sequence.P0.shape == (3, 4)
    assert sequence.gt.shape == (NUM_FRAMES, 3, 4)
    assert sequence.left(0).shape == (HEIGHT, WIDTH)


def test_preloaded_and_streamed_frames_agree(tmp_path):
    root = _write_synthetic_sequence(tmp_path)
    streamed = KittiSequence("00", data_root=str(root), low_memory=True)
    preloaded = KittiSequence("00", data_root=str(root), low_memory=False)
    for i in range(NUM_FRAMES):
        assert np.array_equal(streamed.left(i), preloaded.left(i))
        assert np.array_equal(streamed.right(i), preloaded.right(i))


def test_border_mask_blanks_the_left_band():
    mask = left_border_mask(HEIGHT, WIDTH, border=96)
    assert mask[:, :96].max() == 0
    assert mask[:, 96:].min() == 255


def test_pipeline_produces_a_finite_trajectory(tmp_path):
    root = _write_synthetic_sequence(tmp_path)
    sequence = KittiSequence("00", data_root=str(root), low_memory=True)

    result = run_slam(sequence, detector="orb", stereo_matcher="bm",
                      loop_closure=False, verbose=False)
    trajectory = result["trajectory"]

    assert trajectory.shape == (NUM_FRAMES, 3, 4)
    assert np.isfinite(trajectory).all()
    assert np.allclose(trajectory[0], np.eye(4)[:3, :])


def test_loop_closer_registers_keyframes(tmp_path):
    root = _write_synthetic_sequence(tmp_path)
    sequence = KittiSequence("00", data_root=str(root), low_memory=True)

    result = run_slam(sequence, detector="orb", stereo_matcher="bm",
                      loop_closure=True, verbose=False)
    closer = result["loop_closer"]

    assert closer is not None
    assert len(closer) >= 1
