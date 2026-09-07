# Derived from https://github.com/FoamoftheSea/KITTI_visual_odometry (Nate Cibik),
# licensed under GPL-3.0. Refactored and extended; see README "Attribution".
"""Loading of KITTI odometry sequences (grayscale stereo pairs, calibration, poses).

The expected directory layout is the one produced by unpacking the official
KITTI odometry archives side by side::

    <data_root>/
        data_odometry_gray/dataset/sequences/<seq>/image_0/*.png
        data_odometry_gray/dataset/sequences/<seq>/image_1/*.png
        data_odometry_calib/dataset/sequences/<seq>/calib.txt
        data_odometry_poses/dataset/poses/<seq>.txt

The root is taken from the ``data_root`` argument, falling back to the
``KITTI_ROOT`` environment variable.  Ground-truth poses are only published for
sequences 00-10; for 11-21 ``gt`` is ``None``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


def _resolve_root(data_root: Optional[str]) -> Path:
    root = data_root or os.environ.get("KITTI_ROOT")
    if root is None:
        raise ValueError(
            "No KITTI root given. Pass data_root=... or set the KITTI_ROOT "
            "environment variable to the directory holding data_odometry_gray/, "
            "data_odometry_calib/ and data_odometry_poses/."
        )
    path = Path(root).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"KITTI root does not exist: {path}")
    return path


class KittiSequence:
    """One KITTI odometry sequence.

    Parameters
    ----------
    sequence:
        Two-character sequence id, e.g. ``"00"``.
    data_root:
        Directory holding the unpacked KITTI archives.  Defaults to ``$KITTI_ROOT``.
    low_memory:
        If ``True`` images are streamed from disk through :meth:`frames` instead of
        being held in RAM.  A full sequence of ~4500 stereo pairs is several GB
        decoded, so this is the right choice on a laptop.
    progress_bar:
        Show a progress bar while preloading (ignored when ``low_memory``).
    """

    def __init__(
        self,
        sequence: str = "00",
        data_root: Optional[str] = None,
        low_memory: bool = True,
        progress_bar: bool = False,
    ) -> None:
        root = _resolve_root(data_root)
        self.sequence = sequence
        self.low_memory = low_memory

        self.seq_dir = root / "data_odometry_gray" / "dataset" / "sequences" / sequence
        self.calib_file = (
            root / "data_odometry_calib" / "dataset" / "sequences" / sequence / "calib.txt"
        )
        self.poses_file = root / "data_odometry_poses" / "dataset" / "poses" / f"{sequence}.txt"

        left_dir = self.seq_dir / "image_0"
        right_dir = self.seq_dir / "image_1"
        if not left_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {left_dir}")

        self.left_image_files = sorted(left_dir.glob("*.png"))
        self.right_image_files = sorted(right_dir.glob("*.png"))
        if len(self.left_image_files) != len(self.right_image_files):
            raise RuntimeError(
                f"Stereo pair count mismatch in sequence {sequence}: "
                f"{len(self.left_image_files)} left vs {len(self.right_image_files)} right"
            )
        self.num_frames = len(self.left_image_files)

        self.P0, self.P1, self.P2, self.P3 = self._load_calibration()
        self.gt = self._load_poses()

        first = cv2.imread(str(self.left_image_files[0]), cv2.IMREAD_GRAYSCALE)
        if first is None:
            raise RuntimeError(f"Could not read {self.left_image_files[0]}")
        self.imheight, self.imwidth = first.shape[:2]

        self.images_left: Optional[list] = None
        self.images_right: Optional[list] = None
        if not low_memory:
            self._preload(progress_bar)

    # ------------------------------------------------------------------ loading

    def _load_calibration(self):
        if not self.calib_file.is_file():
            raise FileNotFoundError(f"Missing calibration file: {self.calib_file}")
        projections = {}
        with open(self.calib_file) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                key, _, values = line.partition(":")
                try:
                    projections[key.strip()] = np.fromstring(values, sep=" ").reshape(3, 4)
                except ValueError:
                    continue
        try:
            return (projections["P0"], projections["P1"], projections["P2"], projections["P3"])
        except KeyError as exc:  # pragma: no cover - malformed calib file
            raise RuntimeError(f"calib.txt is missing {exc} in {self.calib_file}") from exc

    def _load_poses(self) -> Optional[np.ndarray]:
        if not self.poses_file.is_file():
            # Sequences 11-21 are the held-out test set: no ground truth published.
            return None
        poses = np.loadtxt(self.poses_file)
        return poses.reshape(-1, 3, 4)

    def _preload(self, progress_bar: bool) -> None:
        indices = range(self.num_frames)
        if progress_bar:
            try:
                from tqdm import tqdm

                indices = tqdm(indices, desc=f"loading seq {self.sequence}")
            except ImportError:
                pass
        self.images_left = []
        self.images_right = []
        for i in indices:
            self.images_left.append(
                cv2.imread(str(self.left_image_files[i]), cv2.IMREAD_GRAYSCALE)
            )
            self.images_right.append(
                cv2.imread(str(self.right_image_files[i]), cv2.IMREAD_GRAYSCALE)
            )

    # ------------------------------------------------------------------ access

    def left(self, index: int) -> np.ndarray:
        if self.images_left is not None:
            return self.images_left[index]
        return cv2.imread(str(self.left_image_files[index]), cv2.IMREAD_GRAYSCALE)

    def right(self, index: int) -> np.ndarray:
        if self.images_right is not None:
            return self.images_right[index]
        return cv2.imread(str(self.right_image_files[index]), cv2.IMREAD_GRAYSCALE)

    def frames(self, num_frames: Optional[int] = None) -> Iterator[tuple]:
        """Yield ``(index, left, right, next_left)`` for consecutive frame pairs."""
        total = self.num_frames if num_frames is None else min(num_frames, self.num_frames)
        previous_left = self.left(0)
        for i in range(total - 1):
            left = previous_left
            right = self.right(i)
            next_left = self.left(i + 1)
            previous_left = next_left
            yield i, left, right, next_left

    def __len__(self) -> int:
        return self.num_frames

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"KittiSequence(sequence={self.sequence!r}, frames={self.num_frames}, "
            f"size={self.imwidth}x{self.imheight}, ground_truth={self.gt is not None})"
        )


def decompose_projection_matrix(p: np.ndarray):
    """Split a 3x4 projection matrix into intrinsics, rotation and translation."""
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    return k, r, t
