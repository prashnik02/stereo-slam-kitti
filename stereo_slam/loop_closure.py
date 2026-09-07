# Part of stereo-slam-kitti, a GPL-3.0 derivative of
# https://github.com/FoamoftheSea/KITTI_visual_odometry. This module is original work.
"""Appearance-based loop detection over a keyframe database.

This is a deliberately simple place-recognition scheme: every keyframe stores the
descriptors of its left image, and a candidate frame is matched against all of
them by brute force.  Enough surviving matches means "we have been here before".

Two honest caveats, both of which matter if you read the numbers in the README:

* Correction is a **snap**, not an optimisation.  On detection the current pose is
  overwritten with the stored keyframe pose, which removes the accumulated offset
  at that instant but leaves every intermediate pose untouched and introduces a
  discontinuity in the trajectory.  Proper loop closure distributes the error over
  the loop with a pose-graph optimisation (see README, "Limitations").
* Brute-force descriptor matching against every keyframe is O(N) per query.  A
  bag-of-words vocabulary (DBoW2) or a learned global descriptor is what real
  systems use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .features import filter_matches, match_features


@dataclass
class Keyframe:
    """A remembered place: its pose in the world and the descriptors that identify it."""

    index: int
    pose: np.ndarray
    keypoints: Sequence = field(repr=False)
    descriptors: np.ndarray = field(repr=False)
    image: Optional[np.ndarray] = field(default=None, repr=False)


class KeyframeLoopCloser:
    """Maintains the keyframe database and reports loop closures.

    Parameters
    ----------
    check_interval:
        How many frames to travel before querying the database again.
    loop_threshold:
        Match count above which two frames are declared the same place.
    revisit_threshold:
        Weaker match count meaning "close to a known place" - not a loop, but a
        reason not to spend a new keyframe here yet.
    min_frames_between_loops:
        Guard against re-triggering on the keyframe we have just closed against.
    """

    def __init__(
        self,
        detector: str = "orb",
        matching: str = "BF",
        match_ratio: float = 0.5,
        check_interval: int = 20,
        loop_threshold: int = 50,
        revisit_threshold: int = 5,
        min_frames_between_loops: int = 50,
    ) -> None:
        self.detector = detector
        self.matching = matching
        self.match_ratio = match_ratio
        self.check_interval = check_interval
        self.loop_threshold = loop_threshold
        self.revisit_threshold = revisit_threshold
        self.min_frames_between_loops = min_frames_between_loops

        self.keyframes: List[Keyframe] = []
        self.loop_events: List[dict] = []
        self._since_check = 0

    # ------------------------------------------------------------------ helpers

    def add_keyframe(self, index, pose, keypoints, descriptors, image=None) -> Keyframe:
        keyframe = Keyframe(index, np.array(pose, copy=True), keypoints, descriptors, image)
        self.keyframes.append(keyframe)
        self._since_check = 0
        return keyframe

    def _score(self, descriptors, keyframe: Keyframe) -> int:
        raw = match_features(
            descriptors,
            keyframe.descriptors,
            matching=self.matching,
            detector=self.detector,
            sort=True,
        )
        return len(filter_matches(raw, self.match_ratio))

    # ------------------------------------------------------------------ query

    def update(self, index, pose, keypoints, descriptors, image=None) -> np.ndarray:
        """Feed one frame in; get back the (possibly corrected) pose.

        The database is only queried every ``check_interval`` frames, since
        querying is the expensive part of the loop.
        """
        if not self.keyframes:
            self.add_keyframe(index, pose, keypoints, descriptors, image)
            return pose

        self._since_check += 1
        if self._since_check < self.check_interval:
            return pose

        best_score = 0
        for keyframe in self.keyframes:
            if index - keyframe.index < self.min_frames_between_loops:
                continue
            score = self._score(descriptors, keyframe)
            best_score = max(best_score, score)

            if score > self.loop_threshold:
                self.loop_events.append(
                    {
                        "frame": index,
                        "keyframe": keyframe.index,
                        "matches": score,
                        "position_jump": float(
                            np.linalg.norm(pose[:3, 3] - keyframe.pose[:3, 3])
                        ),
                    }
                )
                self._since_check = 0
                return np.array(keyframe.pose, copy=True)

        if best_score > self.revisit_threshold:
            # Near something we already know - hold off on a new keyframe.
            self._since_check = max(self._since_check - 1, 0)
            return pose

        self.add_keyframe(index, pose, keypoints, descriptors, image)
        return pose

    def __len__(self) -> int:
        return len(self.keyframes)
