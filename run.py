#!/usr/bin/env python3
"""Run stereo visual odometry / SLAM on one or more KITTI sequences.

Examples
--------
    python run.py --sequence 00 --detector orb --stereo-matcher bm
    python run.py --sequence 00 05 07 --no-loop-closure --output results/
    python run.py --sequence 00 --frames 300 --lengths 10 20 50 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stereo_slam import KittiSequence, run_slam
from stereo_slam.metrics import evaluate, format_report
from stereo_slam.plotting import plot_error_over_distance, plot_trajectory_2d


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stereo visual odometry / SLAM on KITTI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sequence", nargs="+", default=["00"], help="KITTI sequence id(s)")
    parser.add_argument(
        "--data-root",
        default=None,
        help="KITTI root directory (defaults to the KITTI_ROOT environment variable)",
    )
    parser.add_argument("--detector", default="orb", choices=["orb", "sift", "akaze", "surf"])
    parser.add_argument("--matching", default="BF", choices=["BF", "FLANN"])
    parser.add_argument("--stereo-matcher", default="sgbm", choices=["bm", "sgbm"])
    parser.add_argument(
        "--match-ratio", type=float, default=0.5, help="Lowe ratio test threshold"
    )
    parser.add_argument(
        "--no-loop-closure", action="store_true", help="Pure odometry, no keyframe database"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="Only process the first N frames"
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the whole sequence into RAM (fast, needs several GB)",
    )
    parser.add_argument(
        "--lengths",
        type=float,
        nargs="+",
        default=[100, 200, 300, 400, 500, 600, 700, 800],
        help="Sub-sequence lengths for the KITTI metric (use shorter ones with --frames)",
    )
    parser.add_argument(
        "--align", action="store_true", help="Rigidly align to ground truth before computing ATE"
    )
    parser.add_argument("--output", default="results", help="Directory for poses, metrics, plots")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for sequence_id in args.sequence:
        sequence = KittiSequence(
            sequence_id,
            data_root=args.data_root,
            low_memory=not args.preload,
            progress_bar=not args.quiet,
        )
        if not args.quiet:
            print(f"\n{sequence}")

        result = run_slam(
            sequence,
            detector=args.detector,
            matching=args.matching,
            stereo_matcher=args.stereo_matcher,
            match_ratio=args.match_ratio,
            loop_closure=not args.no_loop_closure,
            num_frames=args.frames,
            verbose=not args.quiet,
        )
        trajectory = result["trajectory"]

        tag = f"{sequence_id}_{args.detector}_{args.stereo_matcher}"
        tag += "_noloop" if args.no_loop_closure else "_loop"
        np.savetxt(output_dir / f"poses_{tag}.txt", trajectory.reshape(-1, 12))

        if sequence.gt is None:
            print(f"sequence {sequence_id}: no ground truth published, skipping evaluation")
            continue

        gt = sequence.gt[: len(trajectory)]
        metrics = evaluate(gt, trajectory, lengths=args.lengths, align=args.align)
        metrics["config"] = vars(args) | {"sequence": sequence_id}
        metrics["runtime"] = {
            "mean_frame_time_s": float(result["frame_times"].mean()),
            "fps": float(1.0 / result["frame_times"].mean()),
            "failed_frames": len(result["failed_frames"]),
        }
        closer = result["loop_closer"]
        if closer is not None:
            metrics["loop_closure"] = {
                "keyframes": len(closer),
                "detections": len(closer.loop_events),
                "events": closer.loop_events,
            }

        with open(output_dir / f"metrics_{tag}.json", "w") as handle:
            json.dump(metrics, handle, indent=2, default=str)

        print()
        print(format_report(metrics, title=f"Sequence {sequence_id} ({tag})"))
        summary[tag] = metrics

        if not args.no_plots:
            import matplotlib

            matplotlib.use("Agg")
            events = closer.loop_events if closer is not None else None
            fig = plot_trajectory_2d(
                {"Ground truth": gt, "Estimate": trajectory},
                title=f"KITTI sequence {sequence_id} - {tag}",
                loop_events=events,
            )
            fig.savefig(output_dir / f"trajectory_{tag}.png", dpi=150)
            fig = plot_error_over_distance(gt, trajectory)
            fig.savefig(output_dir / f"drift_{tag}.png", dpi=150)

    if summary:
        with open(output_dir / "summary.json", "w") as handle:
            json.dump(summary, handle, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
