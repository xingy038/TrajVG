#!/usr/bin/env python3
import argparse
from pathlib import Path

from viz import process_point_cloud_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TrajVG NPZ to viz data.bin without starting a server."
    )
    parser.add_argument("input_npz", help="Path to input .npz file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .bin path (default: same name as input, .bin)",
    )
    parser.add_argument("--fps", type=float, default=4.0, help="Playback fps")
    args = parser.parse_args()

    input_path = Path(args.input_npz)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".bin")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_point_cloud_data(str(input_path), str(output_path), fps=args.fps)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
