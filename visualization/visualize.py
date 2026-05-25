# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

import os
import numpy as np
import json
import struct
import zlib
import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer
import socket
import sys

VIZ_HTML_PATH = Path(__file__).with_name("viz.html")
DEFAULT_PORT = 8000


def compress_and_write(filename, header, blob):
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(blob)


def process_point_cloud_data(npz_file, output_file, fps=4, fov_deg=60.0):
    data = np.load(npz_file)

    pose_key = "extrinsics" if "extrinsics" in data else "camera_poses"
    if "world_points" not in data or pose_key not in data:
        raise ValueError(
            "npz must contain 'world_points' (T,N,3) and 'extrinsics' or 'camera_poses' (T,4,4)."
        )

    pts_cam0 = data["world_points"].astype(np.float32)
    extrinsics = data[pose_key].astype(np.float32)

    if pts_cam0.ndim != 3 or pts_cam0.shape[2] != 3:
        raise ValueError("world_points must have shape (T, N, 3).")

    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError("extrinsics must have shape (T, 4, 4).")

    T, N, _ = pts_cam0.shape
    if extrinsics.shape[0] != T:
        raise ValueError("world_points and extrinsics must have the same T.")

    colors = None
    if "colors" in data:
        colors = data["colors"]
        if colors.ndim == 3 and colors.shape[0] == T and colors.shape[1] == N:
            if colors.dtype != np.uint8:
                colors = (np.clip(colors.astype(np.float32), 0.0, 1.0) * 255).astype(
                    np.uint8
                )
        else:
            raise ValueError(
                "If 'colors' is provided, it must have shape (T, N, 3), "
                f"but has shape {colors.shape}."
            )

    trajectories = None
    M = 0
    if "trajectories" in data:
        trajectories = data["trajectories"].astype(np.float32)
        if trajectories.ndim != 3 or trajectories.shape[0] != T or trajectories.shape[2] != 3:
            raise ValueError(
                "trajectories must have shape (T, M, 3), "
                f"but has shape {trajectories.shape}."
            )
        M = trajectories.shape[1]

    normalized_extrinsics = extrinsics
    inv_extrinsics = np.linalg.inv(extrinsics)

    flat_pts = pts_cam0.reshape(-1, 3)
    if flat_pts.size > 0:
        radii = np.linalg.norm(flat_pts, axis=1)
        max_r = float(np.percentile(radii, 95))
        cameraZ = max(3.0, 2.0 * max_r)
    else:
        cameraZ = 3.0

    arrays = {
        "points": pts_cam0,
        "extrinsics": normalized_extrinsics,
        "inv_extrinsics": inv_extrinsics,
    }

    if colors is not None:
        arrays["colors"] = colors.astype(np.uint8)

    if trajectories is not None:
        arrays["trajectories"] = trajectories

    header = {}
    blob_parts = []
    offset = 0
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        arr_bytes = arr.tobytes()
        header[key] = {
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "offset": offset,
            "length": len(arr_bytes),
        }
        blob_parts.append(arr_bytes)
        offset += len(arr_bytes)

    raw_blob = b"".join(blob_parts)
    compressed_blob = zlib.compress(raw_blob, level=9)

    header["meta"] = {
        "totalFrames": int(T),
        "baseFrameRate": int(fps),
        "numPoints": int(N),
        "numTrajectoryPoints": int(M),
        "fov": float(fov_deg),
        "original_aspect_ratio": 1.0,
        "fixed_aspect_ratio": 1.0,
        "cameraZ": float(cameraZ),
    }

    compress_and_write(output_file, header, compressed_blob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input .npz file")
    parser.add_argument("--fps", type=int, default=4, help="Base frame rate for playback")
    parser.add_argument(
        "--fov", type=float, default=60.0, help="Camera vertical field of view in degrees"
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to serve the visualization (default: {DEFAULT_PORT})",
    )

    args = parser.parse_args()

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        process_point_cloud_data(
            args.input_file,
            temp_path / "data.bin",
            fps=args.fps,
            fov_deg=args.fov,
        )

        if not VIZ_HTML_PATH.exists():
            raise FileNotFoundError(f"Missing visualization HTML: {VIZ_HTML_PATH}")
        shutil.copy(VIZ_HTML_PATH, temp_path / "index.html")

        os.chdir(temp_path)

        host = "127.0.0.1"
        port = args.port

        Handler = SimpleHTTPRequestHandler
        httpd = None

        try:
            httpd = ThreadingTCPServer((host, port), Handler)
        except OSError as e:
            # If the port is in use, try a random free port
            if hasattr(e, "errno") and e.errno == getattr(socket, "EADDRINUSE", None):
                print(f"Port {port} is already in use, trying a random port...")
                try:
                    httpd = ThreadingTCPServer((host, 0), Handler)
                    port = httpd.server_address[1]
                except OSError as e2:
                    print(f"Failed to bind to a random port: {e2}", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"Failed to start server: {e}", file=sys.stderr)
                sys.exit(1)

        if httpd:
            print(f"Serving at http://{host}:{port}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")
            finally:
                httpd.server_close()


if __name__ == "__main__":
    main()
