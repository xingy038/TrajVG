import os
import sys
import json
import struct
import zlib
import errno
import shutil
import argparse
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingTCPServer

DEFAULT_PORT = 8000


def compress_and_write(filename: str, header: dict, blob: bytes) -> None:
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(blob)


def process_point_cloud_data(npz_file: str, output_file: str, fps: float = 4.0) -> None:
    """
    New NPZ only:
      - world_points: (T, N, 3) float32
      - colors:       (T, N, 3) uint8
      - extrinsics:   (T, 4, 4) float32
      - trajectories: (T, M, 3) float32, invalid points are NaN
    """
    data = np.load(npz_file)

    required = ["world_points", "colors", "extrinsics", "trajectories"]
    for k in required:
        if k not in data:
            raise ValueError(f"NPZ missing key: {k}. Expect keys: {required}")

    world_points = np.ascontiguousarray(data["world_points"].astype(np.float32))   # (T,N,3)
    colors = np.ascontiguousarray(data["colors"].astype(np.uint8))                # (T,N,3)
    extrinsics = np.ascontiguousarray(data["extrinsics"].astype(np.float32))      # (T,4,4)
    trajectories = np.ascontiguousarray(data["trajectories"].astype(np.float32))  # (T,M,3)

    if world_points.ndim != 3 or world_points.shape[-1] != 3:
        raise ValueError(f"world_points shape must be (T,N,3), got {world_points.shape}")
    if colors.shape != world_points.shape:
        raise ValueError(f"colors shape must match world_points, got {colors.shape} vs {world_points.shape}")
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"extrinsics shape must be (T,4,4), got {extrinsics.shape}")
    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError(f"trajectories shape must be (T,M,3), got {trajectories.shape}")

    T, N, _ = world_points.shape
    T2, M, _ = trajectories.shape
    if T2 != T:
        raise ValueError(f"world_points T={T} but trajectories T={T2} mismatch")
    if extrinsics.shape[0] != T:
        raise ValueError(f"extrinsics T={extrinsics.shape[0]} mismatch world_points T={T}")

    arrays = {
        "world_points": world_points,
        "colors": colors,
        "extrinsics": extrinsics,
        "trajectories": trajectories,
    }

    header = {}
    blob_parts = []
    offset = 0
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        b = arr.tobytes()
        header[key] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "offset": int(offset),
            "length": int(len(b)),
        }
        blob_parts.append(b)
        offset += len(b)

    raw_blob = b"".join(blob_parts)
    compressed_blob = zlib.compress(raw_blob, level=9)

    header["meta"] = {
        "totalFrames": int(T),
        "numPoints": int(N),
        "numTrajectories": int(M),
        "baseFrameRate": float(fps),
    }

    compress_and_write(str(output_file), header, compressed_blob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input .npz file (new format)")
    parser.add_argument("--fps", type=float, default=4.0, help="Playback fps used by viewer")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help="HTTP server port")
    args = parser.parse_args()

    # viz.html is in the same directory as this script
    script_dir = Path(__file__).parent
    viz_html_path = script_dir / "viz.html"
    if not viz_html_path.exists():
        raise FileNotFoundError(f"viz.html not found next to script: {viz_html_path}")

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1) write data.bin
        process_point_cloud_data(args.input_file, temp_path / "data.bin", fps=args.fps)

        # 2) copy viz.html -> index.html
        shutil.copy(viz_html_path, temp_path / "index.html")

        # 3) serve
        os.chdir(temp_path)
        host = "127.0.0.1"
        port = args.port
        Handler = SimpleHTTPRequestHandler

        try:
            httpd = ThreadingTCPServer((host, port), Handler)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                print(f"Port {port} is in use, trying a random port...")
                httpd = ThreadingTCPServer((host, 0), Handler)
                port = httpd.server_address[1]
            else:
                raise

        print(f"Serving at http://{host}:{port}/?data=data.bin")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
        finally:
            httpd.server_close()


if __name__ == "__main__":
    main()
