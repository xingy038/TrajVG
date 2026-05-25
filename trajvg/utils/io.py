import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _target_size(width: int, height: int, pixel_limit: int, multiple: int = 14) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: width={width}, height={height}")

    scale = math.sqrt(pixel_limit / float(width * height))
    target_w = width * scale
    target_h = height * scale
    grid_w = max(1, round(target_w / multiple))
    grid_h = max(1, round(target_h / multiple))

    while (grid_w * multiple) * (grid_h * multiple) > pixel_limit:
        if (grid_w / max(grid_h, 1)) > (target_w / max(target_h, 1e-6)):
            grid_w -= 1
        else:
            grid_h -= 1
        grid_w = max(1, grid_w)
        grid_h = max(1, grid_h)

    return grid_w * multiple, grid_h * multiple


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_image_dir(path: Path, interval: int) -> list[Image.Image]:
    filenames = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    return [Image.open(filename).convert("RGB") for filename in filenames[::interval]]


def _load_video(path: Path, interval: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {path}")

    frames = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % interval == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_idx += 1
    cap.release()
    return frames


def load_images_as_tensor(path: str | os.PathLike, interval: int = 1, pixel_limit: int = 255000) -> torch.Tensor:
    """Load an image directory or video as a float tensor with shape [T, 3, H, W]."""
    path = Path(path)
    interval = max(1, int(interval))

    if path.is_dir():
        frames = _load_image_dir(path, interval)
    elif path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        frames = _load_video(path, interval)
    else:
        raise ValueError(f"Input must be an image directory or video file: {path}")

    if not frames:
        raise ValueError(f"No frames were loaded from: {path}")

    target_w, target_h = _target_size(*frames[0].size, pixel_limit=pixel_limit)
    tensors = []
    for frame in frames:
        if frame.size != (target_w, target_h):
            frame = frame.resize((target_w, target_h), Image.Resampling.LANCZOS)
        tensors.append(_pil_to_tensor(frame))

    return torch.stack(tensors, dim=0)


def video_to_uint8(video: torch.Tensor) -> torch.Tensor:
    """Convert [T, 3, H, W] video tensor to uint8 without changing layout."""
    if video.dtype == torch.uint8:
        return video

    vmin = float(video.min().item())
    vmax = float(video.max().item())
    if 0.0 <= vmin and vmax <= 1.0:
        video = video * 255.0
    elif -1.0 <= vmin and vmax <= 1.0:
        video = video * 127.5 + 127.5
    elif not (0.0 <= vmin and vmax <= 255.0):
        video = (video - vmin) / (vmax - vmin + 1e-6) * 255.0

    return video.round().clamp(0, 255).to(torch.uint8)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _colorize_xyz(xyz: np.ndarray) -> np.ndarray:
    finite = np.isfinite(xyz).all(axis=1)
    colors = np.zeros_like(xyz, dtype=np.float32)
    if finite.any():
        valid_xyz = xyz[finite]
        lo = valid_xyz.min(axis=0)
        hi = valid_xyz.max(axis=0)
        colors[finite] = (valid_xyz - lo) / (hi - lo + 1e-8)
    return colors


def write_ply(xyz, rgb=None, path: str | os.PathLike = "output.ply") -> None:
    """Write xyz points and optional RGB colors to a PLY file."""
    xyz = _to_numpy(xyz).reshape(-1, 3).astype(np.float32)
    if rgb is None:
        rgb = _colorize_xyz(xyz)
    else:
        rgb = _to_numpy(rgb).reshape(-1, 3).astype(np.float32)
        if rgb.max(initial=0.0) > 1.0:
            rgb = rgb / 255.0

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = (rgb[finite].clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    normals = np.zeros_like(xyz, dtype=np.float32)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertices = np.empty(xyz.shape[0], dtype=dtype)
    vertices[:] = list(map(tuple, np.concatenate([xyz, normals, rgb], axis=1)))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))
