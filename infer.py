import argparse
import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from trajvg import TrajVG
from trajvg.utils import depth_edge, load_images_as_tensor, video_to_uint8, write_ply


def load_mask(mask_path: str, height: int, width: int, device: torch.device) -> torch.Tensor:
    path = Path(mask_path)
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        image = Image.fromarray((mask != 0).astype(np.uint8) * 255)
    else:
        image = Image.open(path).convert("L")

    if image.size != (width, height):
        image = image.resize((width, height), resample=Image.NEAREST)

    return torch.from_numpy(np.asarray(image) > 0).to(device=device)


def query_points_from_mask(mask: torch.Tensor, stride: int, max_points: int) -> torch.Tensor | None:
    height, width = mask.shape
    offset_x = min(stride // 2, width - 1)
    offset_y = min(stride // 2, height - 1)
    xs = torch.arange(offset_x, width, stride, device=mask.device)
    ys = torch.arange(offset_y, height, stride, device=mask.device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    coords = coords[mask[coords[:, 1], coords[:, 0]]]

    if coords.numel() == 0:
        yx = torch.nonzero(mask, as_tuple=False)
        if yx.numel() == 0:
            return None
        coords = yx[:, [1, 0]]

    if coords.shape[0] > max_points:
        step = max(1, coords.shape[0] // max_points)
        coords = coords[::step][:max_points]

    return coords.unsqueeze(0).to(dtype=torch.float32)


def load_state_dict(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path, device=str(device))
    else:
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)

    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")

    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def load_model(args, device: torch.device) -> TrajVG:
    if args.checkpoint is None and args.repo_id is None:
        raise ValueError("Provide --checkpoint or --repo-id.")

    if args.repo_id is not None:
        model = TrajVG.from_pretrained(args.repo_id)
    else:
        model = TrajVG()
        state = load_state_dict(args.checkpoint, device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

    return model.to(device).eval()


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    major, _ = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def save_results(res: dict, frames: torch.Tensor, args) -> None:
    conf = torch.sigmoid(res["conf"][..., 0]) > args.conf_thresh
    non_edge = ~depth_edge(res["local_points"][..., 2], rtol=args.depth_edge_rtol)
    point_mask = torch.logical_and(conf, non_edge)[0]

    colors = frames.permute(0, 2, 3, 1)
    write_ply(res["points"][0][point_mask].detach().cpu(), colors[point_mask].detach().cpu(), args.output_ply)
    print(f"Saved point cloud: {args.output_ply}")

    if args.output_npz is None:
        return

    video_u8 = video_to_uint8(frames).permute(0, 2, 3, 1).cpu().numpy()
    trajectories = res["traj3d_preds"][-1][0].detach().cpu().float().numpy()
    visibility = res["vis_scores"]
    if visibility.dim() == 4 and visibility.shape[-1] == 1:
        visibility = visibility[..., 0]
    visibility = (visibility[0] >= args.traj_conf).detach().cpu().numpy().astype(bool)
    trajectories[~visibility] = np.nan

    points = res["points"][0].detach().cpu().float().numpy()
    poses = res["camera_poses"][0].detach().cpu().float().numpy()
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        world_points=points.reshape(points.shape[0], -1, 3).astype(np.float32),
        extrinsics=poses.astype(np.float32),
        colors=video_u8.reshape(video_u8.shape[0], -1, 3).astype(np.uint8),
        trajectories=trajectories.astype(np.float32),
        visibility=visibility,
    )
    print(f"Saved trajectories: {args.output_npz}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TrajVG inference on an image directory or video.")
    parser.add_argument("--input", required=True, help="Path to an image directory or video file.")
    parser.add_argument("--checkpoint", default=None, help="Path to a .bin/.pt/.pth/.safetensors checkpoint.")
    parser.add_argument("--repo-id", default=None, help="Optional Hugging Face repo id for TrajVG.from_pretrained().")
    parser.add_argument("--output-ply", default="outputs/result.ply", help="Output point cloud path.")
    parser.add_argument("--output-npz", default="outputs/result.npz", help="Optional output NPZ path. Use '' to disable.")
    parser.add_argument("--device", default="cuda", help="Device, for example cuda or cpu.")
    parser.add_argument("--interval", type=int, default=1, help="Frame sampling interval.")
    parser.add_argument("--pixel-limit", type=int, default=255000, help="Resize budget before inference.")
    parser.add_argument("--resize-height", type=int, default=None, help="Optional exact resize height.")
    parser.add_argument("--resize-width", type=int, default=None, help="Optional exact resize width.")
    parser.add_argument("--query-mask", default=None, help="Optional first-frame mask image or .npy file.")
    parser.add_argument("--max-query-points", type=int, default=8192, help="Maximum mask-sampled query points.")
    parser.add_argument("--traj-stride", type=int, default=20, help="Grid stride for default query points.")
    parser.add_argument("--traj-conf", type=float, default=0.2, help="Trajectory visibility threshold.")
    parser.add_argument("--conf-thresh", type=float, default=0.1, help="Point cloud confidence threshold.")
    parser.add_argument("--depth-edge-rtol", type=float, default=0.04, help="Depth-edge rejection threshold.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast.")
    args = parser.parse_args()
    if args.output_npz == "":
        args.output_npz = None
    return args


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    device = torch.device(args.device)
    model = load_model(args, device)
    frames = load_images_as_tensor(args.input, interval=args.interval, pixel_limit=args.pixel_limit).to(device)

    if args.resize_height is not None or args.resize_width is not None:
        if args.resize_height is None or args.resize_width is None:
            raise ValueError("Set both --resize-height and --resize-width.")
        frames = F.interpolate(frames, size=(args.resize_height, args.resize_width), mode="bilinear", align_corners=False)

    _, _, height, width = frames.shape
    query_points = None
    if args.query_mask is not None:
        mask = load_mask(args.query_mask, height, width, device)
        query_points = query_points_from_mask(mask, args.traj_stride, args.max_query_points)
        if query_points is None:
            print("Query mask is empty; falling back to grid queries.")

    with torch.no_grad(), autocast_context(device, enabled=not args.no_amp):
        result = model(frames.unsqueeze(0), query_points=query_points, sample_stride=args.traj_stride)

    save_results(result, frames, args)


if __name__ == "__main__":
    main()
