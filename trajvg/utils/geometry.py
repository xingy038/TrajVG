import torch
import torch.nn.functional as F


def homogenize_points(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def depth_edge(
    depth: torch.Tensor,
    atol: float | None = None,
    rtol: float | None = None,
    kernel_size: int = 3,
    mask: torch.Tensor | None = None,
) -> torch.BoolTensor:
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2)
        diff = diff + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
    else:
        diff = F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2)
        diff = diff + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2)

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    return edge.reshape(*shape)
