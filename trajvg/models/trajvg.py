import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder
from .layers.dpt_head import DPTHead
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
from .layers.track_head import TrackHead
from .layers.conv_head import ConvHead


class TrajVG(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
        ):
        super().__init__()

        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError

        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)
        ])
        self.dec_embed_dim = dec_embed_dim

        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-6)

        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = ConvHead(
            num_features=4, 
            dim_in=dec_embed_dim,
            projects=nn.Identity(),
            dim_out=[2, 1], 
            dim_proj=1024,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm='group_norm',
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True
        )

        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = ConvHead(
            num_features=4, 
            dim_in=dec_embed_dim,
            projects=nn.Identity(),
            dim_out=[1], 
            dim_proj=1024,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm='group_norm',
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True
        )

        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        self.track_decoder = DPTHead(dim_in=2 * self.dec_embed_dim, features=128, down_ratio=2)
        self.track_head = TrackHead()

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        output = []
        prev_hidden_norm = None

        hidden = hidden.reshape(B * N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(
            B * N, *self.register_token.shape[-2:]
        )

        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2, device=hidden.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

            if i % 2 == 0:
                curr_norm = hidden.reshape(B * N, hw, -1)
            else:
                curr_norm = hidden.reshape(B, N, hw, -1).reshape(B * N, hw, -1)

            if prev_hidden_norm is None:
                prev_hidden_norm = curr_norm
            else:
                temp_hidden = torch.cat([prev_hidden_norm, curr_norm], dim=-1).reshape(B, N, -1, hw)
                output.append(temp_hidden.float())
                prev_hidden_norm = None

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1), output

    @torch.no_grad()
    def forward(self, imgs, query_points=None, sample_stride=16):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, C, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        imgs_bn = imgs.reshape(B * N, C, H, W)
        hidden = self.encoder(imgs_bn, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos, output = self.decode(hidden, N, H, W)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        track_hidden = self.track_decoder(output, imgs.reshape(B, N, C, H, W), patch_start_idx=self.patch_start_idx)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            point_hidden = point_hidden.float()
            xy, z = self.point_head(point_hidden[:, self.patch_start_idx:].float(), patch_h=patch_h, patch_w=patch_w)
            xy = xy.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            z = z.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)

            z = torch.exp(z.clamp(max=15.0))
            local_points = torch.cat([xy * z, z], dim=-1)

            conf_hidden = conf_hidden.float()
            conf = self.conf_head(conf_hidden[:, self.patch_start_idx:].float(), patch_h=patch_h, patch_w=patch_w)[0]
            conf = conf.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)

            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(
                camera_hidden[:, self.patch_start_idx:], patch_h, patch_w
            ).reshape(B, N, 4, 4)

            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

            if query_points is None:
                offset_x = sample_stride // 2
                offset_y = sample_stride // 2
                if offset_x >= W: offset_x = 0
                if offset_y >= H: offset_y = 0
                xs = torch.arange(offset_x, W, sample_stride, device=imgs.device)
                ys = torch.arange(offset_y, H, sample_stride, device=imgs.device)
                grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
                coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).to(imgs.dtype)
                query_points = coords.unsqueeze(0).repeat(B, 1, 1)

            track_hidden = track_hidden.float()

            point_full = local_points.permute(0, 1, 4, 2, 3).reshape(B * N, 3, H, W)

            points_cam0 = local_points[:, 0]
            points_cam0 = points_cam0.permute(0, 3, 1, 2).contiguous()

            u = query_points[..., 0]
            v = query_points[..., 1]
            grid_x = (u / (W - 1)) * 2 - 1
            grid_y = (v / (H - 1)) * 2 - 1
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid_bn = grid.unsqueeze(2)

            query_points3d = F.grid_sample(points_cam0, grid_bn, mode='nearest', align_corners=True, padding_mode='border')
            query_points3d = query_points3d.squeeze(-1).permute(0, 2, 1)

            traj3d_uvd, vis_scores = self.track_head(query_points2d=query_points, query_points3d=query_points3d, fmaps=track_hidden, pointmaps=point_full)

            traj3d_preds = []
            for coords3d in traj3d_uvd:
                xyz_cam = torch.einsum('bnij, bnmj -> bnmi', camera_poses, homogenize_points(coords3d))[..., :3]
                traj3d_preds.append(xyz_cam)

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            vis_scores=vis_scores,
            traj3d_preds=traj3d_preds,
        )
