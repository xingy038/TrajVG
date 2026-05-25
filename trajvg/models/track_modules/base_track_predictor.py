import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from .blocks import EfficientUpdateFormer
from .utils import posenc, sample_features4d
from .knn_utils import KNNCorrFeature4D
class BaseTrackerPredictor(nn.Module):
    def __init__(
        self,
        stride=2,
        corr_levels=3,
        latent_dim=128,
        hidden_size=384,
        use_spaceatt=True,
        depth=6,
        predict_conf=True,
    ):
        super().__init__()
        self.stride = stride
        self.latent_dim = latent_dim
        self.corr_levels = corr_levels
        self.hidden_size = hidden_size
        self.predict_conf = predict_conf
        self.k_neighbors = 16

        self.corr_module = KNNCorrFeature4D(
            corr_levels=corr_levels,
            feat_dim=latent_dim,
            k_neighbors=self.k_neighbors,
            transformer_dim=latent_dim,
        )
        self.pcorr_dim = corr_levels * latent_dim

        self.emb_dim = 33 * 3 * 2
        
        vis_dim = 1
        self.updater_input_dim = vis_dim + self.emb_dim + self.pcorr_dim

        self.query_ref_token = nn.Parameter(
            torch.randn(1, 2, self.updater_input_dim)
        )

        self.updateformer = EfficientUpdateFormer(
            space_depth=depth if use_spaceatt else 0,
            time_depth=depth,
            input_dim=self.updater_input_dim,
            hidden_size=self.hidden_size,
            output_dim=4,
            mlp_ratio=4.0,
            add_space_attn=use_spaceatt,
            linear_layer_for_vis_conf=False
        )

        self.fmap_norm = nn.LayerNorm(self.latent_dim)

    def _normalize_pointmaps(self, pointmaps, query_points3d):
        B, S, C, H, W = pointmaps.shape
        pm_flat = pointmaps.view(B, S, C, -1)

        pm_mean = pm_flat.mean(dim=-1, keepdim=True)
        pm_std = pm_flat.std(dim=-1, keepdim=True)
        pm_std = pm_std.clamp(min=1e-6)

        min_pts = pm_mean - 3.0 * pm_std
        max_pts = pm_mean + 3.0 * pm_std
        scale = (max_pts - min_pts).clamp(min=1e-6)

        pm_01 = (pm_flat - min_pts) / scale
        pm_norm_flat = pm_01 * 2.0 - 1.0

        pointmaps_norm = pm_norm_flat.view(B, S, C, H, W)

        min0 = min_pts[:, 0]
        max0 = max_pts[:, 0]
        scale0 = (max0 - min0).clamp(min=1e-6)

        min0_coord = min0.squeeze(-1).unsqueeze(1)
        scale0_coord = scale0.squeeze(-1).unsqueeze(1)

        query_points3d_01 = (query_points3d - min0_coord) / scale0_coord
        query_points3d_norm = query_points3d_01 * 2.0 - 1.0

        return pointmaps_norm, min_pts, max_pts, query_points3d_norm

    def _denormalize_coords(self, coords_3d_norm, min_pts, max_pts):
        scale = (max_pts - min_pts).clamp(min=1e-6)

        min_pts_coord = min_pts.squeeze(-1).unsqueeze(2)
        scale_coord = scale.squeeze(-1).unsqueeze(2)

        coords_01 = (coords_3d_norm + 1.0) / 2.0
        coords = coords_01 * scale_coord + min_pts_coord
        return coords

    def forward(
        self,
        query_points2d,
        query_points3d,
        fmaps,
        pointmaps,
        iters=6,
        apply_sigmoid=True,
    ):
        B, N, _ = query_points3d.shape
        Bf, S, C, Hf, Wf = fmaps.shape
        assert Bf == B
        
        fmaps = fmaps.permute(0, 1, 3, 4, 2)
        fmaps = self.fmap_norm(fmaps)
        fmaps = fmaps.permute(0, 1, 4, 2, 3)

        if pointmaps.dim() == 4:
            pointmaps = pointmaps.view(B, S, 3, pointmaps.shape[-2], pointmaps.shape[-1])
        
        if (pointmaps.shape[-2:] != (Hf, Wf)):
            pm_flat = pointmaps.view(B * S, 3, pointmaps.shape[-2], pointmaps.shape[-1])
            pm_flat = F.interpolate(pm_flat, size=(Hf, Wf), mode="nearest-exact")
            pointmaps = pm_flat.view(B, S, 3, Hf, Wf)

        pointmaps_norm, min_pts, max_pts, qnorm = self._normalize_pointmaps(pointmaps, query_points3d)

        coords_3d = torch.zeros(B, S, N, 3, device=query_points3d.device, dtype=query_points3d.dtype)
        coords_3d[:, 0] = qnorm

        pcd_pyramid = [pointmaps_norm]
        feat_pyramid = [fmaps]
        
        for _ in range(self.corr_levels - 1):
            pm = pcd_pyramid[-1]
            Bp, Sp, Cp, Hp, Wp = pm.shape
            pm_flat = pm.reshape(Bp * Sp, Cp, Hp, Wp)
            pm_flat = F.interpolate(pm_flat, scale_factor=0.5, mode="nearest-exact")
            _, _, Hp2, Wp2 = pm_flat.shape
            pcd_pyramid.append(pm_flat.view(Bp, Sp, Cp, Hp2, Wp2))
            
            fm = feat_pyramid[-1]
            Bf2, Sf2, Cf2, Hf2, Wf2 = fm.shape
            fm_flat = fm.reshape(Bf2 * Sf2, Cf2, Hf2, Wf2)
            next_fm = F.avg_pool2d(fm_flat, kernel_size=2, stride=2)
            _, _, Hf3, Wf3 = next_fm.shape
            feat_pyramid.append(next_fm.view(Bf2, Sf2, Cf2, Hf3, Wf3))

        query_uv_0 = query_points2d[..., :2] / self.stride

        support_tokens_pyramid = self.corr_module.build_support_pyramid(
            pcd_pyramid=pcd_pyramid,
            feat_pyramid=feat_pyramid,
            query_coords_0=qnorm,
            query_uv_0=query_uv_0,
        )

        coord3d_preds = []
        vis = torch.zeros((B, S, N), device=fmaps.device).float()
        
        for _ in range(iters):
            curr_coords_3d = coords_3d.detach()
            vis = vis.detach()
            
            pcorr = self.corr_module(
                pcd_pyramid=pcd_pyramid,
                feat_pyramid=feat_pyramid,
                curr_coords_3d=curr_coords_3d,
                support_tokens_pyramid=support_tokens_pyramid,
            )

            rel3d = curr_coords_3d - curr_coords_3d[:, 0:1]
            rel3d_emb = posenc(rel3d, min_deg=-2, max_deg=14)

            pos_emb = posenc(curr_coords_3d, min_deg=-2, max_deg=14)

            x = torch.cat([
                vis[..., None],
                rel3d_emb,
                pos_emb,
                pcorr
            ], dim=-1)

            ref_tokens_seq = torch.cat([
                self.query_ref_token[:, 0:1], 
                self.query_ref_token[:, 1:2].expand(-1, S - 1, -1)
            ], dim=1) 
            
            x = x + ref_tokens_seq.unsqueeze(2)

            x = x.permute(0, 2, 1, 3) 
            delta = self.updateformer(x)
            
            delta = delta.permute(0, 2, 1, 3)
            
            coords_3d = curr_coords_3d + delta[..., :3]
            vis = vis + delta[..., 3]

            coord3d_preds.append(coords_3d)

        vis_pred = torch.sigmoid(vis) if apply_sigmoid else vis

        coord3d_preds = [self._denormalize_coords(c, min_pts, max_pts) for c in coord3d_preds]

        return coord3d_preds, vis_pred
