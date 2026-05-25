import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Tuple
import math

from .modules import Mlp
from .utils import posenc, sample_features4d

try:
    import third_party.pointops2.functions.pointops as pointops
except ImportError:
    try:
        import pointops2.pointops as pointops
    except ImportError:
        pointops = None


def _knn(k: int, queries: torch.Tensor, references: torch.Tensor):
    if pointops is not None and queries.is_cuda and references.is_cuda:
        queries = queries.contiguous()
        references = references.contiguous()
        B, N, _ = queries.shape
        M = references.shape[1]
        
        q_offset = (torch.arange(B, device=queries.device, dtype=torch.int32) + 1) * N
        r_offset = (torch.arange(B, device=references.device, dtype=torch.int32) + 1) * M
        
        idx, dist = pointops.knnquery(
            k, 
            references.view(-1, 3), 
            queries.view(-1, 3), 
            r_offset, 
            q_offset
        )
        idx = idx.view(B, N, k)
        
        batch_starts = torch.arange(B, device=queries.device)[:, None, None] * M
        idx = idx - batch_starts.int()
        
        return dist, idx.long()
    else:
        dists = torch.cdist(queries, references)
        knn_dists, knn_indices = torch.topk(dists, k, dim=-1, largest=False, sorted=True)
        return knn_dists, knn_indices


class KNNCorrFeature4D(nn.Module):
    def __init__(
        self, 
        corr_levels: int, 
        feat_dim: int, 
        k_neighbors: int,
        transformer_dim: int,
    ):
        super().__init__()
        self.corr_levels = corr_levels
        self.k_neighbors = k_neighbors
        self.feat_dim = feat_dim
        self.transformer_dim = transformer_dim
        
        self.in_norm = nn.LayerNorm(feat_dim)
        
        posenc_dim = 4 * 33 
        
        self.posenc_mlps = nn.ModuleList([
            Mlp(
                in_features=posenc_dim,
                out_features=transformer_dim,
                hidden_features=transformer_dim
            )
            for _ in range(corr_levels)
        ])

        self.corr_mlps = nn.ModuleList([
            Mlp(
                in_features=(1 + transformer_dim) * k_neighbors,
                hidden_features=transformer_dim * 4,
                out_features=transformer_dim,
            )
            for _ in range(corr_levels)
        ])

    def _calc_depth_norm(self, pcd_map: torch.Tensor, clip_min: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = pcd_map[..., 2:3]
        
        mean_z = z.mean(dim=1, keepdim=True)
        std_z = z.std(dim=1, keepdim=True).clamp(min=clip_min)
        
        z_norm = (z - mean_z) / std_z
        return z_norm, mean_z, std_z

    def _get_support_info(
        self,
        level_idx: int,
        pcd_map_0: torch.Tensor,
        feat_map_0: torch.Tensor,
        query_coords_0: torch.Tensor,
        query_uv_0: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = query_coords_0.shape

        map_pcds = rearrange(pcd_map_0, "b c h w -> b (h w) c")
        map_feats = rearrange(feat_map_0, "b c h w -> b (h w) c")

        map_z_norm, mean_z, std_z = self._calc_depth_norm(map_pcds)
        map_pcds_4d = torch.cat([map_pcds, map_z_norm], dim=-1)

        query_z = query_coords_0[..., 2:3]
        query_z_norm = (query_z - mean_z) / std_z
        query_coords_4d = torch.cat(
            [query_coords_0, query_z_norm], dim=-1
        )

        _, knn_idx = _knn(self.k_neighbors, query_coords_0, map_pcds)

        neighbor_coords_4d = torch.gather(
            map_pcds_4d,
            1,
            repeat(knn_idx, "b n k -> b (n k) c", c=4),
        ).view(B, N, self.k_neighbors, 4)

        neighbor_feats = torch.gather(
            map_feats,
            1,
            repeat(knn_idx, "b n k -> b (n k) c", c=self.feat_dim),
        ).view(B, N, self.k_neighbors, self.feat_dim)

        self_coords_4d = query_coords_4d.unsqueeze(2)

        query_feats = sample_features4d(feat_map_0, query_uv_0)
        self_feats = query_feats.unsqueeze(2)

        support_coords_4d = torch.cat(
            [self_coords_4d, neighbor_coords_4d], dim=2
        )
        support_feats = torch.cat(
            [self_feats, neighbor_feats], dim=2
        )

        rel_pos = support_coords_4d - self_coords_4d
        rel_pos_emb = posenc(rel_pos, min_deg=-2, max_deg=14)
        rel_pos_tok = self.posenc_mlps[level_idx](rel_pos_emb)

        support_feats = self.in_norm(support_feats)

        return support_feats + rel_pos_tok

    def build_support_pyramid(
        self,
        pcd_pyramid: List[torch.Tensor],
        feat_pyramid: List[torch.Tensor],
        query_coords_0: torch.Tensor,
        query_uv_0: torch.Tensor,
    ) -> List[torch.Tensor]:
        support_tokens_pyramid: List[torch.Tensor] = []

        H0, W0 = pcd_pyramid[0].shape[-2:]

        for i in range(self.corr_levels):
            pcd_map_0  = pcd_pyramid[i][:, 0]
            feat_map_0 = feat_pyramid[i][:, 0]
            Hi, Wi = pcd_map_0.shape[-2:]

            uv_i = query_uv_0.clone()
            uv_i[..., 0] = uv_i[..., 0] * (Wi / W0)
            uv_i[..., 1] = uv_i[..., 1] * (Hi / H0)

            support_tokens = self._get_support_info(
                level_idx=i,
                pcd_map_0=pcd_map_0,
                feat_map_0=feat_map_0,
                query_coords_0=query_coords_0,
                query_uv_0=uv_i,
            )
            support_tokens_pyramid.append(support_tokens)

        return support_tokens_pyramid

    def forward(
        self,
        pcd_pyramid: List[torch.Tensor],
        feat_pyramid: List[torch.Tensor],
        curr_coords_3d: torch.Tensor,
        support_tokens_pyramid: List[torch.Tensor],
    ) -> torch.Tensor:
        B, S, N, _ = curr_coords_3d.shape
        corr_embs_list: List[torch.Tensor] = []

        for i in range(self.corr_levels):
            support_tokens = support_tokens_pyramid[i]
            support_tokens_bs = repeat(
                support_tokens, "b n k d -> (b s) n k d", s=S
            )

            pcd_bs = rearrange(pcd_pyramid[i], "b s c h w -> (b s) (h w) c")
            feat_bs = rearrange(feat_pyramid[i], "b s c h w -> (b s) (h w) c")
            queries_bs = rearrange(curr_coords_3d, "b s n c -> (b s) n c")

            map_z_norm, mean_z, std_z = self._calc_depth_norm(pcd_bs)
            pcd_bs_4d = torch.cat([pcd_bs, map_z_norm], dim=-1)

            query_z = queries_bs[..., 2:3]
            query_z_norm = (query_z - mean_z) / std_z
            query_coords_4d = torch.cat(
                [queries_bs, query_z_norm], dim=-1
            )

            _, knn_indices = _knn(self.k_neighbors, queries_bs, pcd_bs)

            neighbor_coords_4d = torch.gather(
                pcd_bs_4d,
                1,
                repeat(knn_indices, "bs n k -> bs (n k) c", c=4),
            ).view(B * S, N, self.k_neighbors, 4)

            neighbor_feats = torch.gather(
                feat_bs,
                1,
                repeat(knn_indices, "bs n k -> bs (n k) c", c=self.feat_dim),
            ).view(B * S, N, self.k_neighbors, self.feat_dim)

            neighbor_feats = self.in_norm(neighbor_feats)

            query_pos_4d = query_coords_4d.unsqueeze(2)
            rel_pos = neighbor_coords_4d - query_pos_4d
            rel_pos_emb = posenc(rel_pos, min_deg=-2, max_deg=14)
            rel_pos_tok = self.posenc_mlps[i](rel_pos_emb)

            query_feat = support_tokens_bs[:, :, 0, :]
            query_feat = F.normalize(query_feat, dim=-1)
            neighbor_feat_proj = F.normalize(neighbor_feats, dim=-1)

            q = query_feat.unsqueeze(2)
            k = neighbor_feat_proj
            D_feat = q.shape[-1]
            sim = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(D_feat)

            corr_tok = torch.cat([sim, rel_pos_tok], dim=-1)

            corr_tok_flat = rearrange(corr_tok, "bs n k d -> bs n (k d)")
            corr_feat = self.corr_mlps[i](corr_tok_flat)

            out_feat = rearrange(corr_feat, "(b s) n d -> b s n d", b=B, s=S, n=N)
            corr_embs_list.append(out_feat)

        return torch.cat(corr_embs_list, dim=-1)
