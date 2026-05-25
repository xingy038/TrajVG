# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn

from .modules import AttnBlock, CrossAttnBlock
from ..layers.attention import FlashAttentionRope, FlashCrossAttentionRope


class EfficientUpdateFormer(nn.Module):
    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        num_virtual_tracks=64,
        add_space_attn=True,
        linear_layer_for_vis_conf=False,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        if linear_layer_for_vis_conf:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim - 2, bias=True)
            self.vis_conf_head = torch.nn.Linear(hidden_size, 2, bias=True)
        else:
            self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(
            torch.randn(1, num_virtual_tracks, 1, hidden_size)
        )
        self.add_space_attn = add_space_attn
        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=FlashAttentionRope,
                )
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=FlashAttentionRope,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size,
                        hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=FlashCrossAttentionRope,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size,
                        hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=FlashCrossAttentionRope,
                    )
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            torch.nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            if self.linear_layer_for_vis_conf:
                torch.nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)

        def _trunc_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None, add_space_attn=True):
        tokens = self.input_transform(input_tensor)

        B, _, T, _ = tokens.shape
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape
        j = 0
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)
            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)
            if (
                add_space_attn
                and hasattr(self, "space_virtual_blocks")
                and (i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0)
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )

                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = self.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, mask=mask
                )

                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens, mask=mask
                )

                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(
                    0, 2, 1, 3
                )
                j += 1
        tokens = tokens[:, : N - self.num_virtual_tracks]

        flow = self.flow_head(tokens)
        if self.linear_layer_for_vis_conf:
            vis_conf = self.vis_conf_head(tokens)
            flow = torch.cat([flow, vis_conf], dim=-1)

        return flow
