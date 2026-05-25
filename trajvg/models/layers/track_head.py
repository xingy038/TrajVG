# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from ..track_modules.base_track_predictor import BaseTrackerPredictor


class TrackHead(nn.Module):
    def __init__(
        self,
        features=128,
        iters=4,
        stride=2,
        corr_levels=4,
        hidden_size=384,
    ):
        super().__init__()
        self.tracker = BaseTrackerPredictor(
            latent_dim=features,
            stride=stride,
            corr_levels=corr_levels,
            hidden_size=hidden_size,
        )

        self.iters = iters
    
    def forward(self, query_points2d=None, query_points3d=None, fmaps=None, pointmaps=None, iters=None):
        if iters is None:
            iters = self.iters
        traj3d_preds, vis_scores = self.tracker(query_points2d=query_points2d, query_points3d=query_points3d, fmaps=fmaps, pointmaps=pointmaps, iters=iters)
        return traj3d_preds, vis_scores
