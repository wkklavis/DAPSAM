import math

import numpy as np
import torch
from numpy import random
from torch import nn

import torch.nn.functional as F
from torch.nn.init import _no_grad_trunc_normal_

from prostate.segment_anything.modeling.adapter.Filter import ChannelFilter

device = "cuda" if torch.cuda.is_available() else "cpu"

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class PromptGenerator(nn.Module):
    def __init__(self, scale_factor, embed_dim,  depth):
        """
        Args:
        """
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.depth = depth
        self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim)

        for i in range(self.depth):
            lightweight_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor),
                nn.GELU(),
                nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim)
            )
            # channel_filter = ChannelFilter()
            setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)
            # setattr(self, 'channel_filter_{}'.format(str(i)), channel_filter)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _no_grad_trunc_normal_(m.weight, mean=0., std=.02, a=-2., b=2.)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_embeddings(self, x):
        N, C, H, W = x.permute(0, 3, 1, 2).shape
        x = x.reshape(N, C, H*W).permute(0, 2, 1)
        embedding_feature = self.embedding_generator(x)
        embedding_feature = embedding_feature.permute(0, 2, 1).reshape(N, -1, H , W)
        return embedding_feature

    def get_prompt(self, i, feature):#, embedding_feature
        # N, C, H, W = embedding_feature.shape
        feature = feature.permute(0, 3, 1, 2)
        N, C, H, W = feature.shape
        # channel_filter = getattr(self, 'channel_filter_{}'.format(str(i)))
        # filtered_feature = spatial_filter(0.5*feature+embedding_feature)

        # filtered_feature = filtered_feature.reshape(N, C, H * W).permute(0, 2, 1)
        feature = feature.reshape(N, C, H * W).permute(0, 2, 1)

        lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))

        prompt = lightweight_mlp(feature)#+iip 0.5*feature+handcrafted_feature ++embedding_contour 0.5*feature+embedding_feature
        return prompt

    
