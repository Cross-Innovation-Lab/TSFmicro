# -*- coding: utf-8 -*-            
# @Author : BingYu Nan
# @Location : Wuxi
# @Time : 2025/4/21 11:54

from torch import nn

from fusion.model.RMT import RMT


class ConvStem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)  # (b h w c)
        return x


class ts_fusion(nn.Module):
    def __init__(self, num_classes=5):
        super(ts_fusion, self).__init__()

        self.ConvStem_T = ConvStem(in_chans=3, embed_dim=64, norm_layer=nn.LayerNorm)
        self.ConvStem_S = ConvStem(in_chans=3, embed_dim=64, norm_layer=nn.LayerNorm)

        ##classes branch consisting of CA blocks
        self.main_branch =RMT(num_class=num_classes)

    def forward(self, x1, x5):
        ##onset:x1 apex:x5
        act = x5 - x1
        x1 = self.ConvStem_T(x1)
        act = self.ConvStem_S(act)
        act = act + x1
        out = self.main_branch(act)

        return out