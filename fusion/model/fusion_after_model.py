# -*- coding: utf-8 -*-            
# @Author : BingYu Nan
# @Location : Wuxi
# @Time : 2025/4/2 14:04
from functools import partial

import torch
from torch import nn
from torchvision.transforms import Resize

from fusion.model.fusion_RMT import fusion_after_rmt
from fusion.model.fusion_vit_pc import fusion_vit
from model.RMT import MemoryEfficientSwish


class TS_micro_after(nn.Module):
    def __init__(self, num_classes=5):
        super(TS_micro_after, self).__init__()

        ##Position Calibration Module(subbranch)
        self.vit_pos=fusion_vit(img_size=14,
        patch_size=1, embed_dim=512, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.)
        self.resize=Resize([14,14])
        ##classes branch consisting of CA blocks
        self.main_branch = fusion_after_rmt(num_class=num_classes)
        self.num_features = 512
        self.proj = nn.Linear(self.num_features, 1024)
        self.norm = nn.BatchNorm2d(1024)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(1024, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x1, x5):
        ##onset:x1 apex:x5
        B = x1.shape[0]
        act = x5 - x1
        out = self.main_branch(act)
        out_reshaped = out.flatten(1, 2)
        # print(out_reshaped.shape)
        out = self.vit_pos(self.resize(x1), out_reshaped).transpose(1, 2).view(B, 512, 14, 14)
        out = out.permute(0, 2, 3, 1)
        x = self.proj(out)  # (b h w c)
        x = self.norm(x.permute(0, 3, 1, 2)).flatten(2, 3)  # (b c h*w)
        x = self.swish(x)

        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x