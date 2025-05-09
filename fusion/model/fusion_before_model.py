# -*- coding: utf-8 -*-            
# @Author : BingYu Nan
# @Location : Wuxi
# @Time : 2025/4/2 14:04
from functools import partial

from torch import nn
from torchvision.transforms import Resize

from PC_module import VisionTransformer_POS
from fusion.model.fusion_RMT import fusion_before_rmt
from fusion.model.fusion_vit_pc import fusion_vit


class TS_micro_before(nn.Module):
    def __init__(self, num_classes=5):
        super(TS_micro_before, self).__init__()

        ##Position Calibration Module(subbranch)
        self.vit_pos=VisionTransformer_POS(img_size=14,
        patch_size=1, embed_dim=512, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.)
        self.resize=Resize([14,14])
        ##classes branch consisting of CA blocks
        self.main_branch =fusion_before_rmt(num_class=num_classes)
        self.transposed_conv1 = nn.ConvTranspose2d(
            in_channels=512,  # 输入通道数
            out_channels=256,  # 输出通道数
            kernel_size=4,  # 卷积核大小
            stride=2,  # 步幅为 2，空间尺寸翻倍
            padding=1  # 填充
        )
        self.transposed_conv2 = nn.ConvTranspose2d(
            in_channels=256,  # 输入通道数
            out_channels=128,  # 输出通道数
            kernel_size=4,  # 卷积核大小
            stride=2,  # 步幅为 2，空间尺寸翻倍
            padding=1  # 填充
        )
        self.transposed_conv3 = nn.ConvTranspose2d(
            in_channels=128,  # 输入通道数
            out_channels=64,  # 输出通道数
            kernel_size=4,  # 卷积核大小
            stride=2,  # 步幅为 2，空间尺寸翻倍
            padding=1  # 填充
        )

    def forward(self, x1, x5):
        ##onset:x1 apex:x5
        B = x1.shape[0]
        # Position Calibration Module (subbranch)
        POS =self.vit_pos(self.resize(x1)).transpose(1,2).view(B,512,14,14)
        POS = self.transposed_conv1(POS)
        POS = self.transposed_conv2(POS)
        POS = self.transposed_conv3(POS)
        act = x5 - x1
        #act=self.conv_act(act)
        # classes branch and fusion
        out = self.main_branch(act, POS)

        return out