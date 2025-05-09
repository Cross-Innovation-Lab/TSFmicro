from functools import partial

from torch import nn
from torchvision.transforms import Resize

from classes.model.s_branch import vit
from classes.model.t_branch import RMT_T3


class ts_micro(nn.Module):
    def __init__(self, num_classes=5):
        super(ts_micro, self).__init__()
        ##Position Calibration Module(subbranch)
        self.vit_pos=vit(img_size=14,
        patch_size=1, embed_dim=512, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.)
        self.resize=Resize([14,14])
        ##classes branch consisting of CA blocks
        self.main_branch =RMT_T3(num_class=num_classes)


    def forward(self, x1, x5):
        ##onset:x1 apex:x5
        B = x1.shape[0]
        # Position Calibration Module (subbranch)
        POS =self.vit_pos(self.resize(x1)).transpose(1,2).view(B,512,14,14)
        act = x5 - x1
        #act=self.conv_act(act)
        # classes branch and fusion
        out = self.main_branch(act, POS)

        return out