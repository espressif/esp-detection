# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv
from .esp_conv import DSConv
from ultralytics.nn.modules.head import Detect

__all__ = "ESPDetect"

class ESPDetect(Detect):
    def __init__(self, nc=1, ch=()):
        """Initializes the ESP detection layer with specified number of classes and channels."""
        # Ultralytics Detect now takes (nc, reg_max, end2end, ch), so pass keywords.
        super().__init__(nc=nc, reg_max=1, ch=ch)
        self.reg_max = 1
        self.no = nc + self.reg_max * 4  # number of outputs per anchor =9

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(DSConv(x, c2, 3), DSConv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        ) #cv2 is CIoU branch
        self.cv3 = nn.ModuleList(
            nn.Sequential(   #cv3 branch is CLS loss branch
            nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
            nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
            nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()  #reg_max=2

    def export_onnx_forward(self, x):
        # self.nl = 3
        box0 = self.cv2[0](x[0])
        score0 = self.cv3[0](x[0])

        box1 = self.cv2[1](x[1])
        score1 = self.cv3[1](x[1])

        box2 = self.cv2[2](x[2])
        score2 = self.cv3[2](x[2])

        return box0, score0, box1, score1, box2, score2
