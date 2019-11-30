# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-11-30 17:48:36
@LastEditTime: 2019-11-30 19:11:04
@Update: 
'''
import torch
from torch import nn

from roi_align import RoIAlign

from backbone import AlexNet, resnet50
from head import RpnHead, HeatmapHead

class SiamRPN(nn.Module):
    """
    Attributes:
        self.z_f: {tensor(1, 256, 6, 6)}
    """
    def __init__(self, out_channels=256, num_anchor=5):
        super(SiamRPN, self).__init__()

        self.backbone = AlexNet()
        self.head = RpnHead(self.backbone.feature_size, out_channels, num_anchor)

        self.z_f = None     # for testing

    # -------------- for testing --------------
    def template(self, z):
        """
        Params:
            z: {tensor(1, 3, 127, 127)}
        """
        self.z_f = self.backbone(z)

    def track(self, x):
        """
        Params:
            x: {tensor(1, 3, 255, 255)}
        Returns:
            pred_cls: {tensor(N, n_anchor, 17, 17)}
            pred_reg: {tensor(N, n_anchor, 17, 17)}
        """
        x_f = self.backbone(x)
        pred_cls, pred_reg = self.head(self.z_f, x_f)
        return pred_cls, pred_reg

    # ------------- for training -------------
    def forward(self, z, x):
        """
        Params:
            z: {tensor(N, 3, 127, 127)}
            x: {tensor(N, 3, 255, 255)}
        Returns:
            pred_cls: {tensor(N, n_anchor, 17, 17)}
            pred_reg: {tensor(N, n_anchor, 17, 17)}
        Notes:
            - 255 // 17 = 15
        """
        z_f = self.backbone(z)
        x_f = self.backbone(x)

        pred_cls, pred_reg = self.head(z_f, x_f)
        return pred_cls, pred_reg


class SiamAF(nn.Module):
    """ TODO:
    Attributes:
        self.z_f: {list[tensor(1, 256, h, w)]}
    Notes:
        - See more details about RoiAlign on https://github.com/longcw/RoIAlign.pytorch
    """
    def __init__(self, backbone=resnet50, roi_size=None):
        super(SiamAF, self).__init__()
        
        self.backbone = backbone

        # head
        self.head = nn.ModuleList()
        for feature_size in self.backbone.feature_size:
            self.head.append(HeatmapHead(feature_size))
        
        # roi
        self.roi = None
        if roi_size is not None:
            self.roi = nn.ModuleList()
            for rs in roi_size:
                self.roi.append(RoIAlign(rs, rs))
                
        self.z_f = None     # for testing

    # -------------- for testing --------------
    def template(self, z):
        """
        Params:
            z: {tensor(1, 3, 127, 127)}
        """
        self.z_f = self.backbone(z)
        if self.roi is not None:
            for i, roi in enumerate(self.roi):
                self.z_f[i] = roi(self.zf[i])

    def track(self, x):
        """
        Params:
            x: {tensor(1, 3, 255, 255)}
        Returns:
            pred_cls: {list[tensor(1, 1, h, w)]}
            pred_reg: {list[tensor(1, 4, h, w)]}
        """
        x_f = self.backbone(x)
        
        pred_cls = []; pred_reg = []
        for i, head in enumerate(self.head):
            _cls, _reg = head(self.z_f[i], x_f[i])
            pred_cls += [_cls]; pred_reg += [_cls]
            
        return pred_cls, pred_reg

    # ------------- for training -------------
    def forward(self, z, x):
        """
        Params:
            z: {tensor(N, 3, 127, 127)}
            x: {tensor(N, 3, 255, 255)}
        Returns:
            pred_cls: {tensor(N, n_anchor, h, w)}
            pred_reg: {tensor(N, n_anchor, h, w)}
        """
        z_f = self.backbone(z)
        if self.roi is not None:
            for i, roi in enumerate(self.roi):
                self.z_f[i] = roi(self.zf[i])

        x_f = self.backbone(x)

        pred_cls, pred_reg = self.head(z_f, x_f)
        return pred_cls, pred_reg