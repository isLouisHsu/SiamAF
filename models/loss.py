# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-11-30 19:46:01
@LastEditTime: 2019-11-30 21:04:34
@Update: 
'''
import torch
from torch import nn

from utils.box_utils import jaccard

class RpnLoss(nn.Module):

    def __init__(self, cls_weight=1., reg_weight=1., feature_size=17,
            anchor_ratios=[0.33, 0.5, 1., 2., 3.], anchor_scales=[8], 
            anchor_thr_low=0.3, anchor_thr_high=0.6, n_pos=16, n_neg=48):
        super(RpnLoss, self).__init__()

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def _anchors(self, feature_size, anchor_ratios, anchor_scales):
        """
        Params:
            feature_size: {int}
            anchor_ratios: {list[int]}
            anchor_scales: {list[int]}
        """
        n_ratios = len(anchor_ratios); n_scales = len(anchor_scales)
        n_anchors = n_ratios * n_scales
        
        anchors = torch.empty(feature_size, feature_size, n_anchors, 4)
        for i in range(feature_size):
            for j in range(feature_size):
                for k in range(n_anchors):
                    anchors.data[i, j, k] = 

        pass

    def forward(self, pred_cls, pred_reg, gt_bbox):
        """
        Params:
            pred_cls: {tensor(N, num_anchor,     H, W)}
            pred_reg: {tensor(N, num_anchor * 4, H, W)}
            gt_bbox:  {tensor(N, 4)}
        Returns:
            
        """
