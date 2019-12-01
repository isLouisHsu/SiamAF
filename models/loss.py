# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-11-30 19:46:01
@LastEditTime: 2019-12-01 12:16:58
@Update: 
'''
import sys
sys.path.append('../')

import torch
from torch import nn

from utils.box_utils import naive_anchors, pair_anchors, jaccard, visualize_anchor

class RpnLoss(nn.Module):

    def __init__(self, cls_weight=1., reg_weight=1., 
            stride=8, template_size=127, search_size=255, feature_size=17,
            anchor_ratios=[0.33, 0.5, 1., 2., 3.], anchor_scales=[8], 
            anchor_thr_low=0.3, anchor_thr_high=0.6, n_pos=16, n_neg=48):
        super(RpnLoss, self).__init__()

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        
        self.stride = stride
        self.template_size = template_size
        self.search_size   = search_size
        self.feature_size  = feature_size

        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales
        self.anchor_thr_low  = anchor_thr_low
        self.anchor_thr_high = anchor_thr_high

        self.n_pos = n_pos
        self.n_neg = n_neg

        self._anchors(anchor_ratios, anchor_scales, stride, search_size, self.feature_size)
        # visualize_anchor(search_size, self.anchor_corner[8, 8])

    def _anchors(self, anchor_ratios, anchor_scales, stride, 
                        search_size, feature_size):
        """
        Params:
            feature_size: {int}
            anchor_ratios: {list[int]}
            anchor_scales: {list[int]}
        """
        anchors_naive = naive_anchors(anchor_ratios, anchor_scales, stride)
        center, corner = pair_anchors(
                    anchors_naive, search_size // 2, feature_size, stride)
        self.anchor_center = torch.from_numpy(center).contiguous()
        self.anchor_corner = torch.from_numpy(corner).contiguous()

    def _match(self, gt_bbox):
        """
        Params:
            gt_bbox:  {tensor(N, 4)}
        """
        anchor = self.anchor_corner.view(-1, 4).to(gt_bbox.device)
        iou = jaccard(gt_bbox, anchor).view(gt_bbox.size(0), *self.anchor_corner.size()[:-1])
        print(iou.max())
        # 0: neg, 1: pos, -1: ignore
        negative = torch.zeros_like(iou)
        positive = torch.ones_like(iou)
        mask     = torch.ones_like(iou) * -1
        mask = torch.where(iou > self.anchor_thr_high, positive, mask)
        mask = torch.where(iou < self.anchor_thr_low,  negative, mask)

        print((mask == 0).sum())
        print((mask == 1).sum())
        print((mask == -1).sum())

        pass

    def forward(self, pred_cls, pred_reg, gt_bbox):
        """
        Params:
            pred_cls: {tensor(N, num_anchor,     H, W)}
            pred_reg: {tensor(N, num_anchor * 4, H, W)}
            gt_bbox:  {tensor(N, 4), double} x1, y1, x2, y2
        Returns:
            
        """
        self._match(gt_bbox)
        pass

if __name__ == "__main__":
    
    loss = RpnLoss()

    n, a, h, w = 2, 5, 17, 17
    pred_cls, pred_reg = torch.rand(n, a, h, w), torch.rand(n, a*4, h, w)
    gt_bbox = torch.tensor([
        [90, 90, 110, 110],
        [70, 90, 110, 130],
    ], dtype=torch.double)
    loss(pred_cls, pred_reg, gt_bbox)