# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-11-30 19:46:01
@LastEditTime: 2019-12-01 14:15:53
@Update: 
'''
import sys
sys.path.append('../')

import numpy as np
import torch
from torch import nn

from utils.box_utils import naive_anchors, pair_anchors, jaccard, visualize_anchor

class RpnLoss(nn.Module):

    def __init__(self, cls_weight=1., reg_weight=1., 
            stride=8, template_size=127, search_size=255, feature_size=17,
            anchor_ratios=[0.33, 0.5, 1., 2., 3.], anchor_scales=[8], 
            anchor_thr_low=0.3, anchor_thr_high=0.6, n_pos=16, n_neg_times=3.,
            vis_anchor=False):
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
        self.n_neg_times = n_neg_times

        self._anchors(anchor_ratios, anchor_scales, stride, search_size, self.feature_size)
        
        if vis_anchor:
            visualize_anchor(search_size, self.anchor_corner[10: 15])

        self.bce = nn.BCELoss()
        self.smoothl1 = nn.SmoothL1Loss()

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
        self.anchor_center = torch.from_numpy(center.reshape(4, -1).T).contiguous()  # (feature_size * feature_size * num_anchor, 4) xc, yc,  w,  h
        self.anchor_corner = torch.from_numpy(corner.reshape(4, -1).T).contiguous()  # (feature_size * feature_size * num_anchor, 4) x1, y1, x2, y2

    def _match(self, gt_bbox):
        """
        Params:
            gt_bbox:  {tensor(N, 4)}
        Returns:
            matched: {tensor(N, A)}
        """
        anchor = self.anchor_corner.to(gt_bbox.device)
        iou = jaccard(gt_bbox, anchor)          # (N, A)

        # 0: neg, 1: pos, -1: ignore
        negative = torch.zeros_like(iou)
        positive = torch.ones_like (iou)
        matched     = torch.ones_like (iou) * -1
        matched = torch.where(iou > self.anchor_thr_high, positive, matched)
        matched = torch.where(iou < self.anchor_thr_low,  negative, matched)

        print((matched == 0).sum())
        print((matched == 1).sum())
        print((matched == -1).sum())

        return matched

    def _encode(self, gt_bbox, anchor_center):
        """
        Params:
            gt_bbox:        {tensor(4), double} x1, y1, x2, y2
            anchor_center:  {tensor(A, 4)}      xc, yc,  w,  h
        Notes:
            x1e = (x1 - xc) / w
            y1e = (y1 - yc) / h
            x2e = (x1 - xc) / w
            y2e = (y2 - yc) / h
        """
        encoded = torch.zeros_like(anchor_center, dtype=torch.float)
        encoded[:, 0] = (gt_bbox[0] - anchor_center[:, 0]) / anchor_center[:, 2]
        encoded[:, 1] = (gt_bbox[1] - anchor_center[:, 1]) / anchor_center[:, 3]
        encoded[:, 2] = (gt_bbox[2] - anchor_center[:, 0]) / anchor_center[:, 2]
        encoded[:, 3] = (gt_bbox[3] - anchor_center[:, 1]) / anchor_center[:, 3]
        return encoded


    def forward(self, pred_cls, pred_reg, gt_bbox):
        """
        Params:
            pred_cls: {tensor(N,    num_anchor, H, W)} [0, 1]
            pred_reg: {tensor(N, 4, num_anchor, H, W)}
            gt_bbox:  {tensor(N, 4), double} x1, y1, x2, y2
        Returns:
            
        """
        loss_cls = 0; loss_reg = 0

        matched = self._match(gt_bbox)     # (N, num_anchor, H， W)
        for i, (cls, reg, gt, mask) in enumerate(zip(pred_cls, pred_reg, gt_bbox, matched)):
            cls = cls.view(-1); reg = reg.view(4, -1).t()
            
            # classification
            cls_pred_pos = torch.masked_select(cls, mask == 1)
            cls_pred_neg = torch.masked_select(cls, mask == 0)
            n_pos, n_neg = cls_pred_pos.size(0), cls_pred_neg.size(0)

            if n_pos == 0:
                loss_cls_i = 0
            else:
                index = np.arange(n_pos); np.random.shuffle(index)
                cls_pred_pos = cls_pred_pos[index][:self.n_pos]
                cls_gt_pos    = torch.ones_like(cls_pred_pos)
                
                index = np.arange(n_neg); np.random.shuffle(index)
                cls_pred_neg = cls_pred_neg[index][:int(n_pos * self.n_neg_times)]
                cls_gt_neg    = torch.zeros_like(cls_pred_neg)
                
                loss_cls_i = self.bce(
                        torch.cat([cls_pred_pos, cls_pred_neg]), 
                        torch.cat([  cls_gt_pos,   cls_gt_neg]))
            loss_cls += loss_cls_i

            # regression
            index = torch.nonzero(mask == 1).squeeze()
            if index.size(0) == 0:
                loss_reg_i = 0
            else:
                reg_pred = torch.index_select(   reg, 0, index)
                anchor   = torch.index_select(self.anchor_center.to(gt_bbox.device), 0, index) # xc, yc, w, h
                reg_gt   = self._encode(gt, anchor)
                loss_reg_i = self.smoothl1(reg_pred, reg_gt)
            loss_reg += loss_reg_i
        
        loss_total = self.cls_weight * loss_cls + self.reg_weight * loss_reg
        return loss_total, loss_cls, loss_reg

# if __name__ == "__main__":
    
#     loss = RpnLoss()

#     n, a, h, w = 2, 5, 17, 17
#     pred_cls, pred_reg = torch.sigmoid(torch.rand(n, a, h, w)), torch.rand(n, 4, a, h, w)
#     gt_bbox = torch.tensor([
#         [70., 111., 179., 143.],
#         [75., 111., 181., 143.],
#     ], dtype=torch.double)
#     loss(pred_cls, pred_reg, gt_bbox)