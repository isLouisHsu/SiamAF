# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-11-30 19:46:01
@LastEditTime: 2019-12-03 20:41:14
@Update: 
'''
import sys
sys.path.append('../')

import numpy as np
import torch
from torch import nn

from utils.box_utils import jaccard, corner2center, encode

class RpnLoss(nn.Module):

    template_size=127
    search_size=255 
    feature_size=17

    def __init__(self, anchors, cls_weight=1., reg_weight=1., pos_thr=0.6,
            anchor_thr_low=0.3, anchor_thr_high=0.6, n_pos=16, n_neg=48):
        super(RpnLoss, self).__init__()

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        
        self.pos_thr = pos_thr
        self.anchor_thr_low  = anchor_thr_low
        self.anchor_thr_high = anchor_thr_high

        self.n_pos = n_pos
        self.n_neg = n_neg

        self.anchor_center, self.anchor_corner = list(
                map(lambda x: torch.from_numpy(x.reshape(4, -1).T).float().contiguous(), anchors))
        
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.smoothl1 = nn.SmoothL1Loss()

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

        # print(iou.max())
        # print((matched == 1).sum())
        # print((matched == 0).sum())
        # print((matched == -1).sum())

        return matched

    def forward(self, pred_cls, pred_reg, gt_bbox):
        """
        Params:
            pred_cls: {tensor(N,    num_anchor, H, W)} [0, 1]
            pred_reg: {tensor(N, 4, num_anchor, H, W)}
            gt_bbox:  {tensor(N, 4), double} x1, y1, x2, y2
        Returns:
            
        """
        loss_cls = 0; loss_reg = 0; acc_cls = 0

        matched = self._match(gt_bbox)     # (N, num_anchor, H， W)
        for i, (cls, reg, gt, mask) in enumerate(zip(pred_cls, pred_reg, gt_bbox, matched)):
            cls = cls.view(-1); reg = reg.view(4, -1).t()
            
            # classification
            cls_pred_pos = torch.masked_select(cls, mask == 1)
            cls_pred_neg = torch.masked_select(cls, mask == 0)
            n_pos, n_neg = cls_pred_pos.size(0), cls_pred_neg.size(0)
            if n_pos == 0:
                index = np.arange(n_neg); np.random.shuffle(index)
                cls_pred_neg = cls_pred_neg[index][:self.n_neg]
                cls_gt_neg   = torch.zeros_like(cls_pred_neg)

                loss_cls_i = self.bce(cls_pred_neg, cls_gt_neg)
            else:
                index = np.arange(n_pos); np.random.shuffle(index)
                cls_pred_pos = cls_pred_pos[index][:self.n_pos]
                cls_gt_pos    = torch.ones_like(cls_pred_pos)
                
                index = np.arange(n_neg); np.random.shuffle(index)
                cls_pred_neg = cls_pred_neg[index][:int(n_pos * self.n_neg / self.n_pos)]
                cls_gt_neg    = torch.zeros_like(cls_pred_neg)
                
                loss_cls_i = self.bce(
                        torch.cat([cls_pred_pos, cls_pred_neg]), 
                        torch.cat([  cls_gt_pos,   cls_gt_neg]))
            loss_cls += loss_cls_i

            # accuracy
            cls_pred_pos = torch.masked_select(cls, mask == 1) >= self.pos_thr
            cls_pred_neg = torch.masked_select(cls, mask != 1) <  self.pos_thr
            acc_cls += (cls_pred_pos.sum() + cls_pred_neg.sum()).double() / (cls_pred_pos.numel() + cls_pred_neg.numel())

            # regression
            index = torch.nonzero(mask == 1).squeeze()
            reg_pred = torch.index_select(   reg, 0, index)
            anchor   = torch.index_select(self.anchor_center.to(gt_bbox.device), 0, index) # xc, yc, w, h
            if anchor.size(0) == 0:
                loss_reg_i = torch.tensor(0.)
            else:
                gtc = torch.tensor(corner2center(gt))
                reg_gt   = encode(gtc, anchor)
                loss_reg_i = self.mse(reg_pred, reg_gt)
            loss_reg += loss_reg_i

        loss_cls, loss_reg, acc_cls = list(map(lambda x: x / gt_bbox.size(0), [loss_cls, loss_reg, acc_cls]))
        loss_total = self.cls_weight * loss_cls + self.reg_weight * loss_reg
        return loss_total, loss_cls, loss_reg, acc_cls

if __name__ == "__main__":
    
    loss = RpnLoss(get_anchor_train())

    n, a, h, w = 2, 5, 17, 17
    pred_cls, pred_reg = torch.sigmoid(torch.rand(n, a, h, w)), torch.rand(n, 4, a, h, w)
    gt_bbox = torch.tensor([
        [70., 111., 179., 143.],
        [75., 111., 181., 143.],
    ], dtype=torch.double)
    loss(pred_cls, pred_reg, gt_bbox)