# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-11-30 19:46:01
@LastEditTime: 2019-12-16 14:03:57
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
    stride = 8

    def __init__(self, anchors, cls_weight=1., reg_weight=1., ct_weight=.1, pos_thr=0.9,
            anchor_thr_low=0.3, anchor_thr_high=0.6, n_pos=16, n_neg=48, r_pos=2):
        super(RpnLoss, self).__init__()

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.ct_weight  = ct_weight
        
        self.pos_thr = pos_thr
        self.anchor_thr_low  = anchor_thr_low
        self.anchor_thr_high = anchor_thr_high

        self.n_pos = n_pos
        self.n_neg = n_neg
        self.r_pos  = r_pos

        self.anchor_center, self.anchor_corner = list(
                map(lambda x: torch.from_numpy(x.reshape(4, -1).T).float().contiguous(), anchors))
        
        self.crossent = nn.CrossEntropyLoss()
        self.l1  = nn.L1Loss()
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

        # DEBUG:
        # from matplotlib import pyplot as plt
        # iou_ = iou.cpu().detach().numpy().reshape(iou.shape[0], 5, 17, 17)
        # fig = plt.figure()
        # for i in range(5):
        #     for j in range(iou.shape[0]):
        #         fig.add_subplot(iou.shape[0], 5, i*iou.shape[0] + j + 1)
        #         plt.imshow(iou_[j, i, :, :])
        # plt.show()

        # 0: neg; 1: pos; 2: part; -1: ignore
        negative = torch.zeros_like(iou)
        part     = torch.ones_like (iou) * 2
        positive = torch.ones_like (iou)
        matched     = torch.ones_like (iou) * -1
        
        matched = torch.where(iou == 0., negative, matched)
        matched = torch.where((iou > 0.) & (iou < self.anchor_thr_low),  part, matched)
        matched = torch.where(iou > self.anchor_thr_high, positive, matched)

        # print(iou.max())
        # print("ignore:   ", (matched == -1).sum())
        # print("positive: ", (matched == 1).sum())
        # print("part:     ", (matched == 2).sum())
        # print("negative: ", (matched == 0).sum())

        return matched

    def forward(self, pred_cls, pred_reg, gt_bbox):
        """
        Params:
            pred_cls: {tensor(N, 2, num_anchor, H, W)}
            pred_reg: {tensor(N, 4, num_anchor, H, W)}
            gt_bbox:  {tensor(N, 4), double} x1, y1, x2, y2
        Returns:
            loss_total, loss_cls, loss_reg, acc_cls: {tensor(1)}
        """
        loss_cls = 0; loss_reg = 0; acc_cls = 0

        matched = self._match(gt_bbox)     # (N, A)
        for i, (cls, reg, gt, mask) in enumerate(zip(pred_cls, pred_reg, gt_bbox, matched)):

            # center
            x1, y1, x2, y2 = gt; cx, cy = (x1 + x2) / 2., (y1 + y2) / 2.
            cx, cy = list(map(lambda x: (x - self.template_size // 2) // self.stride, [cx, cy]))
            x, y = np.meshgrid(np.arange(self.feature_size) - cx.cpu().numpy(),
                        np.arange(self.feature_size) - cy.cpu().numpy())
            dist_to_center = np.abs(x) + np.abs(y)
            pred_center = cls.sum(dim=1).view(2, -1).t()
            gt_center = torch.from_numpy(
                np.where(dist_to_center <= self.r_pos, 
                        np.ones_like(y), np.zeros_like(y))).to(cls.device).view(-1).long()
            loss_ct_i = self.crossent(pred_center, gt_center)
            
            # classification
            cls = cls.view(2, -1).t()
            cls_pred_neg  = torch.index_select(cls, 0, torch.nonzero(mask == 0).view(-1))
            cls_pred_pos  = torch.index_select(cls, 0, torch.nonzero(mask == 1).view(-1))
            cls_pred_part = torch.index_select(cls, 0, torch.nonzero(mask == 2).view(-1))
            n_pos, n_neg, n_part  = cls_pred_pos.size(0), cls_pred_neg.size(0), cls_pred_part.size(0)
            if n_pos == 0:
                index = np.arange(n_neg); np.random.shuffle(index)
                cls_pred_neg = cls_pred_neg[index][:self.n_pos + self.n_neg]     # (n_pos, 2)
                cls_gt_neg   = torch.zeros(self.n_pos + self.n_neg, dtype=torch.long).to(cls_pred_neg.device) # (n_pos, )

                loss_cls_i = self.crossent(cls_pred_neg, cls_gt_neg)
            else:
                # pos
                index = np.arange(n_pos); np.random.shuffle(index)
                cls_pred_pos = cls_pred_pos[index][:self.n_pos]
                cls_gt_pos   = torch.ones(cls_pred_pos.size(0), dtype=torch.long).to(cls_pred_pos.device)
                
                # neg
                n_neg_total = self.n_pos + self.n_neg - cls_pred_pos.size(0)
                ## part
                index = np.arange(n_part); np.random.shuffle(index)
                cls_pred_part = cls_pred_part[index][:int(n_neg_total * (n_part / (n_neg + n_part)))]
                cls_gt_part   = torch.zeros(cls_pred_part.size(0), dtype=torch.long).to(cls_pred_part.device)

                ## neg
                index = np.arange(n_neg); np.random.shuffle(index)
                cls_pred_neg  = cls_pred_neg[index][:int(n_neg_total - cls_pred_part.size(0))]
                cls_gt_neg    = torch.zeros(cls_pred_neg.size(0), dtype=torch.long).to(cls_pred_neg.device)
                
                loss_cls_i = self.crossent(cls_pred_pos, cls_gt_pos) * 0.5 + \
                             self.crossent(
                                    torch.cat([cls_pred_part, cls_pred_neg]), 
                                    torch.cat([cls_gt_part,   cls_gt_neg  ])) * 0.5

            loss_cls += (loss_cls_i + self.ct_weight * loss_ct_i)

            # accuracy
            cls = torch.softmax(cls, dim=1)
            cls_pred_pos = torch.masked_select(cls[:, 1], mask == 1) >= self.pos_thr
            cls_pred_neg = torch.masked_select(cls[:, 1], mask != 1) <  self.pos_thr
            acc_cls += (cls_pred_pos.sum() + cls_pred_neg.sum()).double() / (cls_pred_pos.numel() + cls_pred_neg.numel())

            # regression
            reg_gt = encode(
                torch.tensor(corner2center(gt)), self.anchor_center.to(gt_bbox.device)).\
                    view(*reg.size()[1:], -1).permute(3, 0, 1, 2)   # (4, 5, 17, 17)
            diff = (reg - reg_gt).abs().sum(dim=0).view(-1)         # 5 * 17 * 17
            mask_p = (mask == 1).float()
            mask_p = mask_p / (mask_p.sum() + 1e-7)
            loss_reg += (diff * mask_p).sum()

        loss_cls, loss_reg, acc_cls = list(map(lambda x: x / gt_bbox.size(0), [loss_cls, loss_reg, acc_cls]))
        loss_total = self.cls_weight * loss_cls + self.reg_weight * loss_reg
        
        return loss_total, loss_cls, loss_reg, acc_cls


class HeatmapLoss(nn.Module):

    template_size=127
    search_size=255 

    def __init__(self, cls_weight=1., reg_weight=1., stride=[4, 8], sigma=.2):
        super(HeatmapLoss, self).__init__()

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.stride = stride
        self.sigma = sigma
        
        self.mse = nn.MSELoss()
    
    def _heatmap(self, size, mu, sigma):
        """
        Params:
            size:   {int}
            mu, sigma: {ndarray(2)}
        Returns:
            z: {ndarray(size, size)}
        """
        gaussian = lambda x, u, s: np.exp(- ((x - u) / s) ** 2 / 2) / (np.sqrt(2 * np.pi) * s)
        x = np.arange(size); z1 = gaussian(x, mu[1], sigma[1]); z2 = gaussian(x, mu[0], sigma[0])
        z = np.outer(z1, z2)

        # print(mu, sigma)
        # print(sigma / self.sigma)
        # print(z.sum())
        # from matplotlib import pyplot as plt
        # # plt.figure()
        # # plt.plot(z1)
        # # plt.figure()
        # # plt.plot(z2)
        # plt.figure()
        # plt.imshow(z)
        # plt.show()
        
        return z

    def forward(self, pred_cls, pred_reg, gt_bbox):
        """
        Params:
            pred_cls: {list(tensor(N, 1, H, W))} [0, 1]
            pred_reg: {list(tensor(N, 4, H, W))}
            gt_bbox:  {tensor(N, 4), double} x1, y1, x2, y2
        Returns:
            loss_total, loss_cls, loss_reg, acc_cls: {tensor(1)}
        """
        loss_cls = 0; loss_reg = 0

        for j, (cls_pd, reg_pd, s) in enumerate(zip(pred_cls, pred_reg, self.stride)):
            
            for i, (gt_i, cls_i, reg_i) in enumerate(zip(gt_bbox, cls_pd, reg_pd)):

                x1, y1, x2, y2 = gt_i.cpu().numpy() - (self.search_size - self.template_size) // 2
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2); w, h = x2 - x1, y2 - y1
                
                mu = np.array([cx / s, cy / s]); sigma = self.sigma * np.array([w / s, h / s])
                cls_gt_i = self._heatmap(cls_i.size(-1), mu, sigma)
                cls_gt_i = torch.from_numpy(cls_gt_i).unsqueeze(0).to(cls_i.device).float()
                loss_cls_i = self.mse(cls_i / cls_i.sum(), cls_gt_i)

                

                pass

            # N, _, feature_size, _ = cls.size()
            # center_size = (feature_size - 1) * s
            
            pass
    
