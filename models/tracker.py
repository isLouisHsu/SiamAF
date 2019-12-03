# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-11-30 17:48:36
@LastEditTime: 2019-12-03 21:03:40
@Update: 
'''
import sys
sys.path.append('..')

import cv2
import numpy as np

import torch

from .network import SiamRPN
from utils.box_utils import crop_square_according_to_bbox, pair_anchors, decode, nms

class SiamRPNTracker():

    def __init__(self, anchors_naive, net, device, 
                template_size=127, search_size=255, feature_size=17, stride=8,
                pad=[lambda w, h: (w + h) / 2],
                cls_thresh=0.9999, nms_thresh=0.6):

        self.anchors_naive = anchors_naive
        self.num_anchor = self.anchors_naive.shape[0]

        self.device = device
        self.net = net; self.net.to(device)

        self.template_size = template_size
        self.search_size   = search_size
        self.feature_size  = feature_size
        self.stride        = stride
        self.pad = pad[0]

        self.cls_thresh = cls_thresh
        self.nms_thresh = nms_thresh

        self.template_setted = False

    def set_template(self, template_image, bbox):
        """
        Params:
            template_image: {ndarray(H, W, C)}
            bbox:           {ndarray(4)}
        """
        template_image = crop_square_according_to_bbox(template_image, bbox, self.template_size, self.pad)
        template_image = self._ndarray2tensor(template_image)
        self.net.template(template_image)
        self.template_setted = True

    def delete_template(self):

        self.net.z_f = None
        self.template_setted = False
    
    def track(self, search_image):
        """
        Params:
            template_image: {ndarray(H, W, C)}
        Returns:
            bbox: {ndarray(N, 4)}
        """
        search_image = self._ndarray2tensor(search_image)
        
        with torch.no_grad(): 
            pred_cls, pred_reg = self.net.track(search_image)
            pred_cls = torch.sigmoid(pred_cls.squeeze()); pred_reg = pred_reg.squeeze()

            # pair anchor for search_image
            anchors_center, _ = pair_anchors(
                self.anchors_naive, pred_cls.size()[1:], self.search_size, self.feature_size, self.stride)
            anchors_center = torch.from_numpy(anchors_center).view(4, -1).t().float().to(self.device)   # (A, 4)

            # filter `class score < thresh`
            pred_cls = pred_cls.view(-1)        # (N)
            pred_reg = pred_reg.view(4, -1).t() # (N, 4)
            mask_cls = torch.nonzero(pred_cls > self.cls_thresh)
            if mask_cls.size(0) == 0:
                return np.full((0, 4), 0, dtype=np.float32)

            mask_cls = mask_cls.squeeze()
            anchors_center = torch.index_select(anchors_center, 0, mask_cls)
            pred_cls       = torch.index_select(pred_cls, 0, mask_cls)
            pred_reg       = torch.index_select(pred_reg, 0, mask_cls)

            # refine
            bbox       = torch.zeros_like(pred_reg)
            bbox[:, 0] = pred_reg[:, 0] * anchors_center[:, 2] + anchors_center[:, 0]
            bbox[:, 1] = pred_reg[:, 1] * anchors_center[:, 3] + anchors_center[:, 1]
            bbox[:, 2] = torch.exp(pred_reg)[:, 2] * anchors_center[:, 2]
            bbox[:, 3] = torch.exp(pred_reg)[:, 3] * anchors_center[:, 3]

            # nms
            keep, count = nms(bbox, pred_cls, self.nms_thresh)
            bbox   = torch.index_select(bbox, 0, keep)
        
        bbox  = bbox.cpu().numpy()
        return bbox

    def _ndarray2tensor(self, ndarray):
        return torch.from_numpy(ndarray.transpose(2, 0, 1) / 255.).unsqueeze(0).float().to(self.device)
