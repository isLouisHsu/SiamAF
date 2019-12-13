# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-11-30 17:48:36
@LastEditTime: 2019-12-13 20:49:49
@Update: 
'''
import sys
sys.path.append('..')

import cv2
import numpy as np
from matplotlib import pyplot as plt
from easydict import EasyDict as edict

import torch

from .network import SiamRPN
from utils.box_utils import (
    crop_square_according_to_bbox, 
    get_hamming_window,
    center2corner, corner2center, 
    nms, 
    visualize_anchor, show_bbox)

class SiamRPNTracker():

    def __init__(self, anchors_center, net, device, 
                template_size=127, search_size=255, feature_size=17, center_size=7,
                stride=8, pad=[lambda w, h: (w + h) / 2],
                penalty_k=1.0, window_factor=0.42, momentum=0.295):

        self.device = device
        self.net = net; self.net.to(device)

        self.template_size = template_size
        self.search_size   = search_size
        self.feature_size  = feature_size
        self.center_size   = center_size
        self.stride        = stride
        self.pad = pad[0]

        self.anchors_center = anchors_center
        self.num_anchor = self.anchors_center.shape[1]

        self.penalty_k  = penalty_k
        self.window_factor = window_factor
        self.momentum = momentum

        self.window = get_hamming_window(feature_size, self.num_anchor)
        self.state = edict()

    def set_template(self, template_image, bbox):
        """
        Params:
            template_image: {ndarray(H, W, C)}
            bbox:           {ndarray(4)}        x1, y1, x2, y2
        """
        self._update_state(bbox)
        
        template_image = crop_square_according_to_bbox(
            template_image, bbox, self.template_size, self.pad)
        # show_bbox(template_image, bbox, winname='template_image')
        
        template_tensor = self._ndarray2tensor(template_image)
        self.net.template(template_tensor)

    def delete_template(self):

        self.net.z_f = None
    
    def template_is_setted(self):

        return self.net.z_f is not None

    def track(self, search_image):
        """
        Params:
            template_image: {ndarray(H, W, C)}
        Returns:
            bbox: {ndarray(N, 4)}
        """
        search_crop, (scale, shift) = crop_square_according_to_bbox(
                search_image, self.state.corner, self.search_size, lambda w, h: self.pad(w, h) * 2, return_param=True)

        # show_bbox(search_image, self.state.corner, winname='search_image')
        # show_bbox(search_crop, (self.state.corner - np.concatenate([shift, shift])) * scale, winname='search_crop')
        
        with torch.no_grad(): 
            pred_cls, pred_reg = self.net.track(self._ndarray2tensor(search_crop))   
            score = torch.softmax(pred_cls.squeeze(), dim=0).cpu().numpy()[1]  # (   5, 17, 17)
            pred_reg = pred_reg.squeeze().cpu().numpy()                 # (4, 5, 17, 17)

        # refine
        bbox_center    = np.zeros_like(pred_reg)                        # (4, 5, 17, 17)
        bbox_center[0] = self.anchors_center[2] * pred_reg[0] + self.anchors_center[0]  # xc
        bbox_center[1] = self.anchors_center[3] * pred_reg[1] + self.anchors_center[1]  # yc
        bbox_center[2] =              np.exp(pred_reg[2])* self.anchors_center[2]       #  w
        bbox_center[3] =              np.exp(pred_reg[3])* self.anchors_center[3]       #  h

        # penalty
        r = self._r(bbox_center[2], bbox_center[3])                     # (   5, 17, 17)
        s = self._s(bbox_center[2], bbox_center[3])                     # (   5, 17, 17)
        pr = np.maximum(r / self.state.r, self.state.r / r)             # (   5, 17, 17)
        ps = np.maximum(s / self.state.s, self.state.s / s)             # (   5, 17, 17)
        penalty = np.exp(- (pr * ps - 1) * self.penalty_k)              # (   5, 17, 17)
        pscore = score * penalty                                        # (   5, 17, 17)

        # cosine window
        pscore = self.window * self.window_factor + \
                            pscore * (1 - self.window_factor)           # (   5, 17, 17)

        # fig = plt.figure()
        # for i_anchor in range(self.num_anchor):
        #     fig.add_subplot(2, 5, i_anchor + 1)
        #     plt.imshow(score[i_anchor])
        #     fig.add_subplot(2, 5, i_anchor + 6)
        #     plt.imshow(pscore[i_anchor])
        # plt.show()

        # pick the highest score
        a, r, c = np.unravel_index(pscore.argmax(), pscore.shape)
        res_center = bbox_center[:, a, r, c]; score = score[a, r, c]
        # show_bbox(search_crop, np.array(center2corner(res_center)), winname='search_crop_output')

        # ------------------------------------------------------
        # get back!
        res_center /= scale         # scale, ( w,  h)
        res_center[:2] += shift     # shift, (xc, yc)
        
        # momentum
        momentum = pscore[a, r, c] * self.momentum
        res_center[2:] = res_center[2:] * momentum + self.state.center[2:] * (1 - momentum)     #  w,  h
        
        res_corner = np.array(center2corner(res_center))
        # show_bbox(search_image, res_corner, score, self.state.center[:2], winname='search_image_output')

        self._update_state(res_corner)

        return res_corner, score

    def _ndarray2tensor(self, ndarray):
        return torch.from_numpy(ndarray.transpose(2, 0, 1) / 255.).unsqueeze(0).float().to(self.device)

    def _update_state(self, bbox):
        """ Update state
        Params:
            bbox: {ndarray(4)} x1, y1, x2, y2
        """
        self.state.corner = bbox                           # x1, y1, x2, y2
        self.state.center = np.array(corner2center(bbox))  # xc, yc,  w,  h

        w, h = self.state.center[2:]
        self.state.r = self._r(w, h)
        self.state.s = self._s(w, h)

    def _r(self, w, h):
        return w / h
    
    def _s(self, w, h):
        p = self.pad(w, h)
        return np.sqrt(w + p) * np.sqrt(h + p)