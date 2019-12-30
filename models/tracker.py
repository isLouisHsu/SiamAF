# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-11-30 17:48:36
@LastEditTime : 2019-12-30 13:24:03
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
    get_hamming_window, naive_anchors, pair_anchors,
    center2corner, corner2center, 
    nms, 
    visualize_anchor, show_bbox)

class SiamRPNTracker():

    def __init__(self, net, device, anchor,
                template_size=127, search_size=255, feature_size=17, center_size=7,
                pad=[lambda w, h: (w + h) / 2], padval=None,
                penalty_k=1.0, window_factor=0.42, momentum=0.295):

        self.device = device
        self.net = net; self.net.to(device)

        self.template_size = template_size
        self.search_size   = search_size
        self.feature_size  = feature_size
        self.center_size   = center_size
        self.pad = pad[0]
        self.padval = padval

        self.anchor = anchor
        self.num_anchor = len(anchor.anchor_ratios) * len(anchor.anchor_scales)
        self.anchor_naive = naive_anchors(anchor.anchor_ratios, anchor.anchor_scales, anchor.stride)
        self.anchors_center, _ = pair_anchors(self.anchor_naive, (feature_size, feature_size), search_size, feature_size, anchor.stride)

        self.penalty_k  = penalty_k
        self.window_factor = window_factor
        self.momentum = momentum

        self.window = get_hamming_window(feature_size, self.num_anchor)
        self.state = edict()

        self.template_scale = None

    def set_template(self, template_image, bbox):
        """
        Params:
            template_image: {ndarray(H, W, C)}
            bbox:           {ndarray(4)}        x1, y1, x2, y2
        """
        self._update_state(bbox)
        
        template_image, (self.template_scale, _) = crop_square_according_to_bbox(
            template_image, bbox, self.template_size, self.pad, self.padval, return_param=True)
        # show_bbox(template_image, bbox, winname='template_image')
        
        template_tensor = self._ndarray2tensor(template_image)
        self.net.template(template_tensor)

    def delete_template(self):

        self.net.z_f = None
    
    def template_is_setted(self):

        return self.net.z_f is not None

    def track(self, search_image, mode='crop'):
        """
        Params:
            template_image: {ndarray(H, W, C)}
        Returns:
            res_corner: {ndarray(N, 4)}
            score: {ndarray(N)}
        """
        if mode == 'crop':
            return self._track_crop_image(search_image)
        elif mode == 'whole':
            return self._track_whole_image(search_image)

    def _track_crop_image(self, search_image):
        """
        Params:
            template_image: {ndarray(H, W, C)}
        Returns:
            res_corner: {ndarray(N, 4)}
            score: {ndarray(N)}
        """
        search_crop, (scale, shift) = crop_square_according_to_bbox(
                search_image, self.state.corner, self.search_size, lambda w, h: self.pad(w, h) * 2, self.padval, return_param=True)

        # show_bbox(search_image, self.state.corner, winname='search_image')
        show_bbox(search_crop, (self.state.corner - np.concatenate([shift, shift])) * scale, winname='search_crop', waitkey=5)
        
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
        _pscore = (pscore * 255).astype(np.uint8)
        for _ch, _p in enumerate(_pscore):
            cv2.imshow("%d" % _ch, _p)
        cv2.waitKey(5)

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
        # res_center[:2] = res_center[:2] * (1 - momentum) + self.state.center[:2] * momentum
        res_center[2:] = res_center[2:] * momentum + self.state.center[2:] * (1 - momentum)     #  w,  h
        
        res_corner = np.array(center2corner(res_center))
        # show_bbox(search_image, res_corner, score, self.state.center[:2], winname='search_image_output')

        self._update_state(res_corner)

        return res_corner, score

    def _track_whole_image(self, search_image):
        """
        Params:
            template_image: {ndarray(H, W, C)}
        Returns:
            res_corner: {ndarray(N, 4)}
            score: {ndarray(N)}
        """
        h, w, _ = search_image.shape
        h, w = list(map(lambda x: int(x * self.template_scale), [h, w]))
        search_resize = cv2.resize(search_image, (w, h))

        # show_bbox(search_resize, None, winname='search_resize')

        with torch.no_grad(): 
            pred_cls, pred_reg = self.net.track(self._ndarray2tensor(search_resize))   
            score = torch.softmax(pred_cls.squeeze(), dim=0).cpu().numpy()[1]  # (   5, h, w)
            pred_reg = pred_reg.squeeze().cpu().numpy()                 # (4, 5, h, w)

        anchors_center, _ = pair_anchors(self.anchor_naive, 
                score.shape[1:], self.search_size, self.feature_size, self.anchor.stride)
                
        # refine
        bbox_center    = np.zeros_like(pred_reg)                        # (4, 5, h, w)
        bbox_center[0] = anchors_center[2] * pred_reg[0] + anchors_center[0]  # xc
        bbox_center[1] = anchors_center[3] * pred_reg[1] + anchors_center[1]  # yc
        bbox_center[2] =              np.exp(pred_reg[2])* anchors_center[2]       #  w
        bbox_center[3] =              np.exp(pred_reg[3])* anchors_center[3]       #  h

        # penalty
        r = self._r(bbox_center[2], bbox_center[3])                     # (   5, h, w)
        s = self._s(bbox_center[2], bbox_center[3])                     # (   5, h, w)
        pr = np.maximum(r / self.state.r, self.state.r / r)             # (   5, h, w)
        ps = np.maximum(s / self.state.s, self.state.s / s)             # (   5, h, w)
        penalty = np.exp(- (pr * ps - 1) * self.penalty_k)              # (   5, h, w)
        pscore = score * penalty                                        # (   5, h, w)

        # pick the highest score
        a, r, c = np.unravel_index(pscore.argmax(), pscore.shape)
        res_center = bbox_center[:, a, r, c]; score = score[a, r, c]
        res_corner = np.array(center2corner(res_center / self.template_scale))

        # momentum
        momentum = pscore[a, r, c] * self.momentum
        res_center[2:] = res_center[2:] * momentum + self.state.center[2:] * (1 - momentum)     #  w,  h
        
        res_corner = np.array(center2corner(res_center))
        # show_bbox(search_image, res_corner, score, anchors_center[:2, a, r, c], winname='search_image_output(with anchor)')

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


class SiamAFTracker():

    def __init__(self, net, device,
                template_size=127, search_size=255, stride=[4, 8], 
                pad=[lambda w, h: (w + h) / 2], padval=None,
                penalty_k=1.0, window_factor=0., momentum=0.295):

        self.device = device
        self.net = net; self.net.to(device)

        self.template_size = template_size
        self.search_size   = search_size
        self.stride        = stride
        self.pad = pad[0]
        self.padval = padval

        self.penalty_k  = penalty_k
        self.window_factor = window_factor
        self.momentum = momentum

        self.state = edict()
        self.template_scale = None

    def set_template(self, template_image, bbox):
        """
        Params:
            template_image: {ndarray(H, W, C)}
            bbox:           {ndarray(4)}        x1, y1, x2, y2
        """
        self._update_state(bbox)
        
        template_image, (self.template_scale, _) = crop_square_according_to_bbox(
            template_image, bbox, self.template_size, self.pad, self.padval, return_param=True)
        # show_bbox(template_image, bbox, winname='template_image')
        
        template_tensor = self._ndarray2tensor(template_image)
        self.net.template(template_tensor)

    def delete_template(self):

        self.net.z_f = None
    
    def template_is_setted(self):

        return self.net.z_f is not None

    def track(self, search_image, mode='crop'):
        """
        Params:
            template_image: {ndarray(H, W, C)}
        Returns:
            res_corner: {ndarray(N, 4)}
            score: {ndarray(N)}
        """
        if mode == 'crop':
            return self._track_crop_image(search_image)
        elif mode == 'whole':
            return self._track_whole_image(search_image)

    def _track_crop_image(self, search_image):
        """
        Params:
            template_image: {ndarray(H, W, C)}
        Returns:
            res_corner: {ndarray(N, 4)}
            score: {ndarray(N)}
        """
        search_crop, (scale, shift) = crop_square_according_to_bbox(
                search_image, self.state.corner, self.search_size, lambda w, h: self.pad(w, h) * 2, self.padval, return_param=True)

        # show_bbox(search_image, self.state.corner, winname='search_image')
        # show_bbox(search_crop, (self.state.corner - np.concatenate([shift, shift])) * scale, winname='search_crop', waitkey=5)
        
        with torch.no_grad(): 
            pred_cls, pred_reg = self.net.track(self._ndarray2tensor(search_crop))   
            pred_cls = [pc.squeeze().cpu().numpy() for pc in pred_cls]             # [(33, 33), (17, 17)]
            pred_reg = [pr.squeeze().cpu().numpy() for pr in pred_reg]             # [(4, 33, 33), (4, 17, 17)]

        res_corner = []; score = []
        for j, (cls_pd, reg_pd, stride) in enumerate(zip(pred_cls, pred_reg, self.stride)):
            
            size = cls_pd.shape[0]

            r = self._r(reg_pd[2], reg_pd[3])
            s = self._s(reg_pd[2], reg_pd[3])
            pr = np.maximum(r / self.state.r, self.state.r / r)
            ps = np.maximum(s / self.state.s, self.state.s / s)
            penalty = np.exp(- (pr * ps - 1) * self.penalty_k)
            pscore = cls_pd * penalty

            window = get_hamming_window(size)
            pscore = window * self.window_factor + \
                            pscore * (1 - self.window_factor)

            # cv2.imshow("pscore_%d" % j, (pscore * 255).astype(np.uint8)); cv2.waitKey(5)
            plt.figure(j); plt.imshow(pscore); plt.show()
            
            r, c = np.unravel_index(pscore.argmax(), pscore.shape)
            x, y = np.array([r, c]) * stride + reg_pd[:2, r, c] + (self.search_size - self.template_size) // 2
            w, h = reg_pd[2:, r, c]
            center = np.array([x, y, w, h])
            
            show_bbox(search_crop, np.array(center2corner(center)), winname='search_crop_output_%d' % j, waitkey=5)

            # ------------------------------------------------------
            # get back!
            center /= scale         # scale, ( w,  h)
            center[:2] += shift     # shift, (xc, yc)
            
            # momentum
            momentum = pscore[r, c] * self.momentum
            center[2:] = center[2:] * momentum + self.state.center[2:] * (1 - momentum)     #  w,  h
            
            corner = np.array(center2corner(center))
            
            res_corner += [corner]; score += [cls_pd[r, c]]
        
        keep = np.argmax(score); score = score[keep]; res_corner = res_corner[keep]
        show_bbox(search_image, res_corner, score, self.state.center[:2], winname='search_image_output', waitkey=0)

        self._update_state(res_corner)

        return res_corner, score

    def _track_whole_image(self, search_image):
        """
        Params:
            template_image: {ndarray(H, W, C)}
        Returns:
            res_corner: {ndarray(N, 4)}
            score: {ndarray(N)}
        """
        raise NotImplementedError("Error!")

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