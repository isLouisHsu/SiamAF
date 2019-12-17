# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-12-02 09:55:52
@LastEditTime: 2019-12-17 09:51:34
@Update: 
'''
from easydict import EasyDict as edict

configer = edict()

# ------------------ SiamRPN ------------------
configer.siamrpn = edict()

configer.siamrpn.vid = edict()
configer.siamrpn.vid.template_size = 127
configer.siamrpn.vid.search_size   = 255
configer.siamrpn.vid.frame_range   = 100
configer.siamrpn.vid.pad = lambda w, h: (w + h) / 2,
configer.siamrpn.vid.blur= 1
configer.siamrpn.vid.rotate = 5
configer.siamrpn.vid.scale  = 0.05
configer.siamrpn.vid.color  = 1
configer.siamrpn.vid.flip   = 1
configer.siamrpn.vid.mshift = 32

configer.siamrpn.anchor = edict()
configer.siamrpn.anchor.stride = 8
configer.siamrpn.anchor.template_size = 127
configer.siamrpn.anchor.search_size   = 255
configer.siamrpn.anchor.feature_size  = 17
configer.siamrpn.anchor.anchor_ratios = [0.33, 0.5, 1., 2., 3.]
configer.siamrpn.anchor.anchor_scales = [8]
configer.siamrpn.anchor.vis_anchor  = False

configer.siamrpn.net = edict()
configer.siamrpn.net.out_channels = 256
configer.siamrpn.net.num_anchor   = \
    len(configer.siamrpn.anchor.anchor_ratios) * \
    len(configer.siamrpn.anchor.anchor_scales)

configer.siamrpn.loss = edict()
configer.siamrpn.loss.ct_weight  = .1
configer.siamrpn.loss.cls_weight = 1.
configer.siamrpn.loss.reg_weight = .5
configer.siamrpn.loss.pos_thr    = 0.9
configer.siamrpn.loss.anchor_thr_low  = 0.3
configer.siamrpn.loss.anchor_thr_high = 0.6
configer.siamrpn.loss.n_pos = 16
configer.siamrpn.loss.n_neg = 48
configer.siamrpn.loss.r_pos = 2

configer.siamrpn.optimizer = edict()
configer.siamrpn.optimizer.lr = 0.001
configer.siamrpn.optimizer.weight_decay = 5e-4

configer.siamrpn.scheduler = edict()
configer.siamrpn.scheduler.gamma = 0.9

configer.siamrpn.train = edict()
configer.siamrpn.train.batch_size = 48
configer.siamrpn.train.clip_grad  = 10
configer.siamrpn.train.log_dir = '../logs/siamrpn'
configer.siamrpn.train.ckpt = '../ckpt/siamrpn.pkl'
configer.siamrpn.train.cuda = True
configer.siamrpn.train.n_epoch = 100
configer.siamrpn.train.resume = None

configer.siamrpn.tracker = edict()
configer.siamrpn.tracker.pad           = configer.siamrpn.vid.pad
configer.siamrpn.tracker.template_size = configer.siamrpn.anchor.template_size
configer.siamrpn.tracker.search_size   = configer.siamrpn.anchor.search_size
configer.siamrpn.tracker.feature_size  = configer.siamrpn.anchor.feature_size
configer.siamrpn.tracker.center_size   = 7
configer.siamrpn.tracker.penalty_k     = 0.055
configer.siamrpn.tracker.window_factor = 0.32
configer.siamrpn.tracker.momentum      = 0.005


# ------------------ SiamAF ------------------
configer.siamaf = edict()

configer.siamaf.vid = edict()
configer.siamaf.vid.template_size = 127
configer.siamaf.vid.search_size   = 255
configer.siamaf.vid.frame_range   = 20
configer.siamaf.vid.pad = lambda w, h: (w + h) / 2,
configer.siamaf.vid.blur= 1
configer.siamaf.vid.rotate = 5
configer.siamaf.vid.scale  = 0.05
configer.siamaf.vid.color  = 1
configer.siamaf.vid.flip   = 1
configer.siamaf.vid.mshift = 32

configer.siamaf.net = edict()
configer.siamaf.net.roi_size = None

configer.siamaf.loss = edict()
configer.siamaf.loss.cls_weight = 1.0
configer.siamaf.loss.reg_weight = 1.
configer.siamaf.loss.stride = [4, 8]


configer.siamaf.optimizer = edict()
configer.siamaf.optimizer.lr = 0.001
configer.siamaf.optimizer.weight_decay = 5e-4

configer.siamaf.scheduler = edict()
configer.siamaf.scheduler.gamma = 0.9

configer.siamaf.train = edict()
configer.siamaf.train.batch_size = 8
configer.siamaf.train.clip_grad  = 10
configer.siamaf.train.log_dir = '../logs/siamaf'
configer.siamaf.train.ckpt = '../ckpt/siamaf.pkl'
configer.siamaf.train.cuda = False
configer.siamaf.train.n_epoch = 70
configer.siamaf.train.resume = None
