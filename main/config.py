# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-12-02 09:55:52
@LastEditTime: 2019-12-03 21:08:03
@Update: 
'''
from easydict import EasyDict as edict

configer = edict()

# ------------------ SiamRPN ------------------
configer.siamrpn = edict()

configer.siamrpn.vid = edict()
configer.siamrpn.vid.template_size = 127
configer.siamrpn.vid.search_size   = 255
configer.siamrpn.vid.frame_range   = 30
configer.siamrpn.vid.pad = lambda w, h: (w + h) / 2,
configer.siamrpn.vid.blur=0
configer.siamrpn.vid.rotate = 0
configer.siamrpn.vid.scale  = 0.05
configer.siamrpn.vid.color  = 1
configer.siamrpn.vid.flip   = 1

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
configer.siamrpn.loss.cls_weight = 1.0
configer.siamrpn.loss.reg_weight = 0.1
configer.siamrpn.loss.pos_thr    = 0.6
configer.siamrpn.loss.anchor_thr_low  = 0.3
configer.siamrpn.loss.anchor_thr_high = 0.6
configer.siamrpn.loss.n_pos = 16
configer.siamrpn.loss.n_neg = 48

configer.siamrpn.optimizer = edict()
configer.siamrpn.optimizer.lr = 0.001
configer.siamrpn.optimizer.weight_decay = 5e-4

configer.siamrpn.scheduler = edict()
configer.siamrpn.scheduler.gamma = 0.9

configer.siamrpn.train = edict()
configer.siamrpn.train.batch_size = 48
configer.siamrpn.train.log_dir = '../logs/siamrpn'
configer.siamrpn.train.ckpt = '../ckpt/siamrpn.pkl'
configer.siamrpn.train.cuda = True
configer.siamrpn.train.n_epoch = 50
configer.siamrpn.train.resume = '../ckpt/siamrpn.pkl'

configer.siamrpn.tracker = edict()
configer.siamrpn.tracker.template_size = configer.siamrpn.anchor.template_size
configer.siamrpn.tracker.search_size   = configer.siamrpn.anchor.search_size
configer.siamrpn.tracker.feature_size  = configer.siamrpn.anchor.feature_size
configer.siamrpn.anchor.stride         = configer.siamrpn.anchor.stride
configer.siamrpn.tracker.pad           = configer.siamrpn.vid.pad
configer.siamrpn.tracker.cls_thresh    = 0.9999
configer.siamrpn.tracker.nms_thresh    = 0.6