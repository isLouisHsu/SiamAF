# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-12-02 09:55:52
@LastEditTime: 2019-12-02 15:05:14
@Update: 
'''
from easydict import EasyDict as edict

configer = edict()

# ------------------ Dataset ------------------
configer.vid = edict()

configer.vid.template_size = 127
configer.vid.search_size   = 255
configer.vid.frame_range   = 30
configer.vid.pad = lambda w, h: (w + h) / 2,
configer.vid.blur=0
configer.vid.rotate = 5
configer.vid.scale  = 0.05
configer.vid.color  = 1
configer.vid.flip   = 1

# ------------------ SiamRPN ------------------
configer.siamrpn = edict()

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
configer.siamrpn.loss.reg_weight = 0.01
configer.siamrpn.loss.pos_thr    = 0.6
configer.siamrpn.loss.anchor_thr_low  = 0.3
configer.siamrpn.loss.anchor_thr_high = 0.6
configer.siamrpn.loss.n_pos = 16
configer.siamrpn.loss.n_neg = 48

configer.siamrpn.optimizer = edict()
configer.siamrpn.optimizer.lr = 0.01
configer.siamrpn.optimizer.momentum = 0.9
configer.siamrpn.optimizer.weight_decay = 5e-4

configer.siamrpn.scheduler = edict()
configer.siamrpn.scheduler.gamma = 0.9

configer.siamrpn.train = edict()
configer.siamrpn.train.batch_size = 48
configer.siamrpn.train.log_dir = '../logs/siamrpn'
configer.siamrpn.train.ckpt = '../logs/siamrpn/siamrpn.pkl'
configer.siamrpn.train.cuda = True
configer.siamrpn.train.n_epoch = 50