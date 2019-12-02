# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-12-02 09:55:52
@LastEditTime: 2019-12-02 10:41:46
@Update: 
'''
from easydict import EasyDict as edict

configer = edict()

# ------------------ Dataset ------------------
configer.vid = edict()

configer.vid.template_size = 127
configer.vid.search_size   = 255
configer.vid.frame_range   = 30
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

configer.siamrpn.model = edict()
configer.siamrpn.model.out_channels = 256
configer.siamrpn.model.num_anchor   = \
    len(configer.siamrpn.anchor.anchor_ratios) * \
    len(configer.siamrpn.anchor.anchor_scales)

configer.siamrpn.train = edict()
configer.siamrpn.train.batch_size = 32