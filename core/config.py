# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-12-02 09:55:52
@LastEditTime: 2019-12-02 09:57:47
@Update: 
'''
from easydict import EasyDict as edict

configer = edict()

# ------------------ Dataset ------------------
configer.vid = edict()
configer.vid.template_size = 127
configer.vid.search_size   = 255
configer.vid.frame_range   = 30
# ------------------ SiamRPN ------------------
configer.siamrpn = edict()

