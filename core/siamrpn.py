# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-12-02 10:31:12
@LastEditTime: 2019-12-02 10:42:58
@Update: 
'''
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config import configer
from dataset.VID2015 import VID2015PairData
from models.tracker import SiamRPN
from models.loss    import RpnLoss

def train(configer):
    """
    Params:
        configer: {edict}
    """
    trainset = VID2015PairData('train', **configer.vid)
    validset = VID2015PairData('val',   **configer.vid)
    trainloader = 

if __name__ == '__main__':

    train(configer)