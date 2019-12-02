# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-12-02 10:31:12
@LastEditTime: 2019-12-02 16:19:58
@Update: 
'''
import sys
sys.path.append('..')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config import configer
from dataset.VID2015 import VID2015PairData
from models.tracker import SiamRPN
from models.loss    import RpnLoss
from utils.box_utils import get_anchor

use_cuda = cuda.is_available() and configer.siamrpn.train.cuda

def train(configer):
    """
    Params:
        configer: {edict}
    """
    params = configer.siamrpn.train

    # datasets
    trainset = VID2015PairData('train',                     **configer.vid)
    validset = VID2015PairData('val',                       **configer.vid)
    trainloader = DataLoader(trainset, params.batch_size, shuffle=True)
    validloader = DataLoader(validset, params.batch_size, shuffle=True)

    # model
    net = SiamRPN(**configer.siamrpn.net)
    if use_cuda: net.cuda()
    
    # optimize
    loss = RpnLoss(get_anchor(**configer.siamrpn.anchor),   **configer.siamrpn.loss)
    optimizer = optim.Adam(net.parameters(),                 **configer.siamrpn.optimizer)
    scheduler = lr_scheduler.ExponentialLR(optimizer,       **configer.siamrpn.scheduler)

    # train
    writer = SummaryWriter(params.log_dir)

    loss_total_val_best = np.inf

    for i_epoch in range(params.n_epoch):

        if use_cuda: cuda.empty_cache()
        scheduler.step(i_epoch)
        writer.add_scalar('lr', scheduler.get_lr()[-1])

        # -----------------------------------------------------
        net.train()
        loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg = [], [], [], []
        for i_batch, batch in enumerate(trainloader):

            z, _, x, gt = list(map(lambda x: Variable(x).float(), batch))
            if use_cuda: z = z.cuda(); x = x.cuda(); gt = gt.cuda()
            pred_cls, pred_reg = net(z, x)
            loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i = loss(pred_cls, pred_reg, gt)
            optimizer.zero_grad(); loss_total_i.backward(); optimizer.step()

            loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i = list(
                map(lambda x: x.detach().unsqueeze(0), [loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i]))
            loss_total_avg += [loss_total_i]; loss_cls_avg   += [loss_cls_i  ]; loss_reg_avg   += [loss_reg_i  ]; acc_cls_avg    += [acc_cls_i   ]

        loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg = list(
            map(lambda x: torch.cat(x).mean(), [loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg]))
        writer.add_scalars('train', {
                "loss_total":   loss_total_avg, 
                "loss_cls_avg": loss_cls_avg, 
                "loss_reg_avg": loss_reg_avg, 
                "acc_cls_avg":  acc_cls_avg}, global_step=i_epoch)
        
        # -----------------------------------------------------
        net.eval()
        loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg = [], [], [], []
        
        with torch.no_grad():
            for i_batch, batch in enumerate(validloader):

                z, _, x, gt = list(map(lambda x: Variable(x).float(), batch))
                if use_cuda: z = z.cuda(); x = x.cuda(); gt = gt.cuda()
                pred_cls, pred_reg = net(z, x)
                loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i = loss(pred_cls, pred_reg, gt)

                loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i = list(
                    map(lambda x: x.detach().unsqueeze(0), [loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i]))
                loss_total_avg += [loss_total_i]; loss_cls_avg   += [loss_cls_i  ]; loss_reg_avg   += [loss_reg_i  ]; acc_cls_avg    += [acc_cls_i   ]

        loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg = list(
            map(lambda x: torch.cat(x).mean(), [loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg]))
        writer.add_scalars('valid', {
                "loss_total":   loss_total_avg, 
                "loss_cls_avg": loss_cls_avg, 
                "loss_reg_avg": loss_reg_avg, 
                "acc_cls_avg":  acc_cls_avg}, global_step=i_epoch)

        if loss_total_avg < loss_total_val_best:
            loss_total_val_best = loss_total_avg
            torch.save(net.state_dict(), params.ckpt)

    writer.close()

if __name__ == '__main__':

    train(configer)