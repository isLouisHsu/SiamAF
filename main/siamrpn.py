# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-12-02 10:31:12
@LastEditTime : 2019-12-26 10:39:59
@Update: 
'''
import os
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
from dataset.VID2015 import VID2015PairData, VID2015PairDataV2, VID2015SequenceData
from models.network import SiamRPN
from models.loss    import RpnLoss
from utils.box_utils import get_anchor

use_cuda = cuda.is_available() and configer.siamrpn.train.cuda
device = torch.device('cuda:0' if use_cuda else 'cpu')

def train(configer):
    """
    Params:
        configer: {edict}
    """
    params = configer.siamrpn.train

    # datasets
    # trainset = VID2015PairData('train',                     **configer.siamrpn.vid)
    # validset = VID2015PairData('val',                       **configer.siamrpn.vid)
    trainset = VID2015PairDataV2('train',                   **configer.siamrpn.vid)
    validset = VID2015PairDataV2('val',                     **configer.siamrpn.vid)
    trainloader = DataLoader(trainset, params.batch_size, shuffle=True)
    validloader = DataLoader(validset, params.batch_size, shuffle=True)

    # model
    net = SiamRPN(**configer.siamrpn.net)
    if params.resume is not None and os.path.exists(params.resume):
        state = torch.load(params.resume, map_location='cpu')
        net.load_state_dict(state)
    net.to(device)
    
    # optimize
    loss = RpnLoss(get_anchor(**configer.siamrpn.anchor),   **configer.siamrpn.loss)
    optimizer = optim.Adam(net.parameters(),                **configer.siamrpn.optimizer)
    scheduler = lr_scheduler.ExponentialLR(optimizer,       **configer.siamrpn.scheduler)

    # train
    writer = SummaryWriter(params.log_dir)

    loss_total_val_best = np.inf

    for i_epoch in range(params.n_epoch):

        if use_cuda: cuda.empty_cache()
        writer.add_scalar('lr', scheduler.get_lr()[-1], global_step=i_epoch)

        # -----------------------------------------------------
        net.train()
        loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg = [], [], [], []
        for i_batch, batch in enumerate(trainloader):

            z, _, x, gt = list(map(lambda x: Variable(x).float(), batch))
            z = z.to(device); x = x.to(device); gt = gt.to(device)
            pred_cls, pred_reg = net(z, x)
            loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i = loss(pred_cls, pred_reg, gt)

            try:
                optimizer.zero_grad()
                loss_total_i.backward()
                nn.utils.clip_grad_norm_(net.parameters(), params.clip_grad)   # gradient clip
                optimizer.step()
            except:
                pass

            loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i = list(
                map(lambda x: x.cpu().detach().unsqueeze(0), [loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i]))
            loss_total_avg += [loss_total_i]; loss_cls_avg   += [loss_cls_i  ]; loss_reg_avg   += [loss_reg_i  ]; acc_cls_avg    += [acc_cls_i   ]

            writer.add_scalars('training', {
                    "loss_total_i":   loss_total_i, 
                    "loss_cls_i": loss_cls_i, 
                    "loss_reg_i": loss_reg_i, 
                    "acc_cls_i":  acc_cls_i}, global_step=i_epoch * len(trainloader) + i_batch)

        loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg = list(
            map(lambda x: torch.cat(x).mean(), [loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg]))
        writer.add_scalars('train', {
                "loss_total_avg":   loss_total_avg, 
                "loss_cls_avg": loss_cls_avg, 
                "loss_reg_avg": loss_reg_avg, 
                "acc_cls_avg":  acc_cls_avg}, global_step=i_epoch)
        
        # -----------------------------------------------------
        net.eval()
        loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg = [], [], [], []
        
        with torch.no_grad():
            for i_batch, batch in enumerate(validloader):

                z, _, x, gt = list(map(lambda x: Variable(x).float(), batch))
                z = z.to(device); x = x.to(device); gt = gt.to(device)
                pred_cls, pred_reg = net(z, x)
                loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i = loss(pred_cls, pred_reg, gt)

                loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i = list(
                    map(lambda x: x.cpu().detach().unsqueeze(0), [loss_total_i, loss_cls_i, loss_reg_i, acc_cls_i]))
                loss_total_avg += [loss_total_i]; loss_cls_avg   += [loss_cls_i  ]; loss_reg_avg   += [loss_reg_i  ]; acc_cls_avg    += [acc_cls_i   ]

        loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg = list(
            map(lambda x: torch.cat(x).mean(), [loss_total_avg, loss_cls_avg, loss_reg_avg, acc_cls_avg]))
        writer.add_scalars('valid', {
                "loss_total_avg":   loss_total_avg, 
                "loss_cls_avg": loss_cls_avg, 
                "loss_reg_avg": loss_reg_avg, 
                "acc_cls_avg":  acc_cls_avg}, global_step=i_epoch)

        if loss_total_avg < loss_total_val_best:
            loss_total_val_best = loss_total_avg
            torch.save(net.state_dict(), params.ckpt)
        
        scheduler.step(i_epoch)

    writer.close()

# --------------------------------------------------------

import cv2
from models.tracker import SiamRPNTracker
from utils.box_utils import naive_anchors, show_bbox

def testSequence(configer):

    dataset = VID2015SequenceData('val', **configer.siamrpn.vid)

    # initialize anchor
    anchors_center, _ = get_anchor(**configer.siamrpn.anchor)

    # initialize network
    net = SiamRPN(**configer.siamrpn.net)
    net.load_state_dict(
        torch.load(configer.siamrpn.train.ckpt, map_location='cpu'))

    tracker = SiamRPNTracker(anchor=configer.siamrpn.anchor, net=net, device=device, **configer.siamrpn.tracker)

    for i_data, (impaths, annos, ids) in enumerate(dataset):

        # if i_data < 14: continue

        if len(ids) == 0: continue

        for i_id, obj_id in enumerate(ids):

            anno = list(map(lambda x: x[obj_id] if obj_id in x.keys() else None, annos))

            for i_frame, (impath, bbox_gt) in enumerate(zip(impaths, anno)):
                
                image = cv2.imread(impath, cv2.IMREAD_COLOR)
                if bbox_gt is not None and not tracker.template_is_setted():
                    tracker.set_template(image, bbox_gt)
                    continue

                bbox_pred, _ = tracker.track(image, 'crop')
                # bbox_pred, _ = tracker.track(image, 'whole')

                show_bbox(image, bbox_gt, waitkey=5, winname='gt')
                show_bbox(image, bbox_pred, waitkey=5, winname='pred')

                # tracker.set_template(image, bbox_gt)
            
            tracker.delete_template()

if __name__ == '__main__':

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', default='test')
    args = parser.parse_args()

    if args.mode == 'train':
        train(configer)
    elif args.mode == 'test':
        testSequence(configer)