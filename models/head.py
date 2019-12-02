# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-11-30 15:28:34
@LastEditTime: 2019-12-01 12:59:23
@Update: 
'''
import torch
from torch import nn
from torch.nn import functional as F


class RpnHead(nn.Module):
    """ SiamRPN head """

    def __init__(self, in_channels, out_channels=256, num_anchor=5):
        super(RpnHead, self).__init__()
    
        self.template_cls = nn.Conv2d(in_channels, out_channels * num_anchor,     kernel_size=3)
        self.template_reg = nn.Conv2d(in_channels, out_channels * num_anchor * 4, kernel_size=3)

        self.search_cls   = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.search_reg   = nn.Conv2d(in_channels, out_channels, kernel_size=3)

        self.adjust = nn.Conv2d(num_anchor * 4, num_anchor * 4, kernel_size=1)

    def _conv(self, f, k):
        """
        Params:
            f: {tensor(N, ch, Hf, Wf)} 
            k: {tensor(N, ch * k, Hk, Wk)} 
        Returns:
            po: {tensor(N, k, H, W)}
        """
        n, ch = f.size()[:2]
        
        px = f.view( 1, -1, *f.size()[2:])      # (    1, N * ch, Hf, Wf)
        pk = k.view(-1, ch, *k.size()[2:])      # (N * k,     ch, Hk, Wk)
        po = F.conv2d(px, pk, groups=n)         # (    1, N *  k,  H,  W)
        po = po.view(n, -1, *po.size()[2:])     # (    N,      k,  H,  W)
        return po

    def forward(self, z_f, x_f):
        """
        Params:
            z_f: {tensor(N, in_channels, Hz, Wz)} feature extracted from template
            x_f: {tensor(N, in_channels, Hx, Wx)} feature extracted from search
        Returns:
            pred_cls: {tensor(N,    num_anchor, H, W)}
            pred_reg: {tensor(N, 4, num_anchor, H, W)}
        """
        cls_feature = self.search_cls(x_f)      # (N, out_channels, Hx', Wx')
        reg_feature = self.search_reg(x_f)      # (N, out_channels, Hx', Wx')

        cls_kernel = self.template_cls(z_f)     # (N, out_channels * num_anchor,     Hz', Wz')
        reg_kernel = self.template_reg(z_f)     # (N, out_channels * num_anchor * 4, Hz', Wz')

        cls_kernel = cls_kernel.view(-1, cls_feature.size(1), *cls_kernel.size()[2:])   # (N, out_channels * num_anchor,     Hz', Wz')
        reg_kernel = reg_kernel.view(-1, reg_feature.size(1), *reg_kernel.size()[2:])   # (N, out_channels * num_anchor * 4, Hz', Wz')
        
        pred_cls = self._conv(cls_feature, cls_kernel)                  # (N, num_anchor, H, W)
        pred_reg = self.adjust(self._conv(reg_feature, reg_kernel))     # (N, num_anchor * 4, H, W)

        size = [pred_cls.size(0), 4, *pred_cls.size()[1:]]
        pred_reg = pred_reg.view(size)

        return pred_cls, pred_reg
                

class HeatmapHead(nn.Module):
    """ SiamAF head """
    def __init__(self, in_channels):
        super(HeatmapHead, self).__init__()
        
        self.heat = nn.Conv2d(in_channels, 1, kernel_size=3)
        self.reg  = nn.Conv2d(in_channels, 4, kernel_size=3)
        self.adjust = nn.Conv2d(4, 4, kernel_size=1)

    def _conv(self, f, k):
        """
        Params:
            f: {tensor(N, ch, Hf, Wf)} 
            k: {tensor(N, ch, Hk, Wk)} 
        Returns:
            po: {tensor(N, ch, H, W)}
        """
        n, ch = f.size()[:2]
        
        px = f.view( 1, -1, *f.size()[2:])      # (     1, N * ch, Hf, Wf)
        pk = k.view(-1,  1, *k.size()[2:])      # (N * ch,      1, Hk, Wk)
        po = F.conv2d(px, pk, groups=n*ch)      # (     1, N * ch,  H,  W)
        po = po.view(n, -1, *po.size()[2:])     # (     N,     ch,  H,  W)
        return po

    def forward(self, z_f, x_f):
        """
        Params:
            z_f: {tensor(N, in_channels, Hz, Wz)} feature extracted from template
            x_f: {tensor(N, in_channels, Hx, Wx)} feature extracted from search
        Returns:
            pred_map: {tensor(N, 1, H, W)}
            pred_reg: {tensor(N, 4, H, W)}
        """
        corr = self._conv(x_f, z_f)
        pred_heat = self.heat(corr)
        pred_reg  = self.reg (corr)

        return pred_heat, pred_reg
        

if __name__ == "__main__":
    
    in_channels = 256
    m = RpnHead(in_channels)
    # m = AfHead(in_channels)
    z = torch.rand(5, in_channels,  6,  6)
    x = torch.rand(5, in_channels, 22, 22)
    m(z, x)