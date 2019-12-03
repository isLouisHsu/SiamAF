# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-12-01 14:23:43
@LastEditTime: 2019-12-03 12:28:38
@Update: 
'''
import sys
sys.path.append('../')

import os
import cv2
import numpy as np
import xml.dom.minidom as minidom

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter

from easydict import EasyDict as edict

from utils.image_augmentation import *
from utils.box_utils import corner2center, center2corner, show_bbox

# from config import configer
# from utils.box_utils import get_anchor, visualize_anchor
# center, corner = get_anchor(**configer.siamrpn.anchor)

class VID2015PairData(Dataset):
    """
    Params:
        mode: {str} `train` or `val`
    Notes:
        data/
        └── ILSVRC2015_VID
            └── ILSVRC2015
                ├── Annotations
                │   └── VID
                ├── Data
                │   └── VID
                └── ImageSets
                    └── VID
    """
    PATH = '../data/ILSVRC2015_VID/ILSVRC2015/{subdir}/VID/{mode}'

    def __init__(self, mode, 
                template_size=127, search_size=255, frame_range=30, pad=[lambda w, h: (w + h) / 2],
                blur=0, rotate=5, scale=0.05, color=1, flip=1):

        self.mode = mode
        self.template_size = template_size
        self.search_size   = search_size
        self.frame_range   = frame_range
        self.pad = pad[0]
        
        self.blur   = blur
        self.rotate = rotate
        self.scale  = scale
        self.color  = color
        self.flip   = flip
        
        self.transformer = transforms.Compose(
            [transforms.ToPILImage(), ] +
            ([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), ] if color > np.random.rand() else [])
            + ([transforms.RandomHorizontalFlip(), ] if flip > np.random.rand() else [])
        )
        
        self._list_samples()
        self.n_videos = len(self._video_folders)

        print("Dataset [{}] loaded! Totally {} videos".format(mode, self.n_videos))

    def __getitem__(self, index):

        folder = self._video_folders[index]
        datapath = os.path.join(self.PATH.format(subdir='Data',        mode=self.mode), folder)
        annopath = os.path.join(self.PATH.format(subdir='Annotations', mode=self.mode), folder)

        # get a pair
        frames = os.listdir(annopath)
        indexs = list(map(lambda x: int(x.split('.')[0]), frames))

        isdone = False
        while not isdone:
            template_idx = np.random.choice(indexs)
            search_idxs = list(filter(
                lambda x: x > template_idx - self.frame_range and \
                        x < template_idx + self.frame_range + 1 and \
                        x != template_idx, indexs))
            search_idx = np.random.choice(search_idxs)
            template_anno = self._read_annotation_xml(os.path.join(annopath, '{:06d}.xml'.format(template_idx)))
            search_anno   = self._read_annotation_xml(os.path.join(annopath, '{:06d}.xml'.format(search_idx  )))
            common_id = [k for k in template_anno.keys() if k in search_anno.keys()]
            if len(common_id)> 0: isdone = True
            
        # randomly choose an object
        trackid = np.random.choice(common_id)
        template_bbox = template_anno[trackid]
        search_bbox   = search_anno  [trackid]
        
        # ------------- read images -------------
        template_image = cv2.imread(os.path.join(datapath, '{:06d}.JPEG'.format(template_frame_index)))
        search_image   = cv2.imread(os.path.join(datapath, '{:06d}.JPEG'.format(search_frame_index  )))

        # show_bbox(template_image, template_bbox, '[line95] template %d' % template_frame_index)
        # show_bbox(search_image, search_bbox, '[line96] search %d' % search_frame_index)

        # --------- augmentation and crop -------
        template_image, template_bbox = self._augment_crop(template_image, template_bbox, self.template_size)
        search_image, search_bbox = self._augment_crop(search_image, search_bbox, self.search_size)

        # show_bbox(template_image, template_bbox, '[line102] template %d' % template_frame_index)
        # show_bbox(search_image, search_bbox, '[line103] search %d' % search_frame_index)
        # visualize_anchor(search_image, corner[:, :, 8, 8].T)

        # ------------ to tensor ----------------
        template_image = torch.from_numpy(template_image.transpose(2, 0, 1) / 255.)
        search_image   = torch.from_numpy(search_image.transpose(2, 0, 1)   / 255.)
        template_bbox  = torch.from_numpy(template_bbox)
        search_bbox    = torch.from_numpy(search_bbox)

        return template_image, template_bbox, search_image, search_bbox

    def __len__(self):

        return self.n_videos
    
    def vis(self, index):

        folder = self._video_folders[index]
        datapath = os.path.join(self.PATH.format(subdir='Data',        mode=self.mode), folder)
        annopath = os.path.join(self.PATH.format(subdir='Annotations', mode=self.mode), folder)

        n_frames = len(os.listdir(annopath))
        for i_frames in range(n_frames):

            image = cv2.imread(os.path.join(datapath, '{:06d}.JPEG'.format(i_frames)))
            anno  = self._read_annotation_xml(os.path.join(annopath, '{:06d}.xml'.format(i_frames)))
            try:
                bboxes = np.stack([v for v in anno.values()])
            except:
                continue
            show_bbox(image, bboxes, 'vis%d' % index, 30)

    def _list_samples(self):

        self._video_folders = []
        path = self.PATH.format(subdir='Annotations', mode=self.mode)

        if self.mode == 'train':
            for vol in os.listdir(path):
                for vd in os.listdir(os.path.join(path, vol)):
                    self._video_folders += [os.path.join(vol, vd)]
        else:
            for vd in os.listdir(path):
                self._video_folders += [vd]

    def _read_annotation_xml(self, xmlpath):
        """
        Params:
            xmlpath: {str} path
        Returns:
            filename: {str} e.g. 2008_003523.jpg
            objects:  {dict{str: list[int, int, int, int]}} 
                    e.g. {'pottedplant': [218, 235, 375, 500], ...}
        """
        annotations = dict()

        element = minidom.parse(xmlpath).documentElement
        objects = element.getElementsByTagName("object")
        for obj in objects:
            trackid = int(obj.getElementsByTagName("trackid")[0].firstChild.data)
            bndbox    = obj.getElementsByTagName("bndbox")[0]
            bbox = []
            for locat in ["xmin", "ymin", "xmax", "ymax"]:      # x1, y1, x2, y2
                bbox += [int(bndbox.getElementsByTagName(locat)[0].firstChild.data)]
            
            annotations[trackid] = np.array(bbox, dtype=np.float32)

        return annotations

    def _augment_crop(self, im, bbox, size):
        """
        Params:
            im: {ndarray()}
            bbox: {}
            size: {int}
        """
        imh, imw = im.shape[:-1]
        padval = im.mean(0).mean(0)

        # rotate & scale
        rows, cols = im.shape[:-1]
        rotate_rand = self.rotate * (np.random.rand() * 2. - 1)
        scale_rand  = self.scale  * (np.random.rand() * 2. - 1) + 1

        xc, yc, w, h = corner2center(bbox)
        M = cv2.getRotationMatrix2D((xc, yc), rotate_rand, scale_rand)
        im = cv2.warpAffine(im, M, (imw, imh), borderMode=cv2.BORDER_CONSTANT, borderValue=padval)

        bbox = np.array(center2corner(np.array([xc, yc, w * scale_rand, h * scale_rand])))
        xc, yc, w, h = corner2center(bbox)

        # show_bbox(im, bbox, '[line196]')

        # crop
        p = self.pad(w, h)
        a = int(np.sqrt((w + p) * (h + p)) * (size // self.template_size))
        x1 = xc - a // 2; y1 = yc - a // 2
        x2 = x1 + a; y2 = y1 + a

        bbox[[0, 2]] -= x1; bbox[[1, 3]] -= y1
        xc_, yc_, w_, h_ = corner2center(bbox)
        
        M = np.float32([
                [1, 0, xc_ - xc], 
                [0, 1, yc_ - yc]
            ])
        im = cv2.warpAffine(im, M, (a, a), borderMode=cv2.BORDER_CONSTANT, borderValue=padval)
        
        # transform
        im = np.array(self.transformer(im))
        
        # blur
        if self.blur > np.random.rand():
            im = gaussian_filter(im, sigma=(1, 1, 0))

        # resize
        im = cv2.resize(im, (size, size))
        s = a / size
        bbox /= s

        return im, bbox

if __name__ == '__main__':

    dataset = VID2015PairData('train')
    dataset = VID2015PairData('val')

    # for i in range(len(dataset)):
    #     dataset.vis(i)

    for i, (template, template_bbox, search, search_bbox) in enumerate(dataset):

        pass
            

