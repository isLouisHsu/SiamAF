# -*- coding: utf-8 -*-
import cv2
import math
import torch
import numpy as np

def corner2center(corner):
    """
    [x1, y1, x2, y2] --> [cx, cy, w, h]
    """
    x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
    x = (x1 + x2) * 0.5
    y = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h

def center2corner(center):
    """
    [cx, cy, w, h] --> [x1, y1, x2, y2]
    """
    x, y, w, h = center[0], center[1], center[2], center[3]
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return x1, y1, x2, y2

def naive_anchors(ratios=[0.33, 0.5, 1, 2, 3], scalers=[8], stride=8):
    """
    anchors corresponding to score map
    """
    anchor_nums = len(ratios) * len(scalers)
    anchors_naive = np.zeros((anchor_nums, 4), dtype=np.float32)  # (5, 4)
    size = stride * stride                                        # 64
    count = 0
    for r in ratios:
        ws = int(math.sqrt(size*1. / r))
        hs = int(ws * r)

        for s in scalers:
            w = ws * s
            h = hs * s
            anchors_naive[count][:] = [-w*0.5, -h*0.5, w*0.5, h*0.5][:]
            count += 1
    return anchors_naive

# def pair_anchors(anchors_naive, center=255//2, feature_size=17, stride=8):
#     """

#     anchors corresponding to pairs
#     :param center: center of search image
#     :param feature_size: output score size after cross-correlation
#     :return: anchors not corresponding to ground truth
#     """
#     anchor_nums = anchors_naive.shape[0]
    
#     a0x = center - feature_size // 2 * stride    # 255 // 2 - 17 // 2 * 8 = 63
#     ori = np.array([a0x] * 4, dtype=np.float32)     # [63, 63, 63, 63]
#     zero_anchors = anchors_naive + ori

#     x1 = zero_anchors[:, 0]
#     y1 = zero_anchors[:, 1]
#     x2 = zero_anchors[:, 2]
#     y2 = zero_anchors[:, 3]

#     x1, y1, x2, y2 = map(lambda x: x.reshape(anchor_nums, 1, 1), [x1, y1, x2, y2])
#     cx, cy, w, h = corner2center([x1, y1, x2, y2])

#     disp_x = np.arange(0, feature_size).reshape(1, 1, -1) * stride
#     disp_y = np.arange(0, feature_size).reshape(1, -1, 1) * stride

#     cx = cx + disp_x
#     cy = cy + disp_y

#     zero = np.zeros((anchor_nums, feature_size, feature_size), dtype=np.float32)
#     cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
#     x1, y1, x2, y2 = center2corner([cx, cy, w, h])

#     center = np.stack([cx, cy, w, h])
#     corner = np.stack([x1, y1, x2, y2])

#     return center, corner

def pair_anchors(anchors_naive, score_size=(17, 17), 
        search_size=255, feature_size=17, stride=8):
    """

    anchors corresponding to pairs
    :param center: center of search image
    :param feature_size: output score size after cross-correlation
    :return: anchors not corresponding to ground truth
    """
    anchor_nums = anchors_naive.shape[0]
    
    a0x = search_size // 2 - feature_size // 2 * stride    # 255 // 2 - 17 // 2 * 8 = 63
    ori = np.array([a0x] * 4, dtype=np.float32)     # [63, 63, 63, 63]
    zero_anchors = anchors_naive + ori

    x1 = zero_anchors[:, 0]
    y1 = zero_anchors[:, 1]
    x2 = zero_anchors[:, 2]
    y2 = zero_anchors[:, 3]

    x1, y1, x2, y2 = map(lambda x: x.reshape(anchor_nums, 1, 1), [x1, y1, x2, y2])
    cx, cy, w, h = corner2center([x1, y1, x2, y2])

    disp_x = np.arange(0, score_size[1]).reshape(1, 1, -1) * stride
    disp_y = np.arange(0, score_size[0]).reshape(1, -1, 1) * stride

    cx = cx + disp_x
    cy = cy + disp_y

    zero = np.zeros((anchor_nums, score_size[0], score_size[1]), dtype=np.float32)
    cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
    x1, y1, x2, y2 = center2corner([cx, cy, w, h])

    center = np.stack([cx, cy, w, h])
    corner = np.stack([x1, y1, x2, y2])

    return center, corner


def get_anchor_train(stride=8, template_size=127, search_size=255, feature_size=17,
            anchor_ratios=[0.33, 0.5, 1., 2., 3.], anchor_scales=[8], vis_anchor=False):
    """
    Params:
        stride:        {int}
        template_size: {int}
        search_size:   {int}
        feature_size:  {int}
        anchor_ratios: {list[int]}
        anchor_scales: {list[int]}
    Returns:
        center: {ndarray(A, 4)}
        corner: {ndarray(A, 4)}
    """
    anchors_naive = naive_anchors(anchor_ratios, anchor_scales, stride)
    center, corner = pair_anchors(
                anchors_naive, (feature_size, feature_size), search_size, feature_size, stride)

    if vis_anchor:
        visualize_anchor(np.full((search_size, search_size, 3), 255, dtype=np.uint8), 
                corner[:, :, feature_size//2, feature_size//2].T)

    return center, corner

def visualize_anchor(im, anchor):
    """
    Params:
        imsize: {int}
        anchor: {ndarray(n, 4)}
    """
    for a in anchor:
        x1, y1, x2, y2 = list(map(int, a))
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 0), thickness=0)
    cv2.imshow("", im)
    cv2.waitKey(0)

def encode(location, anchor_center):
    """
    Params:
        location:       {tensor(4), double} xc, yc,  w,  h
        anchor_center:  {tensor(A, 4)}      xc, yc,  w,  h
    Returns:
        offset:         {tensor(A, 4), double} xc, yc,  w,  h
    Notes:
        oxc = (bxc - axc) / w
        oyc = (byc - ayc) / h
        ow = log (bw / aw)
        oh = log (bh / ah)
    """
    offset = torch.zeros_like(anchor_center, dtype=torch.float)
    
    offset[:, 0] = (location[0] - anchor_center[:, 0]) / anchor_center[:, 2]
    offset[:, 1] = (location[1] - anchor_center[:, 1]) / anchor_center[:, 3]
    offset[:, 2] = torch.log(location[2] / anchor_center[:, 2])
    offset[:, 3] = torch.log(location[3] / anchor_center[:, 3])

    return offset

def decode(offset, anchor_center):
    """
    Params:
        offset:         {tensor(4), double} xc, yc,  w,  h
        anchor_center:  {tensor(A, 4)}      xc, yc,  w,  h
    Returns:
        location:       {tensor(A, 4), double} xc, yc,  w,  h
    Notes:
        bxc = oxc * w + axc
        byc = oyc * h + ayc
        bw = exp(ow) * aw
        bh = exp(oh) * ah
    """
    location = torch.zeros_like(anchor_center, dtype=torch.float)
    
    location[:, 0] = offset[0] * anchor_center[:, 2] + anchor_center[:, 0]
    location[:, 1] = offset[1] * anchor_center[:, 3] + anchor_center[:, 1]
    location[:, 2] = torch.exp(offset)[2] * anchor_center[:, 2]
    location[:, 3] = torch.exp(offset)[3] * anchor_center[:, 3]

    return location

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def show_bbox(im, bbox, winname="", waitkey=0):
    """
    Params:
        im: {ndarray(H, W, 3)}
        bbox: {ndarray(N, 4)} x1, y1, x2, y2
    """
    image = im.copy()
    bbox = bbox.reshape(-1, 4).astype(np.int)
    for x1, y1, x2, y2 in bbox:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imshow(winname, image)
    cv2.waitKey(waitkey)

def crop_square_according_to_bbox(im, bbox, size=None, pad=None):
    """
    Params:
        image: {ndarray(H, W, C)}
        bbox:  {ndarray(4)}
    Returns:
        im: {ndarray(size, size, C)}
    """
    xc, yc, w, h = corner2center(bbox)
    p = pad(w, h) if pad is not None else 0
    a = int(np.sqrt((w + p) * (h + p)))
    x1 = xc - a // 2; y1 = yc - a // 2; x2 = x1 + a; y2 = y1 + a

    bbox[[0, 2]] -= x1; bbox[[1, 3]] -= y1
    xc_, yc_, w_, h_ = corner2center(bbox)
    
    M = np.float32([
            [1, 0, xc_ - xc], 
            [0, 1, yc_ - yc]
        ])

    padval = im.mean(0).mean(0)
    im = cv2.warpAffine(im, M, (a, a), borderMode=cv2.BORDER_CONSTANT, borderValue=padval)
    im = cv2.resize(im, (size, size)) if size is not None else im
    
    return im