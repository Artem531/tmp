import cv2
import numpy as np
from shapely.geometry import Polygon
import torch
from scipy.optimize import linear_sum_assignment

def calcIoU(refBox, predBox):
    polygon1 = Polygon(refBox)
    polygon2 = Polygon(predBox)
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersect / union

    return iou

def getTpFpFnMasks(binary_ann, binary_pred):
    """
    ref and pred boxes for 1 class
    """
    #cv2.imshow("1", binary_pred * 1.)
    #cv2.imshow("2", binary_ann * 1.)
    #cv2.waitKey(0)

    tp = np.sum(binary_ann * binary_pred)

    fp = np.sum(binary_pred * (1 - binary_ann))
    fn = np.sum(binary_ann * (1 - binary_pred))

    return tp, fp, fn

def getTpFpFn(gt_boxes, pred_boxes, iou_threshold):
    num_pred_boxes = len(pred_boxes)
    num_gt_boxes = len(gt_boxes)

    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, refBox in enumerate(gt_boxes):
        for j, predBox in enumerate(pred_boxes):
            iou_matrix[i, j] = calcIoU(refBox, predBox)
    #print(iou_matrix)
    # Use the Hungarian algorithm to find the optimal set of matches
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    #print(row_ind, col_ind)
    # Count true positives, false positives, and false negatives
    tp = 0
    fp = 0
    for i in range(num_pred_boxes):
        if i in row_ind:
            j = col_ind[np.where(row_ind == i)[0][0]]
            if iou_matrix[i, j] >= iou_threshold:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    fn = num_gt_boxes - tp

    return tp, fp, fn