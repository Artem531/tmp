import cv2
import numpy as np
from shapely.geometry import Polygon
import torch

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

def getTpFpFn(refBoxes, predBoxes, iou_th=0.5):
    """
    ref and pred boxes for 1 class
    """
    IoUMatrix = np.zeros((len(refBoxes), len(predBoxes)))
    for i, refBox in enumerate(refBoxes):
        for j, predBox in enumerate(predBoxes):
            IoUMatrix[i, j] = calcIoU(refBox, predBox)

    IoUMatrixBinary = IoUMatrix > iou_th
    IoUMatrixBinary_sum = np.sum(IoUMatrixBinary, axis=1)
    IoUMatrixBinary_sum_tp_mask = IoUMatrixBinary_sum >= 1

    tp = np.sum(IoUMatrixBinary_sum_tp_mask)
    fn = len(refBoxes) - tp
    fp = len(predBoxes) - tp

    # for i, predIoULine in enumerate(IoUMatrixBinary):
    #     if any(predIoULine == 1):
    #         tp += 1
    #         fp += np.sum(predIoULine == 1) - 1
    #     else:
    #         fn += 1

    return tp, fp, fn