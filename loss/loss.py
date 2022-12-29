import torch
from torch import nn
from loss.losses import *
from loss.utils import *
import numpy as np
from backboned_unet.config import Config as cfg
import queue

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.down_stride = cfg.down_stride

        self.Center_loss = CenterLoss(num_classes=len(cfg.CLASSES_NAME), feat_dim=256)
        self.focal_loss = modified_focal_loss
        self.iou_loss = DIOULoss
        self.l1_loss = F.l1_loss
        self.feature_loss = torch.nn.BCEWithLogitsLoss()
        self.alpha = cfg.loss_alpha
        self.beta = cfg.loss_beta
        self.gamma = cfg.loss_gamma


    def forward(self, pred, gt):
        pred_hm = pred
        imgs, gt_boxes, gt_classes, gt_hm, infos = gt
        cls_loss = self.focal_loss(pred_hm, gt_hm)

        binary_pred = pred.sigmoid()
        # print(binary_ann.shape, binary_pred.shape)

        binary_ann = gt_hm
        tp = binary_ann * binary_pred

        fp = binary_pred * (1 - binary_ann)
        fn = binary_ann * (1 - binary_pred)
        # tn = (1 - binary_ann) * (1 - binary_pred)

        # soft_f1_class1 =  tp / (tp + (fp + fn) / 2 + 1e-8)
        # soft_f1_class0 = tn / (tn + (fp + fn) / 2 + 1e-8)

        b = cfg.f1Beta

        soft_f1_class1 = tp / (tp + (b ** 2 * fp + fn) / (b ** 2 + 1) + 1e-8)
        # soft_f1_class0 = tn / (tn + (b**2 * fp + fn) / (b**2 + 1) + 1e-8)

        cost_class1 = 1 - soft_f1_class1

        return cls_loss, torch.sum(cost_class1)
