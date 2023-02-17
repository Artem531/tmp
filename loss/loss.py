import torch
from torch import nn
from loss.losses import *
from loss.utils import *
import numpy as np
from backboned_unet.config import Config as cfg
import queue


def calLossRateNormalization( listOfLossComponentsRates ):
    """
    listOfLossComponentsRates : numpy
    """
    totalLossRate = np.sum(listOfLossComponentsRates)

    ns = listOfLossComponentsRates / (totalLossRate + 1e-8)
    max_norm_loss = np.max(ns)

    return ns, max_norm_loss

def calcRateOfChange(curLossComponents, prevLossComponents):
    print("loss curLossComponents", curLossComponents)
    print("loss prevLossComponents", prevLossComponents)
    rate = curLossComponents / prevLossComponents
    return rate#, np.max(rate)

def calcLossAdaptationCoef( listOfLossComponentsChangeRates, max_norm_loss ):
    listOfLossAdaptationCoef = []

    # Nominator
    beta = 10
    a = np.exp(beta * (listOfLossComponentsChangeRates - max_norm_loss))

    # denominator
    total = np.sum(a)

    # params
    params = a / (total + 1e-8)
    return params


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
    def __init__(self, cfg):
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


        self.start_adapt_iter = 100000000
        self.n = 50
        self.loss_iters = 0

        self.prev_loss_components = []

    def forward(self, pred, gt):
        pred_hm, pred_wh, pred_offset, pred_similarity = pred
        imgs, gt_boxes, gt_classes, gt_hm, infos = gt
        gt_nonpad_mask = gt_classes.gt(-0.5)

        #print('pred_hm: ', pred_hm.shape, '  gt_hm: ', gt_hm.shape)
        cls_loss = self.focal_loss(pred_hm, gt_hm) / pred_hm.shape[2] / pred_hm.shape[3]

        wh_loss = cls_loss.new_tensor(0.)
        offset_loss = cls_loss.new_tensor(0.)
        features_loss = cls_loss.new_tensor(0.)

        num = 0
        #print(gt_classes, "gt_classes")

        loss_components = []
        for batch in range(imgs.size(0)):
            ct = infos[batch]['ct'].cuda()
            ct_int = ct.long()
            num += len(ct_int)
            batch_pos_pred_wh = pred_wh[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)
            batch_pos_pred_offset = pred_offset[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)

            batch_boxes = gt_boxes[batch][gt_nonpad_mask[batch]]
            wh = torch.stack([
                batch_boxes[:, 2] - batch_boxes[:, 0],
                batch_boxes[:, 3] - batch_boxes[:, 1]
            ]).view(-1) / self.down_stride
            offset = (ct - ct_int.float()).T.contiguous().view(-1)
            #print(batch_pos_pred_wh.shape, wh.shape)
            wh_loss += self.l1_loss(batch_pos_pred_wh, wh, reduction='sum')
            #print(batch_pos_pred_offset.shape, offset.shape)
            offset_loss += self.l1_loss(batch_pos_pred_offset, offset, reduction='sum')

            #batch_x = x[batch, :, ct_int[:, 1], ct_int[:, 0]].swapaxes(0, 1)
            #batch_labels = gt_classes[batch][gt_nonpad_mask[batch]]

            #center_loss += self.Center_loss(batch_x, batch_labels)

            batch_features = infos[batch]['features'].cuda()
            batch_pred_features = pred_similarity[batch]

            # print(batch_features.shape)
            # features = torch.stack([
            #     batch_features[:, 2],
            #     batch_features[:, 3]
            # ])

            batch_features = batch_features[:, ct_int[:, 1], ct_int[:, 0]].view(-1)
            batch_pred_features = batch_pred_features[:, ct_int[:, 1], ct_int[:, 0]].view(-1)
            #print("__________________")
            for i in range(batch_features.shape[0]):
                #print(batch_features[i], batch_pred_features[i])
                features_loss += self.feature_loss(batch_pred_features[i], batch_features[i])

        regr_loss = wh_loss * self.beta + offset_loss * self.gamma

        loss_components.append(cls_loss)
        loss_components.append(offset_loss)
        loss_components.append(wh_loss)
        loss_components.append(features_loss)

        if self.loss_iters > self.start_adapt_iter:
            rate = calcRateOfChange(np.mean(self.prev_loss_components, axis=0),
                                    np.array([loss_i.item() for loss_i in loss_components]))
            norm_rate, max_rate = calLossRateNormalization(rate)
            params = calcLossAdaptationCoef(norm_rate, max_rate)
            assert (len(params) == len(loss_components))

            print(params)
            print(loss_components)
            print(norm_rate)
            returnArray = [ loss_components[0] * params[0], loss_components[1] * params[1],
                            loss_components[2] * params[2], loss_components[3] * params[3] ]
        else:
            # print(unetF1Score.item(), centerNetF1Loss.item(), centerNetLoss.item(), UnetLoss.item())
            # loss = (UnetLoss - unetF1Score + centerNetLoss - centerNetF1Loss) / 4
            # print(unetF1Score.item(), UnetLoss.item())
            returnArray = [cls_loss * self.alpha, regr_loss / (num + 1e-6), features_loss / (num + 1e-6)]

        self.prev_loss_components.append(np.array([loss_i.item() for loss_i in loss_components]))
        if len(self.prev_loss_components) > self.n:
            self.prev_loss_components.pop(0)
        self.loss_iters += 1
        # + center_loss / (num + 1e-6)
        return returnArray