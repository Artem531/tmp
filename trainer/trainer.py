import os, logging

import numpy as np
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

writer_train = SummaryWriter('runs/experiment_1/train')
writer_test = SummaryWriter('runs/experiment_1/eval')

from trainer.lr_scheduler import *

from CenterNet.dataset.barcode import BCDataset as CenterNetDataset
from backboned_unet.config import Config as cfg

train_ds_CN = CenterNetDataset(cfg.root, mode='train', resize_size=cfg.resize_size, classes_name=cfg.CLASSES_NAME, down_stride=1)
eval_ds_CN = CenterNetDataset(cfg.root, mode="valid", resize_size=cfg.resize_size, classes_name=cfg.CLASSES_NAME, down_stride=1)

try:
    import apex
    APEX = True
except:
    APEX = False

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
    beta = 100
    a = np.exp(beta * (listOfLossComponentsChangeRates - max_norm_loss))

    # denominator
    total = np.sum(a)

    # params
    params = a / (total + 1e-8)
    return params


class Trainer(object):
    def __init__(self, config, model, centerNetModelHead, loss_func, centerNetLoss_func, train_loader, val_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_net = model
        self.centerNetHead = centerNetModelHead

        self.losser = loss_func
        self.centerNetLosser = centerNetLoss_func

        self.model_idx = None

        if config.resume:
            self.resume_model()
        else:
            self.optimizer = torch.optim.Adam(list(self.train_net.parameters()) + list(self.centerNetHead.parameters()),
                                               lr=config.lr,
                                               weight_decay=1e-4,
                                               amsgrad=config.AMSGRAD)
            self.lr_schedule = WarmupMultiStepLR(self.optimizer, config.steps, config.gamma,
                                                 warmup_iters=config.warmup_iters)
            self.start_step = 1
            self.best_loss = 1e6

        if config.gpu:
            self.train_net = self.train_net.cuda()
            self.losser = self.losser.cuda()

        if config.apex and APEX:
            self.train_net, self.optimizer = apex.amp.initialize(self.train_net, self.optimizer, opt_level="O1",
                                                                 verbosity=0)
            self.lr_schedule = WarmupMultiStepLR(self.optimizer, config.steps, config.gamma,
                                                 warmup_iters=config.warmup_iters)

        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)

        self.logger = self.init_logger()
        self.logger.info('Trainer OK!')

        self.logger.info(config)

        self.start_adapt_iter = 51
        self.n = 50
        self.loss_iters = 0

        self.prev_loss_components = []

    def init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        handler = logging.FileHandler(os.path.join(self.config.log_dir, "log.txt"))
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
        return logger

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def write_log(self, loss, mode='TRAIN'):
        log = f'[{mode}]TOTAL_STEP: %6d/{self.config.max_iter}' % (self.start_step)
        if loss is not None: log += f'  seg loss: %.3f' % (loss)

        log += f'  lr: %.6f' % (self.get_lr())

        if loss is not None:
            if "eval" in mode:
                writer_test.add_scalar('seg loss', loss, (self.start_step))
            else:
                writer_train.add_scalar('seg loss', loss, (self.start_step))

        self.logger.info(log)

    def train(self):
        self.logger.info('Start trainning...\n')
        self.save_model(False, self.model_idx)
        loss = self.val_one_epoch()
        while self.start_step < self.config.max_iter:
            loss = self.train_one_epoch()
            if self.config.eval:
                loss = self.val_one_epoch()

            self.save_model(loss < self.best_loss)
            self.best_loss = min(self.best_loss, loss)
        self.save_model(False, self.model_idx)

    def getCenterNetLoss(self, gt, UnetMapsFeatures):
        if self.config.gpu:
            gt = [i.cuda() if isinstance(i, torch.Tensor) else i for i in gt]

        pred = self.centerNetHead(UnetMapsFeatures)
        losses = self.centerNetLosser(pred, gt)
        return losses

    def train_one_epoch(self):
        self.train_net.train()
        total_loss = 0.

        for step, (gtUnet) in enumerate(self.train_loader):
            imgs, ann, sample_indexes = gtUnet

            if self.config.gpu:
                imgs, ann = imgs.cuda(), ann.cuda()

            self.optimizer.zero_grad()
            pred, pred_prev = self.train_net(imgs)

            #gtCenterNetHead = train_ds_CN.collate_fn([train_ds_CN.__getitem__(idx) for idx in sample_indexes ])
            #centerNetLoss, centerNetF1Loss = self.getCenterNetLoss(gtCenterNetHead, pred_prev)

            #UnetLoss = self.losser(y_true=ann.long(), y_pred=pred)

            binary_ann = torch.zeros_like(pred)
            for c in range(pred.shape[1]):
                binary_ann[:, c, :, :] = torch.where(ann == c, 1, 0)
                #cv2.imshow("test", (ann.cpu().numpy()[0] == c) * 1.)
                #cv2.waitKey(0)

            binary_pred = pred.sigmoid()
            #print(binary_ann.shape, binary_pred.shape)

            tp = binary_ann * binary_pred

            fp = binary_pred * (1 - binary_ann)
            fn = binary_ann * (1 - binary_pred)
            tn = (1 - binary_ann) * (1 - binary_pred)

            #soft_f1_class1 =  tp / (tp + (fp + fn) / 2 + 1e-8)
            #soft_f1_class0 = tn / (tn + (fp + fn) / 2 + 1e-8)

            b = self.config.f1Beta

            soft_f1_class1 = tp / (tp + (b**2 * fp + fn) / (b**2 + 1) + 1e-8)
            soft_f1_class0 = tn / (tn + (b**2 * fp + fn) / (b**2 + 1) + 1e-8)


            cost_class1 = 1 - soft_f1_class1
            cost_class0 = 1 - soft_f1_class0

            #print(tp.item(), fp.item(), fn.item())
            #unetF1Score = torch.sum((cost_class0 + cost_class1) / 2)
            #unetF1Score = torch.sum(cost_class1)

            loss_components = [torch.sum(cost_class0), torch.sum(cost_class1)]
            if self.loss_iters > self.start_adapt_iter:
                rate = calcRateOfChange(np.mean(self.prev_loss_components, axis=0), np.array([loss_i.item() for loss_i in loss_components]))
                norm_rate, max_rate = calLossRateNormalization(rate)
                params = calcLossAdaptationCoef(norm_rate, max_rate)
                assert(len(params) == len(loss_components))

                print(params)
                print(loss_components)
                print(norm_rate)
                loss = loss_components[0] * params[0] + loss_components[1] * params[1]
            else:
                #print(unetF1Score.item(), centerNetF1Loss.item(), centerNetLoss.item(), UnetLoss.item())
                #loss = (UnetLoss - unetF1Score + centerNetLoss - centerNetF1Loss) / 4
                #print(unetF1Score.item(), UnetLoss.item())
                loss = (torch.sum(cost_class0) + torch.sum(cost_class1)) / 2

            self.prev_loss_components.append( np.array([loss_i.item() for loss_i in loss_components] ) )
            if len(self.prev_loss_components) > self.n:
                self.prev_loss_components.pop(0)
            self.loss_iters += 1

            if self.config.apex and APEX:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            self.lr_schedule.step()

            total_loss += loss.item()
            self.start_step += 1

            if step % self.config.log_interval == 0:
                self.write_log(loss=total_loss / (step + 1))

        return total_loss / (step + 1)

    @torch.no_grad()
    def val_one_epoch(self):
        self.train_net.eval()
        total_loss = 0

        with torch.no_grad():
            for step, (gt) in enumerate(self.val_loader):
                imgs, ann, sample_indexes = gt
                if self.config.gpu:
                    imgs, ann = imgs.cuda(), ann.cuda()

                pred, _ = self.train_net(imgs)
                loss = self.losser(y_true=ann.long(), y_pred=pred)

                total_loss += loss.item()

        self.write_log(total_loss / (step + 1),  mode='eval')
        return total_loss / (step + 1)

    def save_model(self, is_best=False, model_idx=None):
        state = {
            'model': self.train_net.state_dict(),
            'step': self.start_step,
            'optimizer': self.optimizer,
            'lr_schedule': self.lr_schedule,
            'loss': self.best_loss,
            'config': self.config
        }
        if is_best:
            torch.save(state, os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth'))
        torch.save(state, os.path.join(self.config.checkpoint_dir, 'checkpoint.pth'))
        if model_idx != None:
            torch.save(state, os.path.join(self.config.checkpoint_dir, str(model_idx) + ".pth"))

    def resume_model(self):
        path = os.path.join(self.config.checkpoint_dir, 'checkpoint.pth')
        ckp = torch.load(path)
        model_static_dict = ckp['model']
        self.optimizer = ckp['optimizer']
        self.lr_schedule = ckp['lr_schedule']
        self.start_step = 0
        self.best_loss = ckp['loss']
        self.train_net.load_state_dict(model_static_dict)