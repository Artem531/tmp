from dataset.barcode import BCDataset, BCDatasetPatterns
from backboned_unet.config import Config as cfg
from torch.utils.data import DataLoader
from backboned_unet import Unet
import torch
from trainer.trainer import Trainer

from CenterNet.model.head import Head
from loss.loss import Loss

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":
    for i in range(0, 5):
        train_ds = BCDatasetPatterns(cfg.root, mode='train', resize_size=cfg.resize_size, classes_name=cfg.CLASSES_NAME,
                                     patterns_csv_path=cfg.patterns_markup)

        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=train_ds.collate_fn, pin_memory=True)

        eval_ds = BCDatasetPatterns(cfg.root, mode="valid", resize_size=cfg.resize_size, classes_name=cfg.CLASSES_NAME,
                                    patterns_csv_path=cfg.patterns_markup)

        eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=train_ds.collate_fn, pin_memory=True)

        net = Unet(backbone_name=cfg.slug, classes=cfg.num_classes, encoder_freeze=cfg.freeze_backbone)
        centerNetHead = Head(num_classes=cfg.num_classes, channel=cfg.head_channel, input_channel=16)

        if cfg.loadBackBone:
            net.backbone.load_state_dict(torch.load(cfg.backbone_checkpoint_dir))
            print("use new params")

        if cfg.init:
            ckp = torch.load(cfg.init_checkpoint_dir + f"{i}.pth")

            model_static_dict = ckp['model']
            model_static_dict["final_conv.weight"] = net.state_dict()["final_conv.weight"]
            model_static_dict["final_conv.bias"] = net.state_dict()["final_conv.bias"]

            net.load_state_dict(model_static_dict)
            print("resuming")

        if cfg.gpu:
            model = net.cuda()
            centerNetHead = centerNetHead.cuda()

        loss_func = cfg.loss_f
        CN_loss_func = Loss(cfg)

        epoch = 100
        cfg.max_iter = len(train_dl) * epoch
        cfg.steps = (int(cfg.max_iter * 0.6), int(cfg.max_iter * 0.8))

        trainer = Trainer(cfg, model, centerNetHead, loss_func, CN_loss_func, train_dl, eval_dl)
        trainer.model_idx = i;
        trainer.train()