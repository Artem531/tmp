import cv2
from dataset.barcode import BCDataset
from backboned_unet.config import Config as cfg
from torch.utils.data import DataLoader
from backboned_unet import Unet
import torch
from trainer.trainer import Trainer
import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms

path = "/ckp/resnet18Features/best_checkpoint.pth"
ckp = torch.load(path)
cfg = ckp['config']

model = Unet(backbone_name=cfg.slug, classes=cfg.num_classes, encoder_freeze=cfg.freeze_bn)

model.load_state_dict(ckp['model'])
model = model.eval()

if cfg.gpu:
    model = model.cuda()

mode = "eval"
ds = BCDataset('/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-real-512_orig/', resize_size=(512, 512), mode=mode, classes_name=cfg.CLASSES_NAME)
dl = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)

plot_data = []

def preprocess_img(img, input_ksize):
    min_side, max_side = input_ksize
    h, w = img.height, img.width
    _pad = 32
    smallest_side = min(w, h)
    largest_side = max(w, h)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    nw, nh = int(scale * w), int(scale * h)
    img_resized = np.array(img.resize((nw, nh)))

    pad_w = _pad - nw % _pad
    pad_h = _pad - nh % _pad

    img_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    img_paded[:nh, :nw, :] = img_resized

    return img_paded, {'raw_height': h, 'raw_width': w}

for data in tqdm.tqdm(dl):
    img, ann = data
    ipath = ds.ipath

    img = Image.open(ipath).convert('RGB')
    img_paded, info = preprocess_img(img, cfg.resize_size)

    imgs = [img]
    infos = [info]

    input = transforms.ToTensor()(img_paded)
    input = transforms.Normalize(std=cfg.std, mean=cfg.mean)(input)
    inputs = input.unsqueeze(0).cuda()

    masks = model(inputs)

    cv2.imshow("img", img_paded)
    for i, mask in enumerate(masks[0]):
        cv2.imshow(cfg.CLASSES_NAME[i], mask.detach().cpu().numpy())

    ann = ann[0].detach().cpu().numpy() > 0
    ann = ann / 1.
    cv2.imshow("ann", ann)
    cv2.waitKey()