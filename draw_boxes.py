from convert.converter import Converter
from convert.data.data_info import TargetMapInfo

import cv2
from dataset.barcode import BCDataset, BCDatasetValidSyntetic
from torch.utils.data import DataLoader
from backboned_unet import Unet
import torch
import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms

from catalyst.utils.config import _load_ordered_yaml as load_ordered_yaml
from convert.utils import process_class_config

path = "/media/artem/A2F4DEB0F4DE85C7/runs/presentation/original+200/0.pth"
ckp = torch.load(path)
cfg = ckp['config']

model = Unet(backbone_name=cfg.slug, classes=cfg.num_classes)

model.load_state_dict(ckp['model'])
model = model.eval()

if cfg.gpu:
    model = model.cuda()

path = "/media/artem/A2F4DEB0F4DE85C7/runs/presentation/f1+100+v1+pretrainCenterNet/0.pth"
ckp = torch.load(path)
cfg = ckp['config']

model_second = Unet(backbone_name=cfg.slug, classes=cfg.num_classes)

model_second.load_state_dict(ckp['model'])
model_second = model_second.eval()

if cfg.gpu:
    model_second = model_second.cuda()

mode = "eval"
ds = BCDatasetValidSyntetic('/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-synth-512/', resize_size=(512, 512),
                    mode=mode, classes_name=cfg.CLASSES_NAME)
dl = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)

plot_data = []

class_config_path = "/home/artem/PycharmProjects/backboned-unet-master/base.yml"
with open(class_config_path, "r") as fin:
    class_config = load_ordered_yaml(fin)

# set default values
class_config = process_class_config(class_config["class_config"])
target_map_info = TargetMapInfo(class_config)

def show_img(img, boxes, clses, scores, name):
    boxes = np.array(boxes, np.int32)
    print(boxes)
    print("_______________")
    for box in boxes:
        img = cv2.polylines(img, [box], True, (255, 0, 0), 2)

    cv2.imshow("ref_" + name, img)
    cv2.waitKey(0)

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

def processModel(model1, inputs, detection_pixel_threshold, img_paded, name):
    masks, _ = model1(inputs)
    pred_map = torch.sigmoid(masks)

    numpy_masks = pred_map.detach().cpu().numpy()

    converter = Converter(TargetMapInfo())
    converter.detection_pixel_threshold = detection_pixel_threshold
    detected_objects = converter.postprocess_target_map(1 - numpy_masks)
    print(np.max(1 - numpy_masks))
    print(detected_objects)

    barcodesBoxes = []
    cls = []
    score = []
    for obj in detected_objects[0]:
        box = []
        for val in obj.location:
            box.append([ int(val[0]), int(val[1]) ])
        barcodesBoxes.append(box)
        cls.append(obj.class_name)
        score.append(1)

    show_img( img_paded, barcodesBoxes, cls, score, name )
    #for i, mask in enumerate(pred_map[0]):
    #    cv2.imshow(cfg.CLASSES_NAME[i] + "_" + name, mask.detach().cpu().numpy() * 1.)



for data in tqdm.tqdm(dl):
    img, _ = data
    ipath = ds.ipath

    img = Image.open(ipath).convert('RGB')
    img_paded, info = preprocess_img(img, cfg.resize_size)

    imgs = [img]
    infos = [info]

    input = transforms.ToTensor()(img_paded)
    input = transforms.Normalize(std=cfg.std, mean=cfg.mean)(input)
    inputs = input.unsqueeze(0).cuda()

    processModel(model_second, inputs, 0.000813850038109756, img_paded.copy(), "Dice")
    processModel(model, inputs, 0.13827078683035715, img_paded.copy(), "original+200")



    cv2.waitKey()