from convert.converter import Converter
from convert.data.data_info import TargetMapInfo

import cv2
from dataset.barcode import BCDatasetValidSyntetic
from torch.utils.data import DataLoader
from backboned_unet import Unet
import torch
import tqdm
from PIL import Image
from metrics import FtMetricsCalculator
from torchvision import transforms
import numpy as np
from catalyst.utils.config import _load_ordered_yaml as load_ordered_yaml
from convert.utils import process_class_config
from backboned_unet.config import Config as cfg
from PrecRecF1Analize import getTpFpFnMasks
from multiprocessing import Pool
from PIL import Image, ImageDraw
import os
from multiprocessing import get_context

class_config_path = "/home/artem/PycharmProjects/backboned-unet-master/base.yml"
with open(class_config_path, "r") as fin:
    class_config = load_ordered_yaml(fin)

# set default values
class_config = process_class_config(class_config["class_config"])
target_map_info = TargetMapInfo(class_config)

fbeta = 1

mode = "eval"
ds = BCDatasetValidSyntetic('/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-synth-512/', resize_size=(512, 512),
                    mode=mode, classes_name=cfg.CLASSES_NAME)
dl = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)
#print(len(dl))
plot_data = []



def show_img(img, boxes, clses, scores):
    boxes = np.array(boxes, np.int32)
    for box in boxes:
        img = cv2.polylines(img, [box], True, (255, 0, 0), 2)

    cv2.imshow("ref", img)
    cv2.waitKey(0)


def preprocess_img(image, input_ksize, boxes=None):
    '''
    resize image and bboxes
    Returns
    image_paded: input_ksize
    bboxes: [None,4]
    '''
    min_side, max_side = input_ksize
    h, w, _ = image.shape

    smallest_side = min(w, h)
    largest_side = max(w, h)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    nw, nh = int(scale * w), int(scale * h)
    if scale != 1:
        image_resized = cv2.resize(image, (nw, nh))
    else:
        image_resized = image

    pad_w = max_side - nw
    pad_h = max_side - nh

    image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized

    if boxes is None:
        return image_paded
    else:
        boxes[:, :, 0] = boxes[:, :, 0] * scale
        boxes[:, :, 1] = boxes[:, :, 1] * scale
        return image_paded, boxes


num = 0


def prepareBoxes(data):
    img, refBoxes = data
    orig_refBoxes = list(refBoxes)[0]
    ipath = ds.ipath

    img = Image.open(ipath).convert('RGB')
    img = np.array(img)
    refBoxes = np.array(refBoxes)
    img_paded, info = preprocess_img(img, cfg.resize_size, refBoxes)

    input = transforms.ToTensor()(img_paded)
    input = transforms.Normalize(std=cfg.std, mean=cfg.mean)(input)
    inputs = input.unsqueeze(0)
    return inputs.cuda(), orig_refBoxes


def calTpFpFn(work_data):
    th_i, th, refMask, predMask = work_data
    # cv2.imshow("0", predMask)
    # cv2.waitKey(0)
    predMask = predMask > th

    tp, fp, fn = getTpFpFnMasks(refMask, predMask)
    # print(tp, fp, fn, len(refBoxes), len(predBoxes), "!!!!")
    return [tp, fp, fn, th, th_i]


def pool_handler(work, plot_data):
    with get_context("spawn").Pool(7) as p:
        res = p.map(calTpFpFn, work)
        p.close()

    for res_iter_list in res:
        tp, fp, fn, th, th_i = res_iter_list
        plot_data[th_i, 0] += tp
        plot_data[th_i, 1] += fp
        plot_data[th_i, 2] += fn
        plot_data[th_i, 3] = th
    # print(plot_data)


paths = ["/home/artem/PycharmProjects/backboned-unet-new/ckp/OrigDataset/PatternsMulticlass+focal"]

save_cfg = cfg.CLASSES_NAME

if __name__ == '__main__':
    for base_model_path in paths:
        model_name = base_model_path.split("/")[-1]

        if base_model_path == "models/PatternsModels+FocalLoss" or "models/PatternsModels+FocalLoss+200":
            print("use!!!!")
            cfg.CLASSES_NAME = ('Empty', 'EAN', 'QRCode', 'Postnet',
                                'DataMatrix', 'PDF417', 'Aztec',
                                '{"barcode":"ean_pattern"}',
                                '{"barcode":"qr_pattern"}',
                                '{"barcode":"aztec_pattern"}',
                                '{"barcode":"pdf417_pattern"}',
                                '{"barcode":"post_pattern"}',
                                '{"barcode":"datamatrix_pattern"}')
            cfg.num_classes = len(cfg.CLASSES_NAME)
        else:
            cfg.CLASSES_NAME = save_cfg
            cfg.num_classes = len(cfg.CLASSES_NAME)

        for network_id in range(0, 5):
            path = base_model_path + f"/{str(network_id)}.pth"
            print(path)
            ckp = torch.load(path)
            model = Unet(backbone_name=cfg.slug, classes=cfg.num_classes, encoder_freeze=cfg.freeze_backbone)

            model.load_state_dict(ckp['model'])
            model = model.eval()

            if cfg.gpu:
                model = model.cuda()

            num_statistic = 4
            num_th = 20
            plot_data = np.zeros((num_th, num_statistic))

            work_data = []
            stop_flag = 0
            for th_i in range(num_th):
                if th_i == 0:
                    th = 0.5
                    min_bound = 0
                    max_bound = 1
                else:
                    tp = plot_data[th_i - 1][0]
                    fp = plot_data[th_i - 1][1]
                    print(tp, fp)
                    prec = tp / (tp + fp)
                    print("prec", prec, th)
                    if prec > 0.8:
                        max_bound = th
                        th = th - (abs(max_bound - min_bound) / 2)
                    else:
                        min_bound = th
                        th = th + (abs(max_bound - min_bound) / 2)

                    if abs(prec - 0.8) < 0.01:
                        stop_flag = stop_flag + 1
                        if stop_flag == 2:
                            print(stop_flag, "times ~0.8 stoping!")
                            break

                for num, data in tqdm.tqdm(enumerate(dl)):
                    inputs, refBoxes = prepareBoxes(data)

                    width, height = inputs.shape[2], inputs.shape[3]
                    refMask = Image.new('L', (width, height), 0)

                    for polygon in refBoxes:
                        ImageDraw.Draw(refMask).polygon(polygon, outline=0, fill=1)

                    refMask = np.array(refMask)

                    masks, _ = model(inputs.cuda())
                    pred_map = torch.sigmoid(masks)

                    # for i, mask in enumerate(pred_map[0]):
                    #     cv2.imshow(cfg.CLASSES_NAME[i] + "_", mask.detach().cpu().numpy() * 1.)
                    #     cv2.waitKey()

                    numpy_masks = np.max(pred_map[:, 1:].detach().cpu().numpy(), axis=1)[np.newaxis]

                    work_data.append([th_i, th, refMask, numpy_masks])

                    if num % 500 == 0:
                        pool_handler(work_data, plot_data)  # instantiating without any argument
                        work_data = []

                if len(work_data) != 0:
                    pool_handler(work_data, plot_data)  # instantiating without any argument
                    work_data = []

            b = fbeta
            print(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2], th)

            plot_data = np.array(plot_data)

            base = "plots/full_dataset/" + model_name

            try:
                os.mkdir(base)
            except OSError as error:
                print(error)

            np.save(base + f"/{str(network_id)}.npy", plot_data)
