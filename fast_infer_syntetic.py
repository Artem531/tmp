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


class_config_path = "/home/artem/PycharmProjects/backboned-unet-master/base.yml"
with open(class_config_path, "r") as fin:
    class_config = load_ordered_yaml(fin)

# set default values
class_config = process_class_config(class_config["class_config"])
target_map_info = TargetMapInfo(class_config)
converter = Converter(TargetMapInfo())

fbeta = 1

mode = "eval"
ds = BCDatasetValidSyntetic('/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-synth-512/', resize_size=(512, 512),
                    mode=mode, classes_name=cfg.CLASSES_NAME)
dl = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)

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

    ipath = ds.ipath

    img = Image.open(ipath).convert('RGB')
    img = np.array(img)
    refBoxes = np.array(refBoxes)
    img_paded, info = preprocess_img(img, cfg.resize_size, refBoxes)


    imgs = [img]
    infos = [info]

    input = transforms.ToTensor()(img_paded)
    input = transforms.Normalize(std=cfg.std, mean=cfg.mean)(input)
    inputs = input.unsqueeze(0)
    return inputs.cuda(), [refBoxes]

for network_id in range(0, 9):
    path = f"/media/artem/A2F4DEB0F4DE85C7/runs/presentation/f1+100/{str(network_id)}.pth"
    print(path)
    ckp = torch.load(path)
    model = Unet(backbone_name=cfg.slug, classes=cfg.num_classes, encoder_freeze=cfg.freeze_backbone)

    model.load_state_dict(ckp['model'])
    model = model.eval()

    if cfg.gpu:
        model = model.cuda()

    num_statistic = 6
    num_th = 25
    num_part_th = num_th // 2
    plot_data = np.zeros((num_th - 1, num_statistic))

    for num, data in tqdm.tqdm(enumerate(dl)):
        inputs, refBoxes = prepareBoxes(data)
        masks, _ = model(inputs.cuda())
        pred_map = torch.sigmoid(masks)

        numpy_masks = pred_map.detach().cpu().numpy()

        if num == 0:
            low_bound = np.percentile(numpy_masks, 80)
            print(low_bound)

        for th_i, th in enumerate( list(np.linspace(0.0000001, low_bound, num_part_th)) \
                  + list(np.linspace(low_bound, 0.99, num_part_th)) ):
            #print(plot_data)
            # print(np.sum(numpy_masks, axis=(0,1)).shape)
            # cv2.imshow("wtf", np.sum(numpy_masks, axis=(0,1)) / np.max(np.sum(numpy_masks, axis=(0,1))))
            # print(np.mean(numpy_masks))
            # cv2.waitKey(0)
            converter.detection_pixel_threshold = th
            detected_objects = converter.postprocess_target_map(1-numpy_masks)

            predBoxes = []
            cls = []
            score = []

            for i, detected_objects_i in enumerate(detected_objects):
                barcodesBoxes = []
                for obj in detected_objects_i:
                    box = []
                    for val in obj.location:
                        box.append(( int(val[0]), int(val[1])))
                    barcodesBoxes.append(box)
                    cls.append(obj.class_name)
                    score.append(1)
                predBoxes.append(barcodesBoxes)

            #show_img( img_paded, barcodesBoxes, cls, score )

            if len(predBoxes[0]) == 0 or len(detected_objects) == 0:
                plot_data[th_i, 0] += 1
                plot_data[th_i, 1] += 0

                plot_data[th_i, 4] += refBoxes[0].shape[1]  # false negatives
                plot_data[th_i, 5] = th
                continue

            #print(refBoxes[0][0], predBoxes[0])
            #print(refBoxes[0][0], predBoxes[0])
            calc = FtMetricsCalculator(refBoxes[0][0], predBoxes[0])
            metrics = calc.analyze(0.5)

            plot_data[th_i, 0] += metrics.tp / (metrics.tp + metrics.fp)#metrics.average_precision_by_area
            plot_data[th_i, 1] += metrics.tp / (metrics.tp + metrics.fn)#metrics.average_recall_by_area
            print(metrics.tp, metrics.fp, metrics.fn, th)
            plot_data[th_i, 3] += metrics.tp   # true positives
            plot_data[th_i, 2] += metrics.fp   # false positives
            plot_data[th_i, 4] += metrics.fn   # false negatives
            plot_data[th_i, 5] = th

    b = fbeta

    plot_data[:, 0] = plot_data[:, 0] / len(dl)
    plot_data[:, 1] = plot_data[:, 1] / len(dl)

    print(plot_data[:, 0], plot_data[:, 1], th)

    plot_data = np.array(plot_data)

    base = "/home/artem/PycharmProjects/backboned-unet-new"
    np.save(base + f"/plots/{str(network_id)}.npy", plot_data)
