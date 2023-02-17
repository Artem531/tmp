from convert.converter import Converter
from convert.data.data_info import TargetMapInfo

import cv2
from dataset.barcode import BCDatasetValid
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
ds = BCDatasetValid('/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-real-512_orig/', resize_size=(512, 512),
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



allRefBoxes = []
allInputs = []
num = 0
for data in tqdm.tqdm(dl):
    num += 1
    img, refBoxes = data

    ipath = ds.ipath

    img = Image.open(ipath).convert('RGB')
    img = np.array(img)
    refBoxes = np.array(refBoxes)
    img_paded, info = preprocess_img(img, cfg.resize_size, refBoxes)
    allRefBoxes.extend([refBoxes])

    imgs = [img]
    infos = [info]

    input = transforms.ToTensor()(img_paded)
    input = transforms.Normalize(std=cfg.std, mean=cfg.mean)(input)
    inputs = input.unsqueeze(0).cuda()
    allInputs.append(inputs)


for network_id in range(10):
    path = f"/home/artem/PycharmProjects/backboned-unet-new/ckp/testNewFeatures/{str(network_id)}.pth"
    print(path)
    ckp = torch.load(path)
    plot_data = []
    model = Unet(backbone_name=cfg.slug, classes=cfg.num_classes, encoder_freeze=cfg.freeze_backbone)

    model.load_state_dict(ckp['model'])
    model = model.eval()

    if cfg.gpu:
        model = model.cuda()

    numpy_masks = None

    for num, inputs in tqdm.tqdm(enumerate(allInputs)):
        masks, _ = model(inputs)
        pred_map = torch.sigmoid(masks)
        if num == 0:
            numpy_masks = pred_map.detach().cpu().numpy()
        else:
            numpy_masks = np.concatenate((numpy_masks, pred_map.detach().cpu().numpy()), axis=0)


    for th in list(np.linspace(0.00, 0.3, 300)) + list(np.linspace(0.3, 0.99, 50)):
        converter.detection_pixel_threshold = th
        detected_objects = converter.postprocess_target_map(1 - numpy_masks)

        allBarcodesBoxes = []
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
            allBarcodesBoxes.append(barcodesBoxes)

        #show_img( img_paded, barcodesBoxes, cls, score )
        prec = 0
        rec = 0

        tp = 0  # true positives
        fp = 0  # false positives
        fn = 0  # false negatives

        for refBoxes, predBoxes in zip(allRefBoxes, allBarcodesBoxes):
            if len(predBoxes) == 0:
                prec += 1
                rec += 0

                fn += refBoxes.shape[1]  # false negatives
                continue
            print(refBoxes[0], predBoxes)
            calc = FtMetricsCalculator(refBoxes[0], predBoxes)
            metrics = calc.analyze(0.5)

            prec += metrics.average_precision_by_area
            rec += metrics.average_recall_by_area

            tp += metrics.tp   # true positives
            fp += metrics.fp   # false positives
            fn += metrics.fn   # false negatives

        b = fbeta
        prec = prec / len(allRefBoxes)
        rec = rec / len(allRefBoxes)
        plot_data.append([prec, rec, tp, fp, fn, th])

        print(prec, rec, th)
    print(plot_data)

    plot_data = np.array(plot_data)

    base = "/home/artem/PycharmProjects/backboned-unet-new"
    np.save(base + f"/plots/{str(network_id)}.npy", plot_data)
