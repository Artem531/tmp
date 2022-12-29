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
from sklearn.metrics import PrecisionRecallDisplay

class_config_path = "/home/artem/PycharmProjects/backboned-unet-master/base.yml"
with open(class_config_path, "r") as fin:
    class_config = load_ordered_yaml(fin)

# set default values
class_config = process_class_config(class_config["class_config"])
target_map_info = TargetMapInfo(class_config)
converter = Converter(TargetMapInfo())

for i in range(10):
    path = f"/home/artem/PycharmProjects/backboned-unet-new/ckp/f1+centerf1+v1+30/{str(i)}.pth"
    print(path)
    ckp = torch.load(path)
    cfg = ckp['config']

    model = Unet(backbone_name=cfg.slug, classes=cfg.num_classes, encoder_freeze=cfg.freeze_backbone)

    model.load_state_dict(ckp['model'])
    model = model.eval()

    if cfg.gpu:
        model = model.cuda()

    mode = "eval"
    ds = BCDatasetValid('/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-real-512_orig/', resize_size=(512, 512), mode=mode, classes_name=cfg.CLASSES_NAME)
    dl = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)

    plot_data = []

    def show_img(img, boxes, clses, scores):
        boxes = np.array(boxes, np.int32)
        for box in boxes:
            img = cv2.polylines(img, [box], True, (255, 0, 0), 2)

        cv2.imshow("ref", img)
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

    for th in list(np.linspace(0., 0.01, 20)) + list(np.linspace(0.01, 0.99, 10)):
        prec = 0
        rec = 0
        num = 0

        for data in tqdm.tqdm(dl):
            num += 1
            img, refBoxes = data
            ipath = ds.ipath

            img = Image.open(ipath).convert('RGB')
            img_paded, info = preprocess_img(img, cfg.resize_size)

            imgs = [img]
            infos = [info]

            input = transforms.ToTensor()(img_paded)
            input = transforms.Normalize(std=cfg.std, mean=cfg.mean)(input)
            inputs = input.unsqueeze(0).cuda()

            masks, _ = model(inputs)
            pred_map = torch.sigmoid(masks)
            numpy_masks = pred_map.detach().cpu().numpy()

            converter.detection_pixel_threshold = th
            detected_objects = converter.postprocess_target_map(1 - numpy_masks)
            #detected_objects = converter.postprocess_target_map(numpy_masks)
            #print(detected_objects)

            barcodesBoxes = []
            cls = []
            score = []
            for obj in detected_objects[0]:
                box = []
                for val in obj.location:
                    box.append(( int(val[0]), int(val[1])))
                barcodesBoxes.append(box)
                cls.append(obj.class_name)
                score.append(1)

            #show_img( img_paded, barcodesBoxes, cls, score )

            if len(barcodesBoxes) != 0:
                calc = FtMetricsCalculator(refBoxes[0], barcodesBoxes)
                metrics = calc.analyze(0.5)

                prec += metrics.average_precision_by_area
                rec += metrics.average_recall_by_area
            else:
                prec += 1
                rec += 0
        plot_data.append([prec / num, rec / num])

    print(prec / num, rec / num, th)
    plot_data.append([prec / num, rec / num])

    print(plot_data)

    plot_data = np.array(plot_data)

    base = "/home/artem/PycharmProjects/backboned-unet-new"
    np.save(base + f"/plots/{str(i)}.npy", plot_data)

    # import matplotlib.pyplot as plt
    # precision = np.around(plot_data[:, 0], decimals=3)
    # recall = np.around(plot_data[:, 1], decimals=3)
    # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    # disp.plot()
    # plt.show()
