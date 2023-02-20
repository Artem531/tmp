from convert.converter import Converter
from convert.data.data_info import TargetMapInfo
import os
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
from PrecRecF1Analize import getTpFpFn
from multiprocessing import Pool

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
    th_i, th, binary_th, converter, numpy_masks, refBoxes = work_data
    #print(th_i)
    # print(plot_data)
    # print(np.sum(numpy_masks, axis=(0,1)).shape)
    # cv2.imshow("wtf", np.sum(numpy_masks, axis=(0,1)) / np.max(np.sum(numpy_masks, axis=(0,1))))
    # print(np.mean(numpy_masks))
    # cv2.waitKey(0)
    #print(th_i)

    converter.detection_pixel_threshold = binary_th # фиксируем так как варироваться будет iou
    detected_objects = converter.postprocess_target_map(numpy_masks)
    predBoxes = []

    for i, detected_objects_i in enumerate(detected_objects):
        for obj in detected_objects_i:
            box = []
            for val in obj.location:
                box.append((int(val[0]), int(val[1])))
            predBoxes.append(box)

    tp, fp, fn = getTpFpFn(refBoxes, predBoxes, iou_th=th)
    #print(tp, fp, fn, len(refBoxes), len(predBoxes), "!!!!")
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
    #print(plot_data)

#
# th_s = [0.005784254807692308,
#         0.0006510416666666666,
#         0.13827078683035715,
#         0.13020241477272726,
#         0.1375732421875,
#         0.0008138500381097561,
#         0.16282894736842105,
#         0.0008951822916666666,
#         0.396484375,
#         0.3752170138888889]
#

paths = ["/home/artem/PycharmProjects/backboned-unet-new/ckp/OrigDataset/PatternsMulticlass+focal"]

base_path = "/home/artem/PycharmProjects/backboned-unet-new/plots/full_dataset/"
plots_num = 8

def gather_data(path):
    global min_rec_infer
    data_plot2 = np.load(path)

    tp = data_plot2[:, 0]
    fp = data_plot2[:, 1]
    fn = data_plot2[:, 2]
    th = data_plot2[:, 3]

    tp_arr = np.array(tp)
    fp_arr = np.array(fp)
    fn_arr = np.array(fn)
    th_arr = np.array(th)

    prec = tp_arr / (tp_arr + fp_arr)

    lower_b = 0.78
    upper_b = 0.82

    return np.mean(th_arr[(prec > lower_b) & (prec < upper_b)])

base_th_path = "/home/artem/PycharmProjects/backboned-unet-new/plots/full_dataset/"

if __name__ == '__main__':
    for configuration_id in range(len(paths)):
        base_path = paths[configuration_id]
        name = base_path.split("/")[-1]

        for network_id in range(0, 9):
            binary_th = gather_data(base_th_path + name + f"/{network_id}.npy")
            print("use:", base_path, binary_th)
            path = base_path + f"/{str(network_id)}.pth"
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
            for num, data in tqdm.tqdm(enumerate(dl)):
                inputs, refBoxes = prepareBoxes(data)
                masks, _ = model(inputs.cuda())
                pred_map = torch.sigmoid(masks)

                numpy_masks = np.max(pred_map[:, 1:].detach().cpu().numpy(), axis=1)[np.newaxis]

                for th_i, th in enumerate( np.linspace(0.0, 1, num_th) ):
                    work_data.append([th_i, th, binary_th, Converter(TargetMapInfo()), numpy_masks, refBoxes])

                if num % 400 == 0:
                    pool_handler(work_data, plot_data)  # instantiating without any argument
                    work_data = []

            if len(work_data) != 0:
                pool_handler(work_data, plot_data)  # instantiating without any argument
                work_data = []

            b = fbeta
            print(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2], th)

            plot_data = np.array(plot_data)

            base_save_path = "/home/artem/PycharmProjects/backboned-unet-new/plots/"
            name = base_path.split("/")[-1]

            try:
                os.mkdir(base_save_path + name)
            except OSError as error:
                print(error)

            np.save(base_save_path + name + f"/{str(network_id)}.npy", plot_data)
