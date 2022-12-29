import random

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from dataset.utils import gaussian_radius, draw_umich_gaussian
import os
import cv2
import xml.etree.ElementTree as ET
from dataset.transform import Transform, TransformValid, JointTransform
import pandas as pd
import math
import pygame
opj = os.path.join
from pathlib import Path
import re

def set_features(num_rect,
            noise_key,
            small_lines_key,
            shift_key,
            big_lines_key,
            ct_int,
            radius,
            features):
    draw_umich_gaussian(features[0], ct_int, radius, k=num_rect)
    draw_umich_gaussian(features[1], ct_int, radius, k=noise_key)
    draw_umich_gaussian(features[2], ct_int, radius, k=small_lines_key)
    draw_umich_gaussian(features[3], ct_int, radius, k=shift_key)
    draw_umich_gaussian(features[4], ct_int, radius, k=big_lines_key)
    return features

def get_empty_features(h, w):
    return np.zeros((5, h, w))

class BCDataset(Dataset):
    def __init__(self, root='/home/artem/PycharmProjects/CenterNetRefs/ZVZ-real-512/', resize_size=(512, 512), mode='train',
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835), classes_name=None, down_stride=4):

        self.CLASSES_NAME = classes_name
        self.class_idx = 0
        self.labels_dict = {}
        self.root = root
        self.transform = TransformValid('pascal_voc')
        if mode == "train":
            self.split_path = root + "split/split_f9_t0,1,2,3,4_seed42/dataset_train.csv"
            self.transform = Transform('pascal_voc')
        elif mode == "valid":
            self.split_path = root + "split/split_f9_t0,1,2,3,4_seed42/dataset_valid.csv"
        else:
            self.split_path = root + "split/split_f9_t0,1,2,3,4_seed42/dataset_infer.csv"

        # if mode == "train":
        #     self.split_path = root + "split/full/dataset_train.csv"
        #     self.transform = Transform('pascal_voc')
        # elif mode == "valid":
        #     self.split_path = root + "split/full/dataset_valid.csv"
        # else:
        #     self.split_path = root + "split/full/dataset_valid.csv"

        self.base_path = root
        self.samples = pd.read_csv(self.split_path)
        self.mode = mode

        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.down_stride = down_stride

        self.finderPatternsPath = "/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-real-512/dataset_markup/new_"

        finderPatternsNames = "/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-real-512/names.txt"
        self.finderPatternsNames = pd.read_csv(finderPatternsNames, sep=" ", header=None)

        finderPatternsIdxs = "/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-real-512/idxs.txt"
        self.finderPatternsIdxs = pd.read_csv(finderPatternsIdxs, sep=" ", header=None)

        self.list_n = []
        for n in self.finderPatternsIdxs[0]:
            self.list_n.append(n.split("_")[0])
        self.list_n = np.array(self.list_n)

        self.category2id = {k: v for v, k in enumerate(self.CLASSES_NAME)}
        self.id2category = {v: k for k, v in self.category2id.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.loc[idx]
        #image_name = "QR-photo_26_0043.jpg"
        #path = "/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-real-512/Image/" + image_name

        ipath = os.path.join(self.base_path, sample.image.replace("\\", "/"))  # self.base_path + sample.image
        #ipath = os.path.join(self.base_path, path)  # self.base_path + sample.image

        self.ipath = ipath
        try:
            img = Image.open(ipath).convert('RGB')
        except:
            return self.__getitem__(random.choice(range(len(self.samples))))
        self.tmp = img

        self.pattern = False
        idx_in_nameList = self.finderPatternsNames.index[self.finderPatternsNames[0] == sample.image[len("image/"):]].tolist()

        idx_in_idxList = self.finderPatternsIdxs.index[self.list_n == str(idx_in_nameList[0] + 1)].tolist()

        if len(idx_in_idxList) != 0:
            self.pattern_path = [self.finderPatternsPath + str(self.finderPatternsIdxs[0][idx]) for idx in idx_in_idxList] #+ '.txt'
            self.pattern = True

        # Delete this to turn on patterns
        self.pattern = False
        xpath = os.path.join(self.base_path, sample.objects.replace("\\", "/")) #self.base_path + sample.objects
        label = self.parse_annotation(xpath) #self.parse_synth_annotation(xpath)
        img = np.array(img)
        raw_h, raw_w, _ = img.shape
        info = {'raw_height': raw_h, 'raw_width': raw_w}
        if self.mode == 'train':
            img, boxes = self.transform(img, label['boxes'], label['labels'])
            boxes = np.array(boxes)
        else:
            boxes = np.array(label['boxes'])
        boxes_w, boxes_h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        ct = np.array([(boxes[:, 0] + boxes[:, 2]) / 2,
                       (boxes[:, 1] + boxes[:, 3]) / 2], dtype=np.float32).T

        img, boxes = self.preprocess_img_boxes(img, self.resize_size, boxes)

        info['resize_height'], info['resize_width'] = img.shape[:2]

        classes = label['labels']

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes).float()
        classes = torch.LongTensor(classes)

        output_h, output_w = info['resize_height'] // self.down_stride, info['resize_width'] // self.down_stride
        boxes_h, boxes_w, ct = boxes_h / self.down_stride, boxes_w / self.down_stride, ct / self.down_stride
        hm = np.zeros((len(self.CLASSES_NAME), output_h, output_w), dtype=np.float32)

        ct[:, 0] = np.clip(ct[:, 0], 0, output_w - 1)
        ct[:, 1] = np.clip(ct[:, 1], 0, output_h - 1)
        info['gt_hm_height'], info['gt_hm_witdh'] = output_h, output_w
        obj_mask = torch.ones(len(classes))

        feature = get_empty_features(output_h, output_w)
        for i, cls_id in enumerate(classes):
            radius = gaussian_radius((np.ceil(boxes_h[i]), np.ceil(boxes_w[i])))
            radius = max(0, int(radius))
            ct_int = ct[i].astype(np.int32)
            if (hm[:, ct_int[1], ct_int[0]] == 1).sum() >= 1.:
                obj_mask[i] = 0
                continue
            draw_umich_gaussian(hm[cls_id], ct_int, radius)
            if hm[cls_id, ct_int[1], ct_int[0]] != 1:
                obj_mask[i] = 0

            cls_name = self.id2category[cls_id.item()]
            # print(cls_name)
            if cls_name in "QRCode":
                num_rect = 1
                noise_key = 1
                small_lines_key = 0
                shift_key = 0
                big_lines_key = 0

                set_features(num_rect,
                          noise_key,
                          small_lines_key,
                          shift_key,
                          big_lines_key,
                          ct_int,
                          radius,
                          feature)

            elif cls_name in "Aztec":
                num_rect = 1
                noise_key = 1
                small_lines_key = 0
                shift_key = 0
                big_lines_key = 0

                set_features(num_rect,
                             noise_key,
                             small_lines_key,
                             shift_key,
                             big_lines_key,
                             ct_int,
                             radius,
                             feature)

            elif cls_name in "DataMatrix":
                num_rect = 0
                noise_key = 1
                small_lines_key = 0
                shift_key = 0
                big_lines_key = 0

                set_features(num_rect,
                             noise_key,
                             small_lines_key,
                             shift_key,
                             big_lines_key,
                             ct_int,
                             radius,
                             feature)

            elif cls_name in "EAN":
                num_rect = 0
                noise_key = 0
                small_lines_key = 1
                shift_key = 0
                big_lines_key = 0

                set_features(num_rect,
                             noise_key,
                             small_lines_key,
                             shift_key,
                             big_lines_key,
                             ct_int,
                             radius,
                             feature)

            elif cls_name in "PDF417":
                num_rect = 0
                noise_key = 1
                small_lines_key = 1
                shift_key = 0
                big_lines_key = 1

                set_features(num_rect,
                             noise_key,
                             small_lines_key,
                             shift_key,
                             big_lines_key,
                             ct_int,
                             radius,
                             feature)

            elif cls_name in "Postnet":
                num_rect = 0
                noise_key = 0
                small_lines_key = 1
                shift_key = 1
                big_lines_key = 0

                set_features(num_rect,
                             noise_key,
                             small_lines_key,
                             shift_key,
                             big_lines_key,
                             ct_int,
                             radius,
                             feature)

            else:
                assert (0 == 0)

        hm = torch.from_numpy(hm)

        features = torch.from_numpy(feature)
        obj_mask = obj_mask.eq(1)
        boxes = boxes[obj_mask]
        classes = classes[obj_mask]
        info['ct'] = torch.tensor(ct)[obj_mask]
        info['features'] = features
        assert hm.eq(1).sum().item() == len(classes) == len(info['ct']), \
            f"index: {index}, hm peer: {hm.eq(1).sum().item()}, object num: {len(classes)}"
        return img, boxes, classes, hm, info

    def preprocess_img_boxes(self, image, input_ksize, boxes=None):
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

        image_resized = cv2.resize(image, (nw, nh))

        pad_w = max_side - nw
        pad_h = max_side - nh

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def parse_synth_annotation(self, annotation_path):
        xml_data = open(annotation_path, 'r').read()  # Read file
        root = ET.XML(xml_data)  # Parse XML
        boxes = []
        labels = []
        difficulties = list()

        for Pages in root:
            for Page in Pages:
                points = []
                for point in Page:
                    attr_p = point.attrib
                    x = float(attr_p['X'])
                    y = float(attr_p['Y'])
                    points.append([x, y])
                points = np.array(points)

                x_min = min(points[0, 0], points[1, 0], points[2, 0], points[3, 0])
                y_min = min(points[0, 1], points[1, 1], points[2, 1], points[3, 1])
                x_max = max(points[0, 0], points[1, 0], points[2, 0], points[3, 0])
                y_max = max(points[0, 1], points[1, 1], points[2, 1], points[3, 1])
            data_i = [x_min, y_min, x_max, y_max]
            labels.append(1)
            difficulties.append('1')
            boxes.append(data_i)

        return {'boxes': np.array(boxes), 'labels': np.array(labels), 'difficulties': np.array(difficulties)}

    def parse_annotation(self, annotation_path):
        xml_data = open(annotation_path, 'r').read()  # Read file
        root = ET.XML(xml_data)  # Parse XML
        boxes = []
        labels = []
        difficulties = list()
        #print("---------------------------------------")
        if self.pattern:
            for pattern_path in self.pattern_path:
                with open(pattern_path) as f:
                    print("pattern_data")
                    pattern_data = f.read()

                    iter = re.finditer(r"\[([0-9, ]+)\]", pattern_data)
                    pattern_boxes = []
                    for m in iter:
                        data_i = np.array([int(val) for val in pattern_data[m.start() + 1:m.end() - 1].split(",")])

                        if not (data_i[0] == data_i[2] or data_i[1] == data_i[3]):
                            pattern_boxes.append(data_i)

                    if len(pattern_boxes) == 3:
                        print("pattern_data", pattern_data, annotation_path)
                        boxes.extend(pattern_boxes)
                        label = self.category2id["FinderPatterns"]
                        labels.append(label)
                        difficulties.append('1')

                        labels.append(label)
                        difficulties.append('1')

                        labels.append(label)
                        difficulties.append('1')

                        #for data_i in pattern_boxes:
                        #    cv2.imshow("test",
                        #    cv2.rectangle(np.array(self.tmp), (data_i[0], data_i[1]), (data_i[2], data_i[3]),
                        #                 (255, 0, 0), 2))
                        #    cv2.waitKey()

        for Pages in root:
            for Page in Pages:
                for child in Page:
                    if child.tag == 'Barcodes':
                        barcodes = child
                        for barcode in barcodes:
                            class_name = barcode.attrib["Type"]

                            #print(class_name, class_name in self.category2id)
                            if class_name in self.category2id:
                                label = self.category2id[class_name]
                            else:
                                if "RoyalMailCode" in class_name:
                                    class_name = "Postnet"

                                elif "JapanPost" in class_name:
                                    class_name = "Postnet"

                                elif "IntelligentMail" in class_name:
                                    class_name = "Postnet"

                                elif "Kix" in class_name:
                                    class_name = "Postnet"

                                elif "EAN" in class_name:
                                    class_name = "EAN"

                                elif "2-Digit" in class_name:
                                    class_name = "EAN"

                                elif "Code" in class_name:
                                    class_name = "EAN"

                                elif "IATA" in class_name:
                                    class_name = "EAN"

                                elif "UPCA" in class_name:
                                    class_name = "EAN"

                                elif "Interleaved25" in class_name:
                                    class_name = "EAN"

                                elif "UCC128" in class_name:
                                    class_name = "EAN"


                                #elif "Aztec" in class_name:
                                #    class_name = "QRCode"
                                #print("->")

                                if class_name in self.category2id:
                                    label = self.category2id[class_name]
                                else:
                                    # print(class_name, annotation_path)
                                    label = 0

                            #print(class_name, class_name in self.category2id, label)
                            #print("/////////////////////////////////////////////////////")
                            for Polygon in barcode:
                                for Points in Polygon:
                                    x = 0
                                    y = 0
                                    points = []
                                    for point in Points:
                                        attr_p = point.attrib
                                        x = float(attr_p['X'])
                                        y = float(attr_p['Y'])
                                        points.append([x, y])
                                    points = np.array(points)

                                    x_min = min(points[0, 0], points[1, 0], points[2, 0], points[3, 0])
                                    y_min = min(points[0, 1], points[1, 1], points[2, 1], points[3, 1])
                                    x_max = max(points[0, 0], points[1, 0], points[2, 0], points[3, 0])
                                    y_max = max(points[0, 1], points[1, 1], points[2, 1], points[3, 1])
                            data_i = [x_min, y_min, x_max, y_max]
                            labels.append(label)
                            difficulties.append('1')
                            boxes.append(data_i)

        return {'boxes': np.array(boxes), 'labels': np.array(labels), 'difficulties': np.array(difficulties)}

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list, hm_list, infos = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []
        pad_hm_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()

        for i in range(batch_size):
            img = imgs_list[i]
            hm = hm_list[i]

            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

            pad_hm_list.append(
                torch.nn.functional.pad(hm, (0, int(max_w // self.down_stride - hm.shape[2]), 0, int(max_h // self.down_stride - hm.shape[1])),
                                        value=0.))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num:
                max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)
        batch_hms = torch.stack(pad_hm_list)
        #print(batch_classes, "batch_classes", self.mode)
        return batch_imgs, batch_boxes, batch_classes, batch_hms, infos

class BCDatasetValid(Dataset):

    def __init__(self, root='/home/artem/PycharmProjects/CenterNetRefs/ZVZ-real-512/', resize_size=(512, 512), mode='train',
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835), classes_name=None, down_stride=4):
        self.CLASSES_NAME = classes_name

        self.class_idx = 0
        self.labels_dict = {}

        self.root = root
        self.transform = TransformValid('pascal_voc')
        if mode == "train":
            self.split_path = root + "split/split_f9_t0,1,2,3,4_seed42/dataset_train.csv"
        elif mode == "valid":
            self.split_path = root + "split/split_f9_t0,1,2,3,4_seed42/dataset_valid.csv"
        else:
            self.split_path = root + "split/split_f9_t0,1,2,3,4_seed42/dataset_infer.csv"

        # if mode == "train":
        #     self.split_path = root + "split/full/dataset_train.csv"
        #     self.transform = Transform('pascal_voc')
        # elif mode == "valid":
        #     self.split_path = root + "split/full/dataset_valid.csv"
        # else:
        #     self.split_path = root + "split/full/dataset_valid.csv"

        self.base_path = root
        self.samples = pd.read_csv(self.split_path)
        self.mode = mode


        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.down_stride = down_stride

        self.category2id = {k: v for v, k in enumerate(self.CLASSES_NAME)}
        self.id2category = {v: k for k, v in self.category2id.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.loc[idx]

        ipath = os.path.join(self.base_path, sample.image.replace("\\", "/")) #self.base_path + sample.image
        self.ipath = ipath

        img = Image.open(ipath).convert('RGB')

        xpath = os.path.join(self.base_path, sample.objects.replace("\\", "/")) #self.base_path + sample.objects
        label = self.parse_annotation(xpath) #self.parse_synth_annotation(xpath)
        img = np.array(img)
        raw_h, raw_w, _ = img.shape
        info = {'raw_height': raw_h, 'raw_width': raw_w}

        boxes = np.array(label['boxes'])
        boxes_w, boxes_h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        ct = np.array([(boxes[:, 0] + boxes[:, 2]) / 2,
                       (boxes[:, 1] + boxes[:, 3]) / 2], dtype=np.float32).T

        img, boxes = self.preprocess_img_boxes(img, self.resize_size, boxes)
        info['resize_height'], info['resize_width'] = img.shape[:2]

        classes = label['labels']

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes).float()
        classes = torch.LongTensor(classes)

        output_h, output_w = info['resize_height'] // self.down_stride, info['resize_width'] // self.down_stride
        boxes_h, boxes_w, ct = boxes_h / self.down_stride, boxes_w / self.down_stride, ct / self.down_stride
        hm = np.zeros((len(self.CLASSES_NAME), output_h, output_w), dtype=np.float32)
        ct[:, 0] = np.clip(ct[:, 0], 0, output_w - 1)
        ct[:, 1] = np.clip(ct[:, 1], 0, output_h - 1)
        info['gt_hm_height'], info['gt_hm_witdh'] = output_h, output_w
        obj_mask = torch.ones(len(classes))
        for i, cls_id in enumerate(classes):
            radius = gaussian_radius((np.ceil(boxes_h[i]), np.ceil(boxes_w[i])))
            radius = max(0, int(radius))
            ct_int = ct[i].astype(np.int32)
            if (hm[:, ct_int[1], ct_int[0]] == 1).sum() >= 1.:
                obj_mask[i] = 0
                continue

            draw_umich_gaussian(hm[cls_id - 1], ct_int, radius)
            if hm[cls_id - 1, ct_int[1], ct_int[0]] != 1:
                obj_mask[i] = 0

        hm = torch.from_numpy(hm)
        obj_mask = obj_mask.eq(1)
        boxes = boxes[obj_mask]
        classes = classes[obj_mask]
        info['ct'] = torch.tensor(ct)[obj_mask]

        assert hm.eq(1).sum().item() == len(classes) == len(info['ct']), \
            f"index: {index}, hm peer: {hm.eq(1).sum().item()}, object num: {len(classes)}"
        return img, boxes, classes, hm, info

    def preprocess_img_boxes(self, image, input_ksize, boxes=None):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = image.shape
        _pad = 32  # 32

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = _pad - nw % _pad
        pad_h = _pad - nh % _pad

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def parse_annotation(self, annotation_path):
        xml_data = open(annotation_path, 'r').read()  # Read file
        root = ET.XML(xml_data)  # Parse XML
        boxes = []
        labels = []
        difficulties = list()

        for Pages in root:
            for Page in Pages:
                for child in Page:
                    if child.tag == 'Barcodes':
                        barcodes = child
                        for barcode in barcodes:
                            class_name = barcode.attrib["Type"]

                            if class_name in self.category2id:
                                label = self.category2id[class_name]
                            else:
                                if "RoyalMailCode" in class_name:
                                    class_name = "Postnet"

                                elif "JapanPost" in class_name:
                                    class_name = "Postnet"

                                elif "IntelligentMail" in class_name:
                                    class_name = "Postnet"

                                elif "Kix" in class_name:
                                    class_name = "Postnet"
                                elif "EAN" in class_name:
                                    class_name = "EAN"

                                elif "2-Digit" in class_name:
                                    class_name = "EAN"

                                elif "Code" in class_name:
                                    class_name = "EAN"

                                elif "IATA" in class_name:
                                    class_name = "EAN"

                                elif "UPCA" in class_name:
                                    class_name = "EAN"

                                elif "Interleaved25" in class_name:
                                    class_name = "EAN"

                                elif "UCC128" in class_name:
                                    class_name = "EAN"

                                #elif "DataMatrix" in class_name:
                                #    class_name = "QRCode"

                                #elif "Aztec" in class_name:
                                #    class_name = "QRCode"

                                if class_name in self.category2id:
                                    label = self.category2id[class_name]
                                else:
                                    print(class_name, annotation_path)
                                    label = 0

                            for Polygon in barcode:
                                for Points in Polygon:
                                    x = 0
                                    y = 0
                                    points = []
                                    for point in Points:
                                        attr_p = point.attrib
                                        x = float(attr_p['X'])
                                        y = float(attr_p['Y'])
                                        points.append([x, y])
                                    points = np.array(points)
                                    #
                                    # x_min = min(points[0, 0], points[1, 0], points[2, 0], points[3, 0])
                                    # y_min = min(points[0, 1], points[1, 1], points[2, 1], points[3, 1])
                                    # x_max = max(points[0, 0], points[1, 0], points[2, 0], points[3, 0])
                                    # y_max = max(points[0, 1], points[1, 1], points[2, 1], points[3, 1])
                            data_i = [points[0, 0], points[0, 1], points[1, 0], points[1, 1], points[2, 0], points[2, 1], points[3, 0], points[3, 1]]
                            labels.append(label)
                            difficulties.append('1')
                            boxes.append(data_i)

        return {'boxes': np.array(boxes), 'labels': np.array(labels), 'difficulties': np.array(difficulties)}

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list, hm_list, infos = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []
        pad_hm_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()

        for i in range(batch_size):
            img = imgs_list[i]
            hm = hm_list[i]

            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

            pad_hm_list.append(
                torch.nn.functional.pad(hm, (0, int(max_w // 4 - hm.shape[2]), 0, int(max_h // 4 - hm.shape[1])),
                                        value=0.))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num:
                max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)
        batch_hms = torch.stack(pad_hm_list)

        return batch_imgs, batch_boxes, batch_classes, batch_hms, infos

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def pol2cart(rho, phi):
    x = rho * np.cos(np.radians(phi))
    y = rho * np.sin(np.radians(phi))
    return [x, y]

class BCJointDataset(BCDataset):
    def  __init__(self, root='/home/artem/PycharmProjects/CenterNetRefs/ZVZ-real-512/', resize_size=(512, 512), mode='train',
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835)):
        super(BCJointDataset, self).__init__(root=root, resize_size=resize_size, mode=mode,
                 mean=mean, std=std)
        #print(mode, "!!!!!!!!!!!!!!!!!!!!")
        if mode == "train":
            self.transform = JointTransform('xy')
        else:
            self.transform = JointTransform('xy', train=False)

    def __getitem__(self, idx):

        sample = self.samples.loc[idx]
        ipath = os.path.join(self.base_path, sample.image.replace("\\", "/"))
        self.ipath = ipath

        img = Image.open(ipath).convert('RGB')

        xpath = os.path.join(self.base_path, sample.objects.replace("\\", "/")) #self.base_path + sample.objects
        label = self.parse_annotation(xpath)
        img = np.array(img)
        raw_h, raw_w, _ = img.shape
        info = {'raw_height': raw_h, 'raw_width': raw_w}
        if self.mode == 'train':
            img, keypoints = self.transform(img, label['keypoints'])
            keypoints_b = []
            i = 0
            poligon = []
            for point in keypoints:
                poligon.append(point)
                if i == 3:
                    i = 0
                    keypoints_b.append(poligon)
                    poligon = []
                    continue
                i += 1
            keypoints = np.array(keypoints_b)
        else:
            keypoints = np.array(label['keypoints'])
            keypoints_b = []
            i = 0
            poligon = []
            for point in keypoints:
                poligon.append(point)
                if i == 3:
                    i = 0
                    keypoints_b.append(poligon)
                    poligon = []
                    continue
                i += 1
            keypoints = np.array(keypoints_b)

        max_keypoint_x = np.max(keypoints[:, :, 0], axis=1)
        min_keypoint_x = np.min(keypoints[:, :, 0], axis=1)
        max_keypoint_y = np.max(keypoints[:, :, 1], axis=1)
        min_keypoint_y = np.min(keypoints[:, :, 1], axis=1)

        boxes_w, boxes_h = max_keypoint_x - min_keypoint_x, max_keypoint_y - min_keypoint_y
        ct = [[(max_keypoint_x_i + min_keypoint_x_i) / 2, (max_keypoint_y_i + min_keypoint_y_i) / 2 ]
              for max_keypoint_x_i, min_keypoint_x_i, max_keypoint_y_i, min_keypoint_y_i in zip(max_keypoint_x, min_keypoint_x, max_keypoint_y, min_keypoint_y)]
        ct = np.array(ct)

        img, keypoints = self.preprocess_img_boxes(img, self.resize_size, keypoints)
        info['resize_height'], info['resize_width'] = img.shape[:2]

        classes = label['labels']

        img = transforms.ToTensor()(img)
        keypoints = torch.from_numpy(keypoints).float()
        classes = torch.LongTensor(classes)

        output_h, output_w = info['resize_height'] // self.down_stride, info['resize_width'] // self.down_stride
        boxes_h, boxes_w, ct = boxes_h / self.down_stride, boxes_w / self.down_stride, ct / self.down_stride
        hm = np.zeros((len(self.CLASSES_NAME), output_h, output_w), dtype=np.float32)
        ct[:, 0] = np.clip(ct[:, 0], 0, output_w - 1)
        ct[:, 1] = np.clip(ct[:, 1], 0, output_h - 1)
        info['gt_hm_height'], info['gt_hm_witdh'] = output_h, output_w
        obj_mask = torch.ones(len(classes))
        for i, cls_id in enumerate(classes):
            radius = gaussian_radius((np.ceil(boxes_h[i]), np.ceil(boxes_w[i])))
            radius = max(0, int(radius))
            ct_int = ct[i].astype(np.int32)
            if (hm[:, ct_int[1], ct_int[0]] == 1).sum() >= 1.:
                obj_mask[i] = 0
                continue

            draw_umich_gaussian(hm[cls_id - 1], ct_int, radius)
            if hm[cls_id - 1, ct_int[1], ct_int[0]] != 1:
                obj_mask[i] = 0

        hm = torch.from_numpy(hm)
        obj_mask = obj_mask.eq(1)
        keypoints = keypoints[obj_mask]
        classes = classes[obj_mask]
        info['ct'] = torch.tensor(ct)[obj_mask]

        assert hm.eq(1).sum().item() == len(classes) == len(info['ct']), \
            f"index: {index}, hm peer: {hm.eq(1).sum().item()}, object num: {len(classes)}"


        keypoints_flatted = []
        for ct_i, poligon_i in zip(ct, keypoints):
            #print(poligon_i[0][0], poligon_i[0][1] )
            cords1 = pygame.math.Vector2(poligon_i[0][0] - ct_i[0], poligon_i[0][1] - ct_i[1]).as_polar()
            cords2 = pygame.math.Vector2(poligon_i[1][0] - ct_i[0], poligon_i[1][1] - ct_i[1]).as_polar()
            cords3 = pygame.math.Vector2(poligon_i[2][0] - ct_i[0], poligon_i[2][1] - ct_i[1]).as_polar()
            cords4 = pygame.math.Vector2(poligon_i[3][0] - ct_i[0], poligon_i[3][1] - ct_i[1]).as_polar()

            poligon_i[0][0], poligon_i[0][1] = cords1
            poligon_i[1][0], poligon_i[1][1] = cords2
            poligon_i[2][0], poligon_i[2][1] = cords3
            poligon_i[3][0], poligon_i[3][1] = cords4
            keypoints_flatted.append([poligon_i[0][0], poligon_i[0][1],
                                      poligon_i[1][0], poligon_i[1][1],
                                      poligon_i[2][0], poligon_i[2][1],
                                      poligon_i[3][0], poligon_i[3][1]])

            # pole = pol2cart(cords1[0], cords1[1])
            # pole[0] += ct_i[0]
            # pole[1] += ct_i[1]
            # print(pole)
            # print("_____________-")
        keypoints_flatted = torch.tensor(keypoints_flatted)
        return img, keypoints_flatted, classes, hm, info

    def parse_annotation(self, annotation_path):
        xml_data = open(annotation_path, 'r').read()  # Read file
        root = ET.XML(xml_data)  # Parse XML
        boxes = []
        labels = []
        difficulties = list()
        keypoints = []
        for Pages in root:
            for Page in Pages:
                for child in Page:
                    if child.tag == 'Barcodes':
                        barcodes = child
                        for barcode in barcodes:
                            class_name = barcode.attrib["Type"]
                            for Polygon in barcode:
                                for Points in Polygon:
                                    x = 0
                                    y = 0
                                    points = []
                                    for point in Points:
                                        attr_p = point.attrib
                                        x = float(attr_p['X'])
                                        y = float(attr_p['Y'])
                                        points.append([x, y])
                                        keypoints.append((x, y))
                            labels.append(1)
                            difficulties.append('1')

        return {'keypoints': keypoints, 'labels': np.array(labels), 'difficulties': np.array(difficulties)}

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list, hm_list, infos = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []
        pad_hm_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()

        for i in range(batch_size):
            img = imgs_list[i]
            hm = hm_list[i]

            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

            pad_hm_list.append(
                torch.nn.functional.pad(hm, (0, int(max_w // 4 - hm.shape[2]), 0, int(max_h // 4 - hm.shape[1])),
                                        value=0.))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num:
                max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)
        batch_hms = torch.stack(pad_hm_list)

        return batch_imgs, batch_boxes, batch_classes, batch_hms, infos

if __name__ == '__main__':
    ds = BCJointDataset('/home/artem/PycharmProjects/CenterNetRefs/ZVZ-real-512/', resize_size=(512, 512))
    dl = DataLoader(ds, batch_size=12, collate_fn=ds.collate_fn)
    batch = next(iter(dl))
    print(batch)
    # print(len(dl))
    # import tqdm
    # for i, data in enumerate(tqdm.tqdm(dl)):
    #     # pass
    #     imgs, boxes, classes, hms, infos = data
    #     for b in range(hms.size(0)):
    #         if hms[b].eq(1).sum(0).gt(0).sum() != classes[b].gt(0).sum():
    #             import pdb; pdb.set_trace()
