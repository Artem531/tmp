import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os
import xml.etree.ElementTree as ET
from dataset.transform import TransformValid, JointTransform
import pandas as pd

opj = os.path.join
from PIL import Image, ImageDraw

class BCDataset(Dataset):
    def __init__(self, root='/home/artem/PycharmProjects/CenterNetRefs/ZVZ-real-512/', resize_size=(512, 512), mode='train',
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835), classes_name=None):

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

        self.base_path = root
        self.samples = pd.read_csv(self.split_path)
        self.mode = mode

        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.down_stride = 4

        self.category2id = {k: v for v, k in enumerate(self.CLASSES_NAME)}
        self.id2category = {v: k for k, v in self.category2id.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.loc[idx]

        ipath = os.path.join(self.base_path, sample.image.replace("\\", "/"))  # self.base_path + sample.image

        self.ipath = ipath
        try:
            img = Image.open(ipath).convert('RGB')
        except:
            print(ipath)
            return self.__getitem__(random.choice(range(len(self.samples))))

        width, height = img.size[0], img.size[1]

        xpath = os.path.join(self.base_path, sample.objects.replace("\\", "/"))
        label = self.parse_annotation(xpath)
        img = np.array(img)
        img = img.astype(np.uint8())

        raw_h, raw_w, _ = img.shape

        polygons = label['polygons']
        classes = np.array(label['classes'])
        mask = Image.new('L', (width, height), 0)

        for polygon, class_label in zip(polygons, classes):
            ImageDraw.Draw(mask).polygon(polygon, outline=0, fill=int(class_label))

        mask = np.array(mask)
        #w, h = self.resize_size
        #mask = cv2.resize(mask, (w, h))

        AnnMap = torch.unsqueeze(torch.Tensor(mask), dim=0)
        img = transforms.ToTensor()(img)

        return img, AnnMap, idx

    def parse_annotation(self, annotation_path):
        xml_data = open(annotation_path, 'r').read()  # Read file
        root = ET.XML(xml_data)  # Parse XML
        polygons = []
        labels = []

        self.barcode_boxes = []

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

                                label = self.category2id[class_name]

                            for Polygon in barcode:
                                for Points in Polygon:
                                    x = 0
                                    y = 0
                                    points = []
                                    for point in Points:
                                        attr_p = point.attrib
                                        x = float(attr_p['X'])
                                        y = float(attr_p['Y'])
                                        points.append((x, y))
                            data_i = points

                            labels.append(label)
                            polygons.append(data_i)
                            self.barcode_boxes.append(data_i)

        return {'polygons': polygons, 'classes': labels}

    def collate_fn(self, data):
        imgs_list, mask_list, idx = zip(*data)
        assert len(imgs_list) == len(mask_list) == len(idx)

        batch_size = len(imgs_list)
        pad_imgs_list = []
        pad_mask_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = 512
        max_w = 512

        for i in range(batch_size):
            img = imgs_list[i]
            ann = mask_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

            pad_mask_list.append(
                torch.nn.functional.pad(ann, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)[0])

        batch_imgs = torch.stack(pad_imgs_list)
        batch_mask = torch.stack(pad_mask_list)

        return batch_imgs, batch_mask, idx


class BCDatasetPatterns(BCDataset):
    def __init__(self, root='/home/artem/PycharmProjects/CenterNetRefs/ZVZ-real-512/', resize_size=(512, 512), mode='train',
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835), classes_name=None, patterns_csv_path=None):

        super(BCDatasetPatterns, self).__init__(root, resize_size, mode, mean, std, classes_name)

        self.csv = pd.read_csv(patterns_csv_path)
        names = []
        for name in self.csv["filename"]:
            names.append(name.split(".")[0])

        self.csv["filename"] = names

    def get_patterns(self, image_name):
        polygons = []
        classes = []

        sample_with_images = self.csv[self.csv["filename"] == image_name.split(".")[0]]
        for polygon_data in zip(sample_with_images["region_shape_attributes"], sample_with_images["region_attributes"]):
            polygon_data, polygon_class = polygon_data

            if polygon_class == "{}":
                continue

            label = self.category2id[polygon_class]
            classes.append(label)

            try:
                _, polygon_data = polygon_data.split("all_points_x")

                polygon_data_x, polygon_data_y = polygon_data.split("all_points_y")

                polygon_data_x = polygon_data_x[3:-3]
                polygon_data_y = polygon_data_y[3:-2]

                polygon_data_x = [int(x) for x in polygon_data_x.split(",")]
                polygon_data_y = [int(x) for x in polygon_data_y.split(",")]

                polygons.append([(x, y) for x, y in zip(polygon_data_x, polygon_data_y)])
            except:
                continue

        return polygons, classes

    def __getitem__(self, idx):
        sample = self.samples.loc[idx]

        ipath = os.path.join(self.base_path, sample.image.replace("\\", "/"))  # self.base_path + sample.image

        self.ipath = ipath
        try:
            img = Image.open(ipath).convert('RGB')
        except Exception as e:
            print(e, ipath)
            return self.__getitem__(random.choice(range(len(self.samples))))

        width, height = img.size[0], img.size[1]

        xpath = os.path.join(self.base_path, sample.objects.replace("\\", "/"))
        label = self.parse_annotation(xpath)
        img = np.array(img)
        img = img.astype(np.uint8())

        raw_h, raw_w, _ = img.shape

        barcode_polygons = label['polygons']
        barcode_classes = label['classes']

        image_name = ipath.split("/")[-1]
        patterns_polygons, patterns_classes = self.get_patterns(image_name)

        if len(barcode_polygons) != 0:
            polygons = barcode_polygons + patterns_polygons
            classes = barcode_classes + patterns_classes
        else:
            polygons = barcode_polygons
            classes = barcode_classes

        classes = np.array(classes)

        full_mask = np.zeros((len(self.category2id), height, width))
        for polygon, class_label in zip(polygons, classes):
            mask = Image.new('L', (width, height), 0)
            ImageDraw.Draw(mask).polygon(polygon, outline=0, fill=1)
            full_mask[class_label] += mask

        mask = np.array(full_mask)
        #w, h = self.resize_size
        #mask = cv2.resize(mask, (w, h))

        AnnMap = torch.unsqueeze(torch.Tensor(mask), dim=0)
        img = transforms.ToTensor()(img)

        return img, AnnMap, idx

class BCDatasetValid(Dataset):
    def __init__(self, root='/home/artem/PycharmProjects/CenterNetRefs/ZVZ-real-512/', resize_size=(512, 512), mode='train',
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835), classes_name=None):

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

        self.base_path = root
        self.samples = pd.read_csv(self.split_path)
        self.mode = mode

        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.down_stride = 4

        self.category2id = {k: v for v, k in enumerate(self.CLASSES_NAME)}
        self.id2category = {v: k for k, v in self.category2id.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.loc[idx]

        ipath = os.path.join(self.base_path, sample.image.replace("\\", "/"))  # self.base_path + sample.image

        self.ipath = ipath
        try:
            img = Image.open(ipath).convert('RGB')
        except:
            return self.__getitem__(random.choice(range(len(self.samples))))

        width, height = img.size[0], img.size[1]

        xpath = os.path.join(self.base_path, sample.objects.replace("\\", "/"))
        label = self.parse_annotation(xpath)
        img = np.array(img)
        img = img.astype(np.uint8())

        raw_h, raw_w, _ = img.shape
        polygons = label['polygons']
        img = transforms.ToTensor()(img)

        return img, polygons

    def parse_annotation(self, annotation_path):
        xml_data = open(annotation_path, 'r').read()  # Read file
        root = ET.XML(xml_data)  # Parse XML
        polygons = []
        labels = []

        self.barcode_boxes = []

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

                                label = self.category2id[class_name]

                            for Polygon in barcode:
                                for Points in Polygon:
                                    x = 0
                                    y = 0
                                    points = []
                                    for point in Points:
                                        attr_p = point.attrib
                                        x = float(attr_p['X'])
                                        y = float(attr_p['Y'])
                                        points.append((int(x), int(y)))
                            data_i = points

                            labels.append(label)
                            polygons.append(data_i)
                            self.barcode_boxes.append(data_i)

        return {'polygons': polygons, 'classes': np.array(labels)}

    def collate_fn(self, data):
        imgs_list, polygon_list = zip(*data)

        batch_size = len(imgs_list)
        pad_imgs_list = []
        pad_mask_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = 512
        max_w = 512

        for i in range(batch_size):
            img = imgs_list[i]

            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, polygon_list



class BCDatasetValidSyntetic(Dataset):
    def __init__(self, root='/media/artem/A2F4DEB0F4DE85C7/myData/datasets/barcodes/ZVZ-synth-512/', resize_size=(512, 512), mode='train',
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835), classes_name=None):

        self.CLASSES_NAME = classes_name

        self.class_idx = 0
        self.labels_dict = {}
        self.root = root
        self.transform = TransformValid('pascal_voc')

        if mode == "train":
            self.split_path = root + "split/full/dataset_train.csv"
        elif mode == "valid":
            self.split_path = root + "split/full/dataset_valid.csv"
        else:
            self.split_path = root + "split/full/dataset_valid.csv"

        self.base_path = root
        self.samples = pd.read_csv(self.split_path)
        self.mode = mode

        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.down_stride = 4

        self.category2id = {k: v for v, k in enumerate(self.CLASSES_NAME)}
        self.id2category = {v: k for k, v in self.category2id.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.loc[idx]

        ipath = os.path.join(self.base_path, sample.image.replace("\\", "/"))  # self.base_path + sample.image

        self.ipath = ipath
        try:
            img = Image.open(ipath).convert('RGB')
        except:
            return self.__getitem__(random.choice(range(len(self.samples))))

        width, height = img.size[0], img.size[1]

        xpath = os.path.join(self.base_path, sample.objects.replace("\\", "/"))
        label = self.parse_synth_annotation(xpath)
        img = np.array(img)
        img = img.astype(np.uint8())

        raw_h, raw_w, _ = img.shape
        polygons = label['polygons']

        mask = Image.new('L', (width, height), 0)

        for polygon in polygons:
            ImageDraw.Draw(mask).polygon(polygon, outline=0, fill=1)
        # cv2.imshow("aaa", np.array(mask) / 1.)
        # cv2.waitKey(0)
        # cv2.imshow("ddd", np.array(img) / 255)
        # cv2.waitKey(0)

        img = transforms.ToTensor()(img)

        return img, polygons

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
                    points.append((int(x), int(y)))

            data_i = points
            labels.append(1)
            difficulties.append('1')
            boxes.append(data_i)

        return {'polygons': boxes, 'labels': np.array(labels)}

    def collate_fn(self, data):
        imgs_list, polygon_list = zip(*data)

        batch_size = len(imgs_list)
        pad_imgs_list = []
        pad_mask_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = 512
        max_w = 512

        for i in range(batch_size):
            img = imgs_list[i]

            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, polygon_list