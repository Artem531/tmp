from albumentations import Compose, BboxParams, KeypointParams, \
    RandomBrightnessContrast, GaussNoise, RGBShift, CLAHE, RandomGamma, HorizontalFlip, RandomResizedCrop


class Transform(object):
    def __init__(self, box_format='coco'):
        self.tsfm = Compose([
            HorizontalFlip(),
            # RandomResizedCrop(512, 512, scale=(0.75, 1)),
            RandomBrightnessContrast(0.4, 0.4),
            GaussNoise(),
            RGBShift(),
            CLAHE(),
            RandomGamma()
        ], bbox_params=BboxParams(format=box_format, min_visibility=0.75, label_fields=['labels']))

    def __call__(self, img, boxes, labels):
        augmented = self.tsfm(image=img, bboxes=boxes, labels=labels)
        img, boxes = augmented['image'], augmented['bboxes']
        return img, boxes


class JointTransform(object):
    def __init__(self, box_format='xy', train=True):
        self.train = train
        self.tsfm = Compose([
            HorizontalFlip(),
            # RandomResizedCrop(512, 512, scale=(0.75, 1)),

            RandomBrightnessContrast(0.4, 0.4),
            GaussNoise(),
            RGBShift(),
            CLAHE(),
            RandomGamma()
        ], keypoint_params=KeypointParams(format=box_format, remove_invisible=True))

        self.tsfm_reserve = Compose([
            # HorizontalFlip(),
            # RandomResizedCrop(512, 512, scale=(0.75, 1)),

            RandomBrightnessContrast(0.4, 0.4),
            GaussNoise(),
            RGBShift(),
            CLAHE(),
            RandomGamma()
        ], keypoint_params=KeypointParams(format=box_format, remove_invisible=True))

    def __call__(self, img, keypoints):
        augmented = self.tsfm(image=img, keypoints=keypoints)
        img, augmented_keypoints = augmented['image'], augmented['keypoints']
        if len(augmented_keypoints) != len(keypoints) or self.train != True:
            augmented = self.tsfm_reserve(image=img, keypoints=keypoints)
            img, augmented_keypoints = augmented['image'], augmented['keypoints']
        return img, augmented_keypoints

class TransformValid(object):
    def __init__(self, box_format='coco'):
        self.tsfm = Compose([
            # HorizontalFlip(),
            # RandomResizedCrop(512, 512, scale=(0.75, 1)),
            RandomBrightnessContrast(0.4, 0.4),
            GaussNoise(),
            RGBShift(),
            CLAHE(),
            RandomGamma()
        ], bbox_params=BboxParams(format=box_format, min_visibility=0.75, label_fields=['labels']))

    def __call__(self, img, boxes, labels):
        augmented = self.tsfm(image=img, bboxes=boxes, labels=labels)
        img, boxes = augmented['image'], augmented['bboxes']
        return img, boxes


