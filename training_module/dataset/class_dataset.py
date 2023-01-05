import os
import torch
import numpy as np
from os import walk
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import albumentations as A
from ..helpers import pixel_to_normalized_coordinates 
 

class SpineDataset(Dataset):
    """
    Creates a proper pytorch Dataset object using Spine dataset;

    Args: root (str): The location of the dataset images annFile (str): The location of the annotations for the
    images in dataset transform (Transform): Transformation that is applied to images and keypoints
    type_normalization (int): Indicates de normalization for keypoints. If "1" values are going to be between [0,1],
    if "2" between [-1,1], and "0" keypoints are not normalize.

    Methods:
        getDicomPath(index):
            Return the path where image is storage. This info is obtained using the annotation file read

        getTarget(index):
            Returns a list that contained all annotation associated to a certain file

        getKeypoints(target):
            Return all keypoints associated to the file

        reshapeKeypoints(keypoints):
            Returns keypoints as an 2D array [x,y], instead of [x, y, visibility]

    Returns:
        Dataset: a dataset made using location for files provided
    """

    def __init__(self, root: str, ann_file: str, transform = None, type_normalization: int = 0):
        self.root = root
        self.coco = COCO(annotation_file=ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.paths = [path['file_name'] for path in self.coco.imgs.values()]
        self.transform = transform
        self.type_normalization = type_normalization
        self.view = [path['coco_url'] for path in self.coco.imgs.values()]

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index: int):
        image_path = self.getDicomPath(index)
        target = self.getTarget(index)
        keypoints = self.getKeypoints(target)

        raw_img = np.array(Image.open(os.path.join(self.root, image_path)).convert("L"), dtype=np.float32)
        keypoints = self.reshapeKeypoints(keypoints)

        if type(self.transform) == A.core.composition.Compose:
            raw_img = np.repeat(raw_img.reshape(raw_img.shape[0], raw_img.shape[1], 1), 3, 2).astype(np.uint8)
            augmentation = self.transform(image=raw_img, keypoints=keypoints)
            raw_img, keypoints = augmentation['image'], augmentation['keypoints']

            _keypoints = np.array(keypoints).copy()

            if self.type_normalization == 1:
                _keypoints = (_keypoints - raw_img.shape[2]) / raw_img.shape[2]

            elif self.type_normalization == 2:
                _keypoints = pixel_to_normalized_coordinates(_keypoints, raw_img.permute(0,2,1).shape[1:])

            return torch.unsqueeze(raw_img[0], dim=0), _keypoints
        elif type(self.transform) == dict:
            raw_img = np.repeat(raw_img.reshape(raw_img.shape[0], raw_img.shape[1], 1), 3, 2).astype(np.uint8)
            
            augmentation = self.transform[self.getView(index)](image=raw_img, keypoints=keypoints)
            
            raw_img, keypoints = augmentation['image'], augmentation['keypoints']

            _keypoints = np.array(keypoints).copy()

            if self.type_normalization == 1:
                _keypoints = (_keypoints - raw_img.shape[2]) / raw_img.shape[2]

            elif self.type_normalization == 2:
                _keypoints = pixel_to_normalized_coordinates(_keypoints, raw_img.permute(0,2,1).shape[1:])

            return torch.unsqueeze(raw_img[0], dim=0), _keypoints
        else:
            return {'image': raw_img,
                    'name': image_path,
                    'keypoints': keypoints}


    def getDicomPath(self, index: int):
        return self.paths[index]


    def getTarget(self, index: int):
        img_id = self.ids[index]
        self.coco.getImgIds()
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        return target

    def getView(self, index: int):
        return self.view[index]

    def getKeypoints(self, target):
        # Number of objects in the image
        num_objs = len(target)
        keypoints = []
        for i in range(num_objs):
            if target[i]['keypoints']:
                keypoints.append(target[i]['keypoints'])
        return keypoints


    def reshapeKeypoints(self, keypoints):
        keypoints = keypoints[0]
        keypoints = np.array_split(keypoints, int(np.size(keypoints) / 3), axis=0)
        new_keypoints = []
        for key in keypoints:
            new_keypoints.append(key[:-1])
        return np.array(new_keypoints)
 

class SpineRXImages(Dataset):

    def __init__(self, root: str, transform=None):
        self.root = root
        self.files = self.get_images(root_path=root)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        raw_img = np.array(Image.open(self.files[index]).convert("L"), dtype=np.float32)

        if self.transform:
            raw_img = np.repeat(raw_img.reshape(raw_img.shape[0], raw_img.shape[1], 1), 3, 2).astype(np.uint8)
            augmentation = self.transform(image=raw_img)
            raw_img = augmentation['image']
            return torch.unsqueeze(raw_img[0], dim=0)
        else:
            return {'image': raw_img,
                    'name': self.files[index]}

    def get_images(self, root_path):
        jpg_files = []
        for (dirpath, dirnames, filenames) in walk(root_path):
            filenames = [ fi for fi in filenames if fi.endswith(('.jpg'))]
            if filenames:
                for filename in filenames:
                    jpg_files.append(os.path.join(dirpath, filename).replace('\\', '/'))
        return jpg_files