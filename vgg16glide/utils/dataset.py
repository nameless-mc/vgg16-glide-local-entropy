from typing import List
import torch.utils.data as data
import PIL.Image as Image
import numpy as np
from vgg16glide.utils.image_transform import ImageTransform
from vgg16glide.utils.image_transform_img import ImageTransformImg

class Data:
    def get(self, phase):
        pass

class ImageData(Data):
    def __init__(self, file_path, label, transform: ImageTransform):
        self.file_path = file_path
        self.label = label
        self.transform = transform

    def get(self, phase='train'):
        path = self.file_path
        img_np = np.loadtxt(path, delimiter=',')

        img_transformed = self.transform(img_np, phase)
        return img_transformed, self.label, path


class ImageDataImg(Data):
    def __init__(self, file_path, label, transform: ImageTransformImg):
        self.file_path = file_path
        self.label = label
        self.transform = transform

    def get(self, phase='train'):
        path = self.file_path
        img = Image.open(path)

        img_transformed = self.transform(img, phase)
        return img_transformed, self.label, path

class ImageDataset(data.Dataset):
    def __init__(self, csv_file_list: List[Data], phase='train'):
        self.csv_file_list = csv_file_list
        self.phase = phase

    def __len__(self):
        return len(self.csv_file_list)

    def __getitem__(self, index):
        return self.csv_file_list[index].get(phase=self.phase)
