from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np


class ImageTransform():

    def __init__(self):

        # dicに訓練用、検証用のトランスフォーマーを生成して格納
        self.data_transform = {
            'train': transforms.Compose([
                Map(0, 10, 0, 255),
                NdarrayToPillImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize(224),      # リサイズ
                transforms.CenterCrop(224),  # 画像中央をresize×resizeでトリミング
                transforms.ToTensor(),          # テンソルに変換
            ]),
            'val': transforms.Compose([
                Map(0, 10, 0, 255),
                NdarrayToPillImage(),
                transforms.Resize(224),      # リサイズ
                transforms.CenterCrop(224),  # 画像中央をresize×resizeでトリミング
                transforms.ToTensor(),          # テンソルに変換
            ])

        }

    def __call__(self, img: np.ndarray, phase='train'):

        return self.data_transform[phase](img)  # phaseはdictのキー


class NdarrayToPillImage(torch.nn.Module):
    def forward(self, img: np.ndarray):
        return Image.fromarray(img).convert("L").convert("RGB")


class Map(torch.nn.Module):

    def __init__(self, input_min, input_max, output_min=0, output_max=1):
        super().__init__()
        self.input_min = input_min
        self.input_max = input_max
        self.output_min = output_min
        self.output_max = output_max

    def forward(self, img: np.ndarray):
        def func(x):
            res = (x-self.input_min)/(self.input_max-self.input_min)
            res = res * (self.output_max - self.output_min)
            return res + self.output_min

        return np.vectorize(func)(img)
