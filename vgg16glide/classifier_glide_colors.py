import glob
import json
import os
import torch
import torch.utils.data
from vgg16glide.classifier import classifier
from vgg16glide.utils.dataset import ImageData, ImageDataImg, ImageDataset
from vgg16glide.utils.image_transform import ImageTransform
from vgg16glide.utils.image_transform_img import ImageTransformImg
from vgg16glide.utils.model import create_model
import numpy as np


def get_img_name_from_path(key: str):
    return key.split("/")[-3]


def change_dict_key(d):
    nd = {}
    for key in d.keys():
        new_key = get_img_name_from_path(key)
        nd[new_key] = d[key]
    return nd


def main(target_step):
    model_epoch = 5
    batch_size = 32
    project = "vgg16-color-step-images"
    model_path = f"models/{project}/{target_step}/model_{model_epoch}.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(False, device, model_path)

    transform = ImageTransformImg()

    dir_path = "dataset"
    file_list = []
    file_path = os.path.join(f"{dir_path}/result_for_humans/img/*/steps/*/image_steps/{target_step}.png")
    for f in glob.glob(file_path):
        file_list.append(ImageDataImg(f, 0, transform))

    dataset = ImageDataset(csv_file_list=file_list, phase='val')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    _, _, d = classifier(
        model, dataloader, device)

    d = change_dict_key(d)

    with open(os.path.join(dir_path, f"classifier_color_step_{target_step}.json"), "w") as f:
        json.dump(d, f, indent=4, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    steps = [95, 90, 85, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    for step in steps:
        main(step)
