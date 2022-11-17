import glob
import json
import os
import torch
import torch.utils.data
from vgg16glide.classifier import classifier
from vgg16glide.utils.dataset import ImageData, ImageDataset
from vgg16glide.utils.image_transform import ImageTransform
from vgg16glide.utils.model import create_model
import numpy as np


def get_img_name_from_path(path_rg, key):
    sp = path_rg.split("*")
    assert len(sp) == 2
    return key[len(sp[0]):-len(sp[1])]


def change_dict_key(d, path_rg):
    nd = {}
    for key in d.keys():
        new_key = get_img_name_from_path(path_rg, key)
        nd[new_key] = d[key]
    return nd


def main(target_step):
    model_epoch = 25
    batch_size = 32
    project = "vgg16-glide-local-entropy-map-0-255"
    model_path = f"models/{project}/{target_step}/model_{model_epoch}.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(False, device, model_path)

    transform = ImageTransform()

    dir_path = "dataset/dataset"
    file_path = "entropy_heatmap/csv"
    file_list = []
    file_path = os.path.join(
        dir_path, "local_entropy", "*", file_path, str(target_step) + ".csv")
    for f in glob.glob(file_path):
        file_list.append(ImageData(f, 0, transform))

    dataset = ImageDataset(csv_file_list=file_list, phase='val')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    _, _, d = classifier(
        model, dataloader, device)

    d = change_dict_key(d, file_path)

    with open(os.path.join(dir_path, f"classifier_step_{target_step}.json"), "w") as f:
        json.dump(d, f, indent=4, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    steps = [95, 90, 85, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    for step in steps:
        main(step)
