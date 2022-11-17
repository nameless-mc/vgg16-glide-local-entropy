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


def main():
    target_step = 85
    model_epoch = 25
    batch_size = 32
    project = "vgg16-glide-local-entropy-map-0-255"
    model_path = f"models/{project}/{target_step}/model_{model_epoch}.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(False, device, model_path)

    transform = ImageTransform()

    dir_path = "dataset/dataset_yolov7_threshold_25"
    file_path = "entropy_heatmap/csv"
    true_file_list = []
    false_file_list = []
    true_file_path = os.path.join(
        dir_path, "0", "local_entropy", "*", file_path, str(target_step) + ".csv")
    false_file_path = os.path.join(
        dir_path, "1", "local_entropy", "*", file_path, str(target_step) + ".csv")

    for f in glob.glob(true_file_path):
        false_file_list.append(ImageData(f, 0, transform))
    for f in glob.glob(false_file_path):
        true_file_list.append(ImageData(f, 1, transform))

    true_dataset = ImageDataset(csv_file_list=true_file_list, phase='val')
    false_dataset = ImageDataset(csv_file_list=false_file_list, phase='val')

    true_dataloader = torch.utils.data.DataLoader(
        true_dataset, batch_size=batch_size, shuffle=False)
    false_dataloader = torch.utils.data.DataLoader(
        false_dataset, batch_size=batch_size, shuffle=False)

    true_outputs, true_preds, true_dict = classifier(
        model, true_dataloader, device)
    false_outputs, false_preds, false_dict = classifier(
        model, false_dataloader, device)

    acc = (true_preds.count(1) + false_preds.count(0)) / \
        (len(true_preds) + len(false_preds))
    tpr = true_preds.count(1) / len(true_preds)
    tnr = false_preds.count(0) / len(false_preds)
    print("ACC: ", acc)
    print("TPR: ", tpr)
    print("TNR: ", tnr)
    t_mean = np.mean(np.array(true_outputs), axis=0)
    f_mean = np.mean(np.array(false_outputs), axis=0)
    print("true_output_mean: ", t_mean)
    print("false_output_mean: ", f_mean)

    true_dict = change_dict_key(true_dict, true_file_path)
    false_dict = change_dict_key(false_dict, false_file_path)

    with open(os.path.join(dir_path, f"true_classifier_step_{target_step}.json"), "w") as f:
        json.dump(true_dict, f, indent=4, ensure_ascii=False, sort_keys=True)

    with open(os.path.join(dir_path, f"false_classifier_step_{target_step}.json"), "w") as f:
        json.dump(false_dict, f, indent=4, ensure_ascii=False, sort_keys=True)

    with open(os.path.join(dir_path, f"classifier_step_{target_step}.json"), "w") as f:
        json.dump(dict(**true_dict, **false_dict), f,
                  indent=4, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
