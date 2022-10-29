import glob
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data
from utils.dataset import ImageData, ImageDataset
from utils.image_transform import ImageTransform
from utils.model import create_model

def classifier(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    results = []
    preds = []
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(inputs)
        outputs = F.softmax(outputs, dim = 1)

        _, pred = torch.max(outputs, dim=1)
        results.extend(outputs.tolist())
        preds.extend(pred.tolist())
    return results, preds

def main():
    target_step = 85
    model_epoch = 25
    batch_size = 32
    project="vgg16-glide-local-entropy-map-0-255"
    target_image_dir = "dataset/wilderness_spreading_all_around_3308"
    print("target_image_dir:", target_image_dir)

    model_path = f"models/{project}/{target_step}/model_{model_epoch}.pth"
    target_file_path = f"{target_image_dir}/entropy_heatmap/csv/{target_step}.csv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(False, device, model_path)

    transform = ImageTransform()

    # data = ImageData(target_file_path, 1, transform)
    # dataset = ImageDataset(csv_file_list=[data], phase='val')
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=False)
    # outputs, labels = classifier(model, dataloader, device)
    # print("rate: ", outputs)
    # print("label: ", labels)

    dir_path = "dataset/val"
    file_path = "entropy_heatmap/csv"
    true_file_list = []
    false_file_list = []
    for f in glob.glob(os.path.join(dir_path, "0", "*", file_path, str(target_step) + ".csv")):
        false_file_list.append(ImageData(f, 0, transform))
    for f in glob.glob(os.path.join(dir_path, "1", "*", file_path, str(target_step) + ".csv")):
        true_file_list.append(ImageData(f, 1, transform))
        
    true_dataset = ImageDataset(csv_file_list=true_file_list, phase='val')
    false_dataset = ImageDataset(csv_file_list=false_file_list, phase='val')

    true_dataloader = torch.utils.data.DataLoader(
        true_dataset, batch_size=batch_size, shuffle=False)
    false_dataloader = torch.utils.data.DataLoader(
        false_dataset, batch_size=batch_size, shuffle=False)
        

    _, true_outputs = classifier(model, true_dataloader, device)
    _, false_outputs = classifier(model, false_dataloader, device)
    acc = (true_outputs.count(1) + false_outputs.count(0)) / (len(true_outputs) + len(false_outputs))
    tpr = true_outputs.count(1) / len(true_outputs)
    tnr = false_outputs.count(0) / len(false_outputs)
    print("ACC: ", acc)
    print("TPR: ", tpr)
    print("TNR: ", tnr)


if __name__ == "__main__":
    main()