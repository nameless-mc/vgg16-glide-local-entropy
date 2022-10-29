import torch.nn as nn
import torchvision
import torch

def create_model(pretrained = False, device = None, model_path = None) -> nn.Module:
    model = torchvision.models.vgg16(pretrained)
    model.classifier[6] = nn.Linear(
    in_features=4096,  # 入力サイズはデフォルトの4096
    out_features=2)   # 出力はデフォルトの1000から2に変更

    if(model_path is not None):
        model.load_state_dict(torch.load(model_path))
    if(device is not None):
        model = model.to(device)
    return model
