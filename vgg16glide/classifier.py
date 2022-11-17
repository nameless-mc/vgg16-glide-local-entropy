import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data
import tqdm


def classifier(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    results = []
    preds = []
    dict = {}
    for inputs, _, paths in tqdm.tqdm(dataloader):
        inputs = inputs.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)

        _, pred = torch.max(outputs, dim=1)

        results.extend(outputs.tolist())
        preds.extend(pred.tolist())
        d = {path: {"pred": pred, "output": output.tolist()}
             for (output, pred, path) in zip(outputs, preds, paths)}
        dict = {**dict, **d}
    return results, preds, dict
