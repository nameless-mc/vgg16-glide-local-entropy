import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

from vgg16glide.utils.dataset import ImageDataset, ImageDataImg
from vgg16glide.utils.image_transform import ImageTransform
import glob
import os
from tqdm import tqdm
from vgg16glide.utils.image_transform_img import ImageTransformImg
from vgg16glide.utils.model import create_model
import wandb

from vgg16glide.utils.dataset import ImageData
from vgg16glide.utils.image_transform import ImageTransform

def run(target_step):
    dir_path = "dataset/test_data/train"
    file_path = "image_steps"
    project_name = 'vgg16-color-step-images'

    num_epochs = 25

    file_list = []
    transform = ImageTransformImg()

    for f in glob.glob(os.path.join(dir_path, "0", "*", file_path, str(target_step) + ".png")):
        file_list.append(ImageDataImg(f, 0, transform))
    for f in glob.glob(os.path.join(dir_path, "1", "*", file_path, str(target_step) + ".png")):
        file_list.append(ImageDataImg(f, 1, transform))

    # ミニバッチのサイズを指定
    batch_size = 32

    # MakeDatasetで前処理後の訓練データと正解ラベルを取得
    dataset = ImageDataset(csv_file_list=file_list, phase='train')
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # データローダーをdictにまとめる
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = create_model(pretrained=True, device=device)

    # 損失関数
    criterion = nn.CrossEntropyLoss()
    # オプティマイザー
    optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # setup w & b
    wandb.init(project=project_name, name=f"target_step {target_step}")
    wandb.config.target_step = target_step
    wandb.config.epochs = num_epochs

    def train_model(model, dataloaders, criterion, optimizer, num_epochs):

        # epochの数だけ
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-------------')

            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0

            # 学習と検証のループ
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # モデルを訓練モードにする
                else:
                    model.eval()   # モデルを検証モードにする

                epoch_loss = 0.0    # 1エポックあたりの損失の和
                epoch_corrects = 0.0  # 1エポックあたりの精度の和

                # 1ステップにおける訓練用ミニバッチを使用した学習
                # tqdmでプログレスバーを表示する
                for inputs, labels, _ in tqdm(dataloaders[phase]):
                    # torch.Tensorオブジェクトにデバイスを割り当てる
                    inputs, labels = inputs.to(device), labels.to(device)

                    # オプティマイザーを初期化
                    optimizer.zero_grad()
                    # 順伝搬（forward）計算
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)  # モデルの出力を取得
                        # 出力と正解ラベルの誤差から損失を取得
                        loss = criterion(outputs, labels)
                        # 出力された要素数2のテンソルの最大値を取得
                        _, preds = torch.max(outputs, dim=1)
                        # 訓練モードではバックプロパゲーション
                        if phase == 'train':
                            loss.backward()  # 逆伝播の処理(自動微分による勾配計算)
                            optimizer.step()  # 勾配降下法でバイアス、重みを更新

                        # ステップごとの損失を加算、inputs.size(0)->32
                        epoch_loss += loss.item() * inputs.size(0)
                        # ステップごとの精度を加算
                        epoch_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                # エポックごとの損失と精度を表示
                epoch_loss = epoch_loss / len(dataloaders[phase].dataset)
                epoch_acc = epoch_corrects.double(
                ) / len(dataloaders[phase].dataset)

                # 出力
                print('{} - loss: {:.4f} - acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if phase == 'train':
                    epoch_train_loss = epoch_loss
                    epoch_train_acc = epoch_acc
                else:
                    epoch_val_loss = epoch_loss
                    epoch_val_acc = epoch_acc

                if (epoch+1) % 5 == 0:
                    model_path = f'model_{epoch+1}.pth'
                    dir = f"models/{project_name}/{target_step}"
                    os.makedirs(dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(dir, model_path))

            wandb.log({'train_loss': epoch_train_loss, "train_acc": epoch_train_acc,
                    "val_loss": epoch_val_loss, "val_acc": epoch_val_acc})

    train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs)

    wandb.finish()


for i in reversed(range(0, 99, 3)):
    print("step:", i)
    run(i)
