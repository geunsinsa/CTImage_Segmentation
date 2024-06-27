import os
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter

from make_slicing_dataset import sliced_dataset
from loss import CombinedLoss
from unet_model import Unet

def make_result_folder(path):
    folder_name = ['model','Loss','result','csvLogger']
    for name in folder_name:
      folder_path = f'{path}/{name}'
      if not os.path.exists(folder_path):
          os.makedirs(folder_path)

def morphological_bg_fg_nii(imgPath):
    data = np.load(imgPath)['data']
    # 각 slice를 모폴로지 연산을 적용하여 전경과 배경 추출 후 NumPy 배열로 변환
    result_slices = []
    for slice_idx in range(data.shape[0]):
        # 이미지 추출
        image = data[slice_idx]
        result = []
        for ch in range(image.shape[0]):
            # 이진화 수행
            _, binary_image = cv2.threshold(image[ch].astype(np.uint8), 70, 255, cv2.THRESH_BINARY)

            # 모폴로지 연산 수행
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

            # 전경과 배경 추출
            foreground = cv2.bitwise_and(image[ch], image[ch], mask=closing)
            result.append(foreground)
        fg_ch_images = np.stack(result)
        # 결과 추가
        result_slices.append(fg_ch_images)

    # 모든 slice를 하나의 NumPy 배열로 병합
    foreground_images = np.stack(result_slices)
    print("전경/배경 분리 완료 ...")
    return foreground_images
def load_dataset(path, train_num=40):
    numPatientCT = np.load(f'{path}/concat_idx.npz')['data']
    concatCT_IMG = ((np.load(f'{path}/concat_images.npz')['data']).astype(np.float32) / 255.0)
    concatCT_LBL = np.load(f'{path}/concat_labels.npz')['data'].astype(np.float32)

    divide_idx = sum(numPatientCT[:train_num])
    trainImage = np.transpose(concatCT_IMG[:divide_idx, :, :],(0,2,3,1))
    trainLabel = np.transpose(concatCT_LBL[:divide_idx, :, :],(0,2,3,1))
    if trainImage.shape[-1] > 1:
        trainLabel = np.tile(trainLabel, (1, 1, 1, trainImage.shape[-1]))
    valImage = np.transpose(concatCT_IMG[divide_idx:, :, :],(0,2,3,1))
    valLabel = np.transpose(concatCT_LBL[divide_idx:, :, :],(0,2,3,1))

    return trainImage, trainLabel, valImage, valLabel

def metrics_table(train_loss, val_loss, metrics, path):
    train_dice_coef = metrics[0]
    val_dice_coef = metrics[1]
    train_iou = metrics[2]
    val_iou = metrics[3]

    loggerDF = pd.DataFrame({'Train_Loss': train_loss, 'Val_Loss': val_loss, 'Train_Dice_Coefficient': train_dice_coef,
                             'Train_IOU_Coefficient': train_iou, 'Val_Dice_Coefficient': val_dice_coef,
                             'Val_IOU_Coefficient': val_iou}, index=range(1, len(train_dice_coef) + 1))

    loggerDF.to_csv(
        f'{path}/csvLogger/logger_{type(model).__name__}_{type(optimizer).__name__}_{type(criterion).__name__}_Aug_{aug}.csv')

def loss_plot(train_loss, val_loss, path):
    plt.figure(figsize=(7, 7))
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.title(f'{type(model).__name__}_{type(optimizer).__name__}_{type(criterion).__name__}')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{path}/Loss/{type(model).__name__}_{type(optimizer).__name__}_{type(criterion).__name__}_Aug_{aug}.png')

def dice_coef_plot(train_dice_coef, val_dice_coef, path):
    plt.figure(figsize=(7, 7))
    plt.plot(train_dice_coef, label='train_dice_coef')
    plt.plot(val_dice_coef, label='val_dice_coef')
    plt.title(f'{type(model).__name__} Dice Coefficient')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{path}/Loss/{type(model).__name__}_{type(optimizer).__name__}_{type(criterion).__name__}_DiceCoef_Aug_{aug}.png')

def iou_plot(train_iou_coef, val_iou_coef, path):
    plt.figure(figsize=(7, 7))
    plt.plot(train_iou_coef, label='train_iou')
    plt.plot(val_iou_coef, label='val_iou')
    plt.title(f'{type(model).__name__} IOU')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{path}/Loss/{type(model).__name__}_{type(optimizer).__name__}_{type(criterion).__name__}_IOU_Aug_{aug}.png')

def validation(model, criterion, val_loader, device):
    model.eval()

    val_loss = []
    val_dice_coef = []
    val_iou = []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):  # tqdm에 자체에 iteration 포함 iter 안해도 됨
            imgs = imgs.to(device)
            labels = labels.to(device)

            with amp.autocast():
                output = model(imgs)
                loss, dice_coef, iou = criterion(output, labels)

            loss = loss.cpu().detach().numpy().item()
            dice_coef = dice_coef.cpu().detach().numpy().item()
            iou = iou.cpu().detach().numpy().item()

            val_loss.append(loss)
            val_dice_coef.append(dice_coef)
            val_iou.append(iou)

            del imgs, labels, output, loss
        _val_loss = np.mean(val_loss)
        _val_dice_coef = np.mean(val_dice_coef)
        _val_iou = np.mean(val_iou)

    return _val_loss, _val_dice_coef, _val_iou

def train(path, batch_size, classes, epochs, model, criterion, optimizer, aug):
    make_result_folder(path) # fn
    trainImage, trainLabel, valImage, valLabel = load_dataset(path)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(trainImage, trainLabel, transform=transform, num_classes=classes, augmentation=aug, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(valImage, valLabel, transform=transform, num_classes=classes, augmentation=aug, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion.to(device)
    scaler = amp.GradScaler()

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    train_loss_logger = []
    val_loss_logger = []
    train_dice_coef_logger = []
    val_dice_coef_logger = []
    train_iou_logger = []
    val_iou_logger = []

    writer = SummaryWriter(log_dir=f"{path}/Log_Dir/")
    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = []
        train_dice_coef = []
        train_iou = []

        for imgs, labels in tqdm(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with amp.autocast():
                output = model(imgs)
                loss, dice_coef, iou = criterion(output, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss = loss.cpu().detach().numpy().item()
            dice_coef = dice_coef.cpu().detach().numpy().item()
            iou = iou.cpu().detach().numpy().item()

            train_loss.append(loss)
            train_dice_coef.append(dice_coef)
            train_iou.append(iou)

            # Explicityly delete for memory
            del imgs, labels, output, loss, dice_coef, iou
            torch.cuda.empty_cache()

        _val_loss, _val_dice_coef, _val_iou = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        _train_dice_coef = np.mean(train_dice_coef)
        _train_iou = np.mean(train_iou)

        print(f'Epoch [{epoch}], Train Loss: [{_train_loss}], Dice_coef: [{_train_dice_coef}], Iou: [{_train_iou}]')
        print(f'Epoch [{epoch}], Val Loss: [{_val_loss}], Dice: [{_val_dice_coef}], Iou : [{_val_iou}]')

        writer.add_scalar('Loss/Train', _train_loss, epoch)
        writer.add_scalar('Loss/Valid', _val_loss, epoch)
        writer.add_scalar('DiceCoef/Train', _train_dice_coef, epoch)
        writer.add_scalar('DiceCoef/Valid', _val_dice_coef, epoch)
        writer.add_scalar('IOU/Train', _train_iou, epoch)
        writer.add_scalar('IOU/Valid', _val_iou, epoch)

        train_loss_logger.append(_train_loss)
        val_loss_logger.append(_val_loss)
        train_dice_coef_logger.append(_train_dice_coef)
        val_dice_coef_logger.append(_val_dice_coef)
        train_iou_logger.append(_train_iou)
        val_iou_logger.append(_val_iou)

        # Save the best model
        if best_val_loss > _val_loss:
            best_val_loss = _val_loss
            print(best_val_loss)
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts,
                       f'{path}/model/{type(model).__name__}_bestest_model_{type(optimizer).__name__}_{type(criterion).__name__}_Aug_{aug}.pt')

    else:
        writer.flush()
        writer.close()
        torch.save(model.state_dict(),
                   f'{path}/model/{type(model).__name__}_latest_model_{type(optimizer).__name__}_{type(criterion).__name__}_Aug_{aug}.pt')

        return [model.state_dict(), best_model_wts, train_loss_logger, val_loss_logger, [train_dice_coef_logger,
                                                                                        val_dice_coef_logger,
                                                                                        train_iou_logger,
                                                                                        val_iou_logger]]




def main(args):
    # slice data 생성
    rawdata_path = 'data/FLARE22Train/'  # 원본 데이터
    savedata_path = args.savedata_path  # 변환 데이터 폴더 생성 및 위치
    huList = [[-260,240]]  # hounsfield value List
    organNum = [int(x) for x in args.organNum.split(',')]  # Organ Number List
    sliced_dataset(rawdata_path, savedata_path, huList, organNum)

    ch = len(huList)  # the number of channels
    data_path = savedata_path  # dataset path
    aug = args.aug  # Train Data Augmentation
    lr = args.lr  # learning rate
    batch_size = args.batch_size  # train batch size, val_batch_size Fix 1
    epochs = args.epochs  # Epoch
    classes = 1 + len(organNum)
    model = Unet(input_channel=ch, num_class=classes)  # Unet

    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Tensorboard 사용 가능 train함수 안에 경로 지정
    result = train(path=data_path, batch_size=batch_size, classes=classes, epochs=epochs, model=model, criterion=criterion, optimizer=optimizer, aug=aug)
    last_model_wts, best_model_wts, train_loss_logger, val_loss_logger, metrics = result
    metrics_table(train_loss_logger, val_loss_logger, metrics, data_path)  # save loss logger
    loss_plot(train_loss_logger, val_loss_logger, data_path)  #
    dice_coef_plot(metrics[0], metrics[1], data_path)
    iou_plot(metrics[2], metrics[3], data_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for Unet model.')
    parser.add_argument('--savedata_path', type=str, required=True, help='변환 데이터 폴더 생성 및 위치')
    parser.add_argument('--organNum', type=str, required=True, help='Organ Number List, 쉼표로 구분된 문자열로 입력')
    parser.add_argument('--aug', action='store_true', help='Train Data Augmentation')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Train batch size, val_batch_size Fix 1')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

    args = parser.parse_args()
    main(args)
