import os
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter

from make_slicing_dataset import sliced_dataset
from loss import CombinedLoss, DiceLoss
from unet_model import Unet
from segnet_model import SegNet
def make_result_folder(path):
    folder_name = ['model','Loss','result','csvLogger']
    for name in folder_name:
      folder_path = f'{path}/{name}'
      if not os.path.exists(folder_path):
          os.makedirs(folder_path)

def load_dataset(path, train_num=40):
    numPatientCT = np.load(f'{path}/concat_idx.npz')['data']
    concatCT_IMG = ((np.load(f'{path}/concat_images.npz')['data']).astype(np.float16) / 255.0)
    concatCT_LBL = np.load(f'{path}/concat_labels.npz')['data'].astype(np.float16)

    divide_idx = sum(numPatientCT[:train_num])
    trainImage = np.transpose(concatCT_IMG[:divide_idx, :, :],(0,2,3,1))
    trainLabel = np.transpose(concatCT_LBL[:divide_idx, :, :],(0,2,3,1))

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
        f'{path}/csvLogger/logger_{type(model).__name__}_{type(optimizer).__name__}_{type(criterion).__name__}.csv')

def loss_plot(train_loss, val_loss, path):
    plt.figure(figsize=(7, 7))
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.title(f'{type(model).__name__}_{type(optimizer).__name__}_{type(criterion).__name__}')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{path}/Loss/{type(model).__name__}_{type(optimizer).__name__}_{type(criterion).__name__}.png')

def dice_coef_plot(train_dice_coef, val_dice_coef, path):
    plt.figure(figsize=(7, 7))
    plt.plot(train_dice_coef, label='train_dice_coef')
    plt.plot(val_dice_coef, label='val_dice_coef')
    plt.title(f'{type(model).__name__} Dice Coefficient')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{path}/Loss/{type(model).__name__}_{type(optimizer).__name__}_{type(criterion).__name__}_DiceCoef.png')

def iou_plot(train_iou_coef, val_iou_coef, path):
    plt.figure(figsize=(7, 7))
    plt.plot(train_iou_coef, label='train_iou')
    plt.plot(val_iou_coef, label='val_iou')
    plt.title(f'{type(model).__name__} IOU')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f'{path}/Loss/{type(model).__name__}_{type(optimizer).__name__}_{type(criterion).__name__}_IOU.png')

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

def train(path, batch_size, classes, epochs, model, criterion, optimizer):
    make_result_folder(path) # fn
    trainImage, trainLabel, valImage, valLabel = load_dataset(path) # fn

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(trainImage, trainLabel, transform=transform, num_classes=classes) # dataset.py
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(valImage, valLabel, transform=transform, num_classes=classes)
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

    writer = SummaryWriter(log_dir=f"{data_path}/Log_Dir/")
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
                       f'{path}/model/{type(model).__name__}_bestest_model_{type(optimizer).__name__}_{type(criterion).__name__}.pt')

    else:
        writer.flush()
        writer.close()
        torch.save(model.state_dict(),
                   f'{path}/model/{type(model).__name__}_latest_model_{type(optimizer).__name__}_{type(criterion).__name__}.pt')

        return [model.state_dict(), best_model_wts, train_loss_logger, val_loss_logger, [train_dice_coef_logger,
                                                                                        val_dice_coef_logger,
                                                                                        train_iou_logger,
                                                                                        val_iou_logger]]






if __name__ == '__main__':
    # slice data 생성
    rawdata_path = 'data/raw_dataset/' # 원본 데이터
    savedata_path = 'Experiment/single_spleen_dataset/' # 변환 데이터 폴더 생성 및 위치
    huList = [[40, 60]] # hounsfield value Listist
    organNum = [3] # Organ Number List
    sliced_dataset(rawdata_path, savedata_path, huList, organNum)


    ch = len(huList) # the number of channels
    data_path = savedata_path  # dataset path
    lr = 0.0001 # learning rate
    batch_size = 4 # train batch size, val_batch_size Fix 1
    epochs = 2 # Epoch
    classes = 1 + len(organNum) # bg + organ # Model Output channel
    # model = Unet(input_channel=ch, num_class=classes) # Unet
    model = SegNet(input_channel=ch, num_class=classes)
    criterion = CombinedLoss() # CombinedLoss or DiceLoss : Hard Dice Loss, loss.py
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # Adam, SGD, RMSprop : Select Optimizer


    # Tensorboadrd 사용 가능 train함수 안에 경로 지정
    result = train(path=data_path, batch_size=batch_size, classes=classes, epochs=epochs, model=model, criterion=criterion, optimizer=optimizer)
    last_model_wts, best_model_wts, train_loss_logger, val_loss_logger, metrics = result
    metrics_table(train_loss_logger, val_loss_logger, metrics, data_path) # save loss logger
    loss_plot(train_loss_logger, val_loss_logger, data_path) #
    dice_coef_plot(metrics[0], metrics[1], data_path)
    iou_plot(metrics[2], metrics[3],data_path)

