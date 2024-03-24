# CT Image Segmentation Tutorial for Beginner
Code for those who are new to segmentation using pytorch

## Requirments
- torch : 2.2.0+cu121
- torchvision : 0.17.0+cu121
- torchsummary : 1.5.1
- tqdm : 4.66.2
- numpy : 1.26.3
- pandas : 2.2.1

## Dataset
- Download [link](https://zenodo.org/records/7860267)
![initial](https://rumc-gcorg-p-public.s3.amazonaws.com/i/2022/03/29/20220309-FLARE22-Pictures-2.png)

## Train Method
1. Download dataset -> decompression
2. Download code.zip -> decompression
3. Generator Folder
- root(CTImage_Segmentation)/data/raw_dataset/images
  - Insert dataset images file .gz
- root(CTImage_Segmentation)/data/raw_dataset/labels
  - Insert dataset labels file .gz
- root(CTImage_Segmentation)/Experiment
4. Set parameters of train.py
  ```
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
    model = FUSeg(input_channel=ch, num_class=classes)
    criterion = CombinedLoss() # CombinedLoss or DiceLoss : Hard Dice Loss, loss.py
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # Adam, SGD, RMSprop : Select Optimizer


    # Tensorboadrd 사용 가능 train함수 안에 경로 지정
    result = train(path=data_path, batch_size=batch_size, classes=classes, epochs=epochs, model=model, criterion=criterion, optimizer=optimizer)
    last_model_wts, best_model_wts, train_loss_logger, val_loss_logger, metrics = result
    metrics_table(train_loss_logger, val_loss_logger, metrics, data_path) # save loss logger
    loss_plot(train_loss_logger, val_loss_logger, data_path) #
    dice_coef_plot(metrics[0], metrics[1], data_path)
    iou_plot(metrics[2], metrics[3],data_path)
  ```
