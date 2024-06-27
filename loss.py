import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def dice_loss(self,predicted, target, smooth=1e-10):
        predicted = torch.where(predicted > 0.5, 1, 0)
        intersection = torch.sum(target*predicted, dim=(2,3)) # 겹치는 부분
        gt = torch.sum(target, dim=(2,3)) # ground truth
        pred = torch.sum(predicted, dim=(2,3)) # predict
        total = gt + pred
        union = total - intersection # 합집합

        dice = (2.*intersection+smooth)/(total+smooth) # dice coefficient
        iou = ((intersection+smooth) /(union+smooth)).mean() # iou value
        dice_loss= 1 - dice.mean() # dice loss

        return dice_loss, iou

    def forward(self, inputs, targets):
        predicted = torch.nn.functional.softmax(inputs, dim=1)
        predicted = predicted[:,1:]
        targets = targets[:,1:]
        dice_loss, iou = self.dice_loss(predicted, targets) # Dice Loss, Iou Value
        return dice_loss, 1-dice_loss, iou

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bcelogitloss = nn.BCEWithLogitsLoss()

    def dice_loss(self, predicted, target, smooth=1e-10):
        predicted = torch.where(predicted > 0.5, 1, 0)
        intersection = torch.sum(target * predicted, dim=(2, 3))  # 겹치는 부분
        gt = torch.sum(target, dim=(2, 3))  # ground truth
        pred = torch.sum(predicted, dim=(2, 3))  # predict
        total = gt + pred
        union = total - intersection  # 합집합

        dice = (2. * intersection + smooth) / (total + smooth)  # dice coefficient
        iou = ((intersection + smooth) / (union + smooth)).mean()  # iou value
        dice_loss = 1 - dice.mean()  # dice loss

        return dice_loss, iou

    def forward(self, inputs, targets):
        predicted = torch.nn.functional.softmax(inputs, dim=1)
        predicted = predicted[:, 1:]
        inputs = inputs[:, 1:]
        targets = targets[:, 1:]

        bcelogit_loss = self.bcelogitloss(inputs, targets)  # BCE Loss 계산
        dice_loss, iou = self.dice_loss(predicted, targets)  # Dice Loss, Iou Value

        # BCE Loss + Dice Loss
        combined_loss = dice_loss + bcelogit_loss

        return combined_loss, 1 - dice_loss, iou
