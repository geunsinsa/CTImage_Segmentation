import torch
from torch.utils.data import Dataset
import albumentations as A

class CustomDataset(Dataset):
    def __init__(self, imgData, labelData, transform=None, num_classes=3, augmentation=None, phase='train'):
        self.imgData = imgData
        self.labelData = labelData
        self.transform = transform
        self.classes = num_classes
        self.augmentation = augmentation
        self.phase = phase

    def __len__(self):
        return len(self.imgData)

    def Augmentation(self, image, label):
        transform_A = A.Compose([
            A.RandomResizedCrop(width=512, height=512, scale=(0.8, 0.9), p=0.3),
            A.GaussNoise(var_limit=(0.05, 0.1), p=0.3),
            A.ElasticTransform(alpha=30, p=0.3)
        ])

        augmented = transform_A(image=image, mask=label)

        return augmented['image'], augmented['mask'][:,:,:1]
    def onehot_encode(self, label):
        shape = label.shape
        onehot_label = torch.zeros((self.classes, shape[1], shape[2]))

        for i in range(self.classes):
            indices = torch.where(label[0] == i)
            onehot_label[i][indices] = 1.0

        return onehot_label

    def __getitem__(self, idx):
        image = self.imgData[idx]
        label = self.labelData[idx]
        if self.phase == 'train':
            if self.augmentation == False:
                image = self.transform(image)
                label = self.transform(label)
                label = self.onehot_encode(label)
                return image, label
            else:
                image, label = self.Augmentation(image, label)
                image = self.transform(image)
                label = self.transform(label)
                label = self.onehot_encode(label)
                return image, label

        elif self.phase == 'val':
            image = self.transform(image)
            label = self.transform(label)
            label = self.onehot_encode(label)
            return image, label

        else:
            return image


