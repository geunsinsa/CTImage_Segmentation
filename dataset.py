import torch# CustomDataset
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, imgData, labelData, transform=None, num_classes=3):
        self.imgData = imgData
        self.labelData = labelData
        self.transform = transform
        self.classes = num_classes

    def __len__(self):
        return len(self.imgData)

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
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
            label = self.onehot_encode(label)
            return image, label
        else:
            return image