import torch
import torch.nn as nn
from torchsummary import summary

# ConvLayer
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bc1 = nn.BatchNorm2d(out_channels)
        self.bc2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.activation(self.bc1(self.conv1(x)))
        out = self.activation(self.bc2(self.conv2(out)))
        return out


# Unet 모델
class Unet(nn.Module):
    def __init__(self, input_channel=1, num_class=2):
        super(Unet, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        num_filter = [64, 128, 256, 512, 1024]

        self.up4 = nn.ConvTranspose2d(num_filter[4], num_filter[4], kernel_size=2, stride=2, padding=0, bias=False)
        self.up3 = nn.ConvTranspose2d(num_filter[3], num_filter[3], kernel_size=2, stride=2, padding=0, bias=False)
        self.up2 = nn.ConvTranspose2d(num_filter[2], num_filter[2], kernel_size=2, stride=2, padding=0, bias=False)
        self.up1 = nn.ConvTranspose2d(num_filter[1], num_filter[1], kernel_size=2, stride=2, padding=0, bias=False)
        # encoding
        self.encoded1 = Conv(input_channel, num_filter[0])
        self.encoded2 = Conv(num_filter[0], num_filter[1])
        self.encoded3 = Conv(num_filter[1], num_filter[2])
        self.encoded4 = Conv(num_filter[2], num_filter[3])
        self.encoded5 = Conv(num_filter[3], num_filter[4])

        self.decoded4 = Conv(num_filter[4] + num_filter[3], num_filter[3])
        self.decoded3 = Conv(num_filter[3] + num_filter[2], num_filter[2])
        self.decoded2 = Conv(num_filter[2] + num_filter[1], num_filter[1])
        self.decoded1 = Conv(num_filter[1] + num_filter[0], num_filter[0])

        self.final = nn.Conv2d(num_filter[0], num_class, kernel_size=3, stride=1, padding=1)

        # initialise weigths
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        e1 = self.encoded1(x)
        e2 = self.encoded2(self.pooling(e1))
        e3 = self.encoded3(self.pooling(e2))
        e4 = self.encoded4(self.pooling(e3))
        e5 = self.encoded5(self.pooling(e4))

        d4 = self.decoded4(torch.cat((e4, self.up4(e5)), dim=1))
        d3 = self.decoded3(torch.cat((e3, self.up3(d4)), dim=1))
        d2 = self.decoded2(torch.cat((e2, self.up2(d3)), dim=1))
        d1 = self.decoded1(torch.cat((e1, self.up1(d2)), dim=1))
        final = self.final(d1)
        return final

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_channel = 2
    num_classes = 1
    unet = Unet(input_channel,num_classes).to(device)
    summary(unet,(input_channel,241, 76))