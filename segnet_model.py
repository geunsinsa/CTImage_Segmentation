import torch
import torch.nn as nn
from torchsummary import summary

# ConvLayer
class Conv2(nn.Module):
    def __init__(self, in_channels, out_channels, BN_momentum=0.5, phase='encode'):
        super(Conv2, self).__init__()
        if phase == 'encode':
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bc1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bc2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.activation = nn.ReLU(inplace=True)
        elif phase == 'decode':
            self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bc1 = nn.BatchNorm2d(in_channels, momentum=BN_momentum)
            self.bc2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.activation(self.bc1(self.conv1(x)))
        out = self.activation(self.bc2(self.conv2(out)))
        return out

class Conv3(nn.Module):
    def __init__(self, in_channels, out_channels, BN_momentum=0.5, phase='encode'):
        super(Conv3, self).__init__()
        if phase == 'encode':
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bc1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bc2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bc3 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.activation = nn.ReLU(inplace=True)
        elif phase == 'decode':
            self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bc1 = nn.BatchNorm2d(in_channels, momentum=BN_momentum)
            self.bc2 = nn.BatchNorm2d(in_channels, momentum=BN_momentum)
            self.bc3 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.activation(self.bc1(self.conv1(x)))
        out = self.activation(self.bc2(self.conv2(out)))
        out = self.activation(self.bc3(self.conv3(out)))
        return out


# Unet 모델
class SegNet(nn.Module):
    def __init__(self, input_channel=3, num_class=2):
        super(SegNet, self).__init__()
        self.encode_pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.decode_pooling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        num_filter = [64, 128, 256, 512]

        # encoding
        self.encoded1 = Conv2(input_channel, num_filter[0], phase='encode')
        self.encoded2 = Conv2(num_filter[0], num_filter[1], phase='encode')
        self.encoded3 = Conv3(num_filter[1], num_filter[2], phase='encode')
        self.encoded4 = Conv3(num_filter[2], num_filter[3], phase='encode')
        self.encoded5 = Conv3(num_filter[3], num_filter[3], phase='encode')

        self.decoded5 = Conv3(num_filter[3], num_filter[3], phase='decode')
        self.decoded4 = Conv3(num_filter[3], num_filter[2], phase='decode')
        self.decoded3 = Conv3(num_filter[2], num_filter[1], phase='decode')
        self.decoded2 = Conv2(num_filter[1], num_filter[0], phase='decode')
        self.decoded1 = Conv2(num_filter[0], num_class, phase='decode')
        self.final = nn.Conv2d(num_class, num_class, kernel_size=3, stride=1, padding=1, bias=False)

        # initialise weigths
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        e1 = self.encoded1(x)
        e1, ind1 = self.encode_pooling(e1) # value, Max value indice
        size1 = e1.size()

        e2 = self.encoded2(e1)
        e2, ind2 = self.encode_pooling(e2)
        size2 = e2.size()

        e3 = self.encoded3(e2)
        e3, ind3 = self.encode_pooling(e3)
        size3 = e3.size()

        e4 = self.encoded4(e3)
        e4, ind4 = self.encode_pooling(e4)
        size4 = e4.size()

        e5 = self.encoded5(e4)
        e5, ind5 = self.encode_pooling(e5)

        d5 = self.decode_pooling(e5, ind5, output_size=size4)
        d5 = self.decoded5(d5)

        d4 = self.decode_pooling(d5, ind4, output_size=size3)
        d4 = self.decoded4(d4)

        d3 = self.decode_pooling(d4, ind3, output_size=size2)
        d3 = self.decoded3(d3)

        d2 = self.decode_pooling(d3, ind2, output_size=size1)
        d2 = self.decoded2(d2)

        d1 = self.decode_pooling(d2, ind1)
        d1 = self.decoded1(d1)

        final = self.final(d1)

        return final


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_channel = 3
    num_classes = 10
    SegNet = SegNet(input_channel,num_classes).to(device)
    summary(SegNet,(input_channel,512,512))
