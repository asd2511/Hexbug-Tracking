import torch
import torch.nn as nn
from torch.autograd import Variable


class CONV(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, h=None):
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, recurrent=False, residual=False):
        super().__init__()
        convBlock = CONV if not recurrent else ConvSRU
        self.maxPool = nn.MaxPool2d(2)
        self.conv = convBlock(in_channels, out_channels)

        self.residual = residual
        if residual:
            self.shortCut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=2, kernel_size=1, padding=0, bias=False)
    def forward(self, x, h = None):
        if not self.residual:
            # return self.conv(x)
            return self.conv(self.maxPool(x), h=h)
        else:
            return self.conv(self.maxPool(x), h=h) + self.shortCut(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, recurrent=False, residual=False):
        super().__init__()
        self.upConv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        convBlock = CONV if not recurrent else ConvSRU
        self.conv = convBlock(in_channels, out_channels)
        self.residual = residual
        if residual:
            self.shortCut = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels,stride=1, kernel_size=1, padding=0, bias=False)
    def forward(self, x, connection, h=None):
        x = self.upConv(x)
        if not self.residual:
            return self.conv(torch.cat([x,connection],dim=1), h=h)
        else:
            return self.conv(torch.cat([x,connection],dim=1), h=h) + self.shortCut(x)

class ConvDRU(nn.Module):
    def __init__(self,
                input_size,
                hidden_size,
                ):
        super(ConvDRU, self).__init__()

        self.reset_gate = CONV(input_size, input_size)
        self.update_gate = CONV(input_size, hidden_size)
        self.out_gate = CONV(input_size, hidden_size)

    def forward(self, input_, h=None):
        update = torch.sigmoid(self.update_gate(input_))
        reset = torch.sigmoid(self.reset_gate(input_))

        out_inputs = torch.tanh(self.out_gate(input_ * reset))
        if h is None:
            h = torch.zeros_like(update).cuda()
        h_new = h * (1 - update) + out_inputs * update
        return h_new

class ConvSRU(nn.Module):
    def __init__(self,
                input_size,
                hidden_size,
                ):
        super(ConvSRU, self).__init__()

        self.update_gate = CONV(input_size, hidden_size)
        self.out_gate = CONV(input_size, hidden_size)


    def forward(self, input_, h=0):
        update = torch.sigmoid(self.update_gate(input_))
        out_inputs = torch.tanh(self.out_gate(input_))
        if h is None:
            h = torch.zeros_like(update).cuda()
        h_new = h * (1 - update) + out_inputs * update
        return h_new


class Unet(nn.Module):
    def __init__(self,  n_channels=1, n_classes=1, recurrent=False, residual=False):
        super().__init__()
        self.typeFlag = type
        # fullSize
        self.down0, self.down1, self.down2, self.down3, self.down4, \
            self.up4, self.up3, self.up2, self.up1, self.up0 = \
            self.initNet(n_channels, recurrent, residual, n_classes)
        self.sigmoid = torch.nn.Sigmoid()


    def initNet(self,n_channels, recurrent, residual, n_classes):
        down0 = CONV(n_channels, 64)
        down1 = DownBlock( 64, 128, recurrent, residual)
        down2 = DownBlock(128, 256, recurrent, residual)
        down3 = DownBlock(256, 512, recurrent, residual)
        down4 = DownBlock(512, 1024, recurrent, residual)
        up4 = UpBlock(1024,512, False, residual)
        up3 = UpBlock(512,256, False, residual)
        up2 = UpBlock(256,128, False, residual)
        up1 = UpBlock(128, 64, False, residual)
        up0 = nn.Conv2d(64, n_classes, kernel_size=1)
        return down0,down1,down2,down3,down4,up4,up3,up2,up1,up0

    def forward(self, x):
        x1 = self.down0(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5,x4)
        x = self.up3(x,x3)
        x = self.up2(x,x2)
        x = self.up1(x,x1)
        x = self.up0(x)
        return x