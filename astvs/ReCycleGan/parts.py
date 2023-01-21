#!/bin/python3
# -*- coding: utf-8 -*-

import os
import warnings
warnings.simplefilter('ignore')
import glob
import time
from utils.types import *

class ReplayBuffer():
    def __init__(self, max_size=50, device: torch.device='cpu'):
        self.max_size = max_size
        self.device = device
        self.data = []

    def push_and_pop(self, data):
        data = data.detach().cpu()
        to_return = []
        for element in data.data:
            #
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.autograd.Variable(torch.cat(to_return)).detach().to(self.device)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def LambdaLR(epochs, offset, decayStartEpoch):
    def step(epoch):
        val = max(0., epoch + offset - decayStartEpoch) / (epochs - decayStartEpoch)
        return 1. - val
    return step

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class ConvUnit(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvUnit(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1,
                                     [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DownLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            ConvUnit(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Generator(torch.nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, 64, 7),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(256),
            torch.nn.ReLU(inplace=True),

            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(64, output_nc, 7),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(torch.nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(128, 256, 4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(256, 512, 4, padding=1),
            torch.nn.InstanceNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(512, 1, 4, padding=1)
        )
        self.flatten = Flatten()

    def forward(self, x):
        x = self.model(x)
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:])
        return self.flatten(x)

class Predictor(torch.nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Predictor, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.inc = ConvUnit(input_nc, 64)
        self.DownLayer1 = DownLayer(64, 128)
        self.DownLayer2 = DownLayer(128, 256)
        self.DownLayer3 = DownLayer(256, 512)
        self.DownLayer4 = DownLayer(512, 512)
        self.UpLayer1 = UpLayer(1024, 256)
        self.UpLayer2 = UpLayer(512, 128)
        self.UpLayer3 = UpLayer(256, 64)
        self.UpLayer4 = UpLayer(128, 64)
        self.outc = OutConv(64, output_nc)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.DownLayer1(x1)
        x3 = self.DownLayer2(x2)
        x4 = self.DownLayer3(x3)
        x5 = self.DownLayer4(x4)
        x = self.UpLayer1(x5, x4)
        x = self.UpLayer2(x, x3)
        x = self.UpLayer3(x, x2)
        x = self.UpLayer4(x, x1)
        logits = self.outc(x)
        out = self.tanh(logits)
        return out


import random
from PIL import Image
import torch.utils.data
class FaceDatasetSquence(torch.utils.data.Dataset):
    def __init__(
            self,
            inputs: str,
            transform: Transforms=None,
            unaligned: bool=False,
            skip: int=0
    ):

        self.INPUTPATH = inputs
        self.TRANSFORM = transform if transform is not None else lambda x: x
        self.unaligned = unaligned
        self.skip = skip
        self.remove_num = (skip + 1) * 2

        framesA, framesB = [], []
        for dir in sorted(os.listdir(os.path.join(self.INPUTPATH, "DomainA"))):
            framesA += sorted(glob.glob(os.path.join(self.INPUTPATH, "DomainA", dir, "*.png")))[:-self.remove_num]
        for dir in sorted(os.listdir(os.path.join(self.INPUTPATH, "DomainB"))):
            framesB += sorted(glob.glob(os.path.join(self.INPUTPATH, "DomainB", dir, "*.png")))[:-self.remove_num]

        domainALen, domainBLen = len(framesA), len(framesB)
        self.framesA, self.framesB = framesA, framesB
        self.domainALen, self.domainBLen = domainALen, domainBLen
        self.count = 1234

    def GetSquentialData(self, file1, seed):
        file, exe = os.path.splitext(file1)
        dir, file = file.rsplit('/', 1)
        file2 = os.path.join(dir, f'{int(file) + self.skip:0=10}.png')
        file3 = os.path.join(dir, f'{int(file) + self.skip * 2:0=10}.png')
        random.seed(seed)
        if self.TRANSFORM:
            file1 = self.TRANSFORM(Image.open(file1).convert('RGB'))
            random.seed(seed)
            file2 = self.TRANSFORM(Image.open(file2).convert('RGB'))
            random.seed(seed)
            file3 = self.TRANSFORM(Image.open(file3).convert('RGB'))
            return file1, file2, file3
        file1 = file1
        random.seed(seed)
        file2 = file2
        random.seed(seed)
        file3 = file3
        return file1, file2, file3

    def __getitem__(self, item):
        seed = self.count
        filesA = self.framesA[item % self.domainALen]
        A1, A2, A3 = self.GetSquentialData(filesA, seed)

        filesB = self.framesB[np.random.randint(0, self.domainBLen - 1)] if self.unaligned else self.framesB[item % self.domainBLen]
        B1, B2, B3 = self.GetSquentialData(filesB, seed + 1)
        self.count += 1
        return OrderedDict({'A1': A1, 'A2': A2, 'A3': A3, 'B1': B1, 'B2': B2, 'B3': B3})

    def __len__(self):
        return max(self.domainALen, self.domainBLen)

class FaceDatasetVideo(torch.utils.data.Dataset):
    def __init__(
            self,
            inputs: str,
            transform: Transforms=None,
            unaligned: bool=False,
            skip: int=0
    ):

        self.INPUTPATH = inputs
        self.transform = transform if transform is not None else lambda x: x
        self.unaligned = unaligned
        self.skip = skip
        self.remove_num = (skip + 1) * 2
        domainFrames = []
        for dir in sorted(os.listdir(self.INPUTPATH)):
            domainFrames += sorted(glob.glob(os.path.join(self.INPUTPATH, dir, '*.png')))[:-self.remove_num]
        self.domainFrames = domainFrames
        self.filesNum = len(domainFrames)

    def GetSquentialData(self, file1, seed):
        file, exe = os.path.splitext(file1)
        dir, file = file.rsplit('/', 1)
        file2 = os.path.join(dir, f'{int(file) + self.skip:0=10}.png')
        file3 = os.path.join(dir, f'{int(file) + self.skip * 2:0=10}.png')
        random.seed(seed)
        if self.transform:
            file1 = self.transform(Image.open(file1).convert('RGB'))
            random.seed(seed)
            file2 = self.transform(Image.open(file2).convert('RGB'))
            random.seed(seed)
            file3 = self.transform(Image.open(file3).convert('RGB'))
            return file1, file2, file3
        file1 = file1
        random.seed(seed)
        file2 = file2
        random.seed(seed)
        file3 = file3
        return file1, file2, file3

    def __getitem__(self, item):
        file = self.domainFrames[item % self.filesNum]
        seed = 123
        item1, item2, item3 = self.GetSquentialData(file, seed)
        return OrderedDict({'1': item1, '2': item2, '3': item3})

    def __len__(self):
        return self.filesNum

if __name__ == "__main__":
    FaceDatasetSquence(inputs='/DATABASE/TrainData')
    FaceDatasetVideo(inputs='/DATABASE/TrainData/DomainA')