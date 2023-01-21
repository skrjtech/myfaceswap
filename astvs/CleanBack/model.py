#!/bin/python3
# -*- coding: utf-8 -*-
import os
import warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "default"
import cv2
import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from utils.types import CaptureT
from Argparse import MakedataArgs, EvalArgs, PluginArgs

def GetChannels(capIdx: CaptureT) -> int:
    cap = cv2.VideoCapture(capIdx)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            w, h, c = frame.shape
            return c
    return -1

class CaptureBase(object):
    def __init__(self, capIdx: CaptureT) -> None:
        CHANNELS = GetChannels(capIdx)
        if CHANNELS < 0:
            assert f"{capIdx}のファイルを正しく開くことができませんでした"
        self.Capture = cv2.VideoCapture(capIdx)
        self.MAXWIDTH = int(self.Capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.MAXHEIGHT = int(self.Capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.MAXFRAMES = int(self.Capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.MAXCHANNELS = CHANNELS
        self.Zeros = np.zeros((self.MAXWIDTH, self.MAXHEIGHT, self.MAXCHANNELS), dtype=np.uint8)
    
    def Read(self):
        if self.Capture.isOpened():
            ret, frame = self.Capture.read()
            if not ret:
                frame = self.Zeros.copy()
            return frame
        return self.Zeros.copy()

class VideoFileProcessing(CaptureBase):
    def __init__(self, capIdx: CaptureT, args) -> None:
        super().__init__(capIdx)
        CHANNELS = GetChannels(capIdx)
        if CHANNELS < 0:
            assert f"{capIdx}のファイルを正しく開くことができませんでした"
        self.Capture = cv2.VideoCapture(capIdx)
        self.MAXWIDTH = int(self.Capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.MAXHEIGHT = int(self.Capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.MAXFRAMES = int(self.Capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.MAXCHANNELS = CHANNELS

        self.Getbuffer = self._GetBuffer2
        try:
            if args.MULTIFRAMES:
                # 2GBずつ取り込む
                self.MAXLIMIT = 2 / ((self.MAXWIDTH * self.MAXHEIGHT * (self.MAXCHANNELS * 8)) / 8 * 10e-9)
                self.RANGE = int(self.MAXFRAMES // self.MAXLIMIT) 
                self.SPLITFRAMES = [int(self.MAXLIMIT) for _ in range(self.RANGE)] + [(self.MAXFRAMES - (self.MAXLIMIT - int(self.MAXLIMIT)))]
                self.Getbuffer = self._GetBuffer1
        except:
            pass
    
    def Getbuffer(self):
        pass

    # for multiframe
    def _GetBuffer1(self):
        for split in self.SPLITFRAMES:
            frames = []
            for f in range(split):
                frames.append(
                    self.Read()
                )
            yield frames
    
    # one frame
    def _GetBuffer2(self):
        for f in range(self.MAXFRAMES):
            yield self.Read()

class Resize(torchvision.transforms.Lambda):
    def __init__(self, WIDTH: int, HEIGHT: int):
        super(Resize, self).__init__(lambd=lambda x: cv2.resize(x, (WIDTH, HEIGHT)))
    
class Base(object):
    def __init__(self, args, WIDTH: int=320, HEIGHT=320) -> None:
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.transform = torchvision.transforms.Compose([
            Resize(WIDTH, HEIGHT),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.DEVICE = torch.device('cpu')
        if args.CUDA:
            if torch.cuda.is_available():
                self.DEVICE = torch.device('cuda:0')
                self.model.cuda()

        self.Predict = self._Predict1
        if args.BATCHSIZE > 1:
            self.Predict = self._Predict2
    
    def Transform(self, input):
        return self.transform(input)
    
    def Predict(self, input):
        pass
    
    def _Predict1(self, input):
        input = self.Transform(input)
        input = input.unsqueeze(0) # C, W, H -> B, C, W, H
        return self.model(input.to(self.DEVICE))['out'].argmax(1).byte().cpu().numpy() # B, C, W, H -> B, W, H
    
    def _Predict2(self, input):
        input = torch.cat(list(map(lambda x: self.Transform(x).unsqueeze(0), input)))
        return self.model(input.to(self.DEVICE))['out'].argmax(1).byte().cpu().numpy() # B, C, W, H -> B, W, H
    

class CleanBackModel(Base):
    def __init__(self, args, WIDTH: int=320, HEIGHT=320) -> None:
        super().__init__(args, WIDTH, HEIGHT)

        self.Plugin = self._Masking1
        try:
            if args.BATCHSIZE > 1:
                self.Plugin = self._Masking2
        except:
            pass

    def Plugin(self, input):
        pass
    
    def _Masking1(self, input):
        Width, Height, _ = input.shape
        output = self.Predict(input)
        mask = (np.where(output > 0., 1., 0.) * 255.).astype(np.uint8)
        mask = np.tile(mask, (3, 1, 1)).transpose((1, 2, 0))
        mask = cv2.resize(mask, (Width, Height))
        output = cv2.bitwise_and(input, mask)
        return output


    def _Masking2(self, input):
        Width, Height, _ = input[0].shape
        output = self.Predict(input)
        mask = (np.where(output > 0., 1., 0.) * 255.).astype(np.uint8)
        mask = np.tile(mask, (1, 3, 1, 1)).transpose((0, 2, 3, 1))
        mask = map(lambda x: cv2.resize(x, (Width, Height)), mask)
        output = map(lambda x: cv2.bitwise_and(input, x), mask)
        return list(output)

class VideoFileCleanBack(object):
    def __init__(self, args: MakedataArgs) -> None:
        self.DomainType = None
        if args.DOMAINA:
            self.DomainType = 'DomainA'
        if args.DOMAINB:
            self.DomainType = 'DomainB'
        self.videoFiles = sorted(glob.glob(os.path.join(args.INPUT, '*')))
        self.CBM = CleanBackModel(args)
        
        self.Processing(args)

    def Processing(self, args):
        for file in self.videoFiles:
            filename = file.split('/')[-1]
            path = os.path.join(args.OUTPUT, self.DomainType, filename)
            if not os.path.isdir(path):
                os.makedirs(path, exist_ok=True)
            VFP = VideoFileProcessing(file, args)
            batch = []
            count = 0
            for buffer in VFP.Getbuffer():
                for frame in buffer:
                    if len(batch) == args.BATCHSIZE:
                        output = self.CBM.Plugin(batch)
                        for out in output:
                            cv2.imwrite(os.path.join(path, f'{count:0=10d}.png'))
                            count += 1
                        batch.clear()

class RealtimeBackClean(object):
    def __init__(self, args: PluginArgs) -> None:
        self.Capture = CaptureBase(args.VIDEO)
        self.CBM = CleanBackModel(args)
    
    def Plugin(self):
        return self.CBM.Plugin(self.Capture.Read())

if __name__ == '__main__':
    pass