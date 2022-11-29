import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import cv2
import numpy as np
import torch
import torchvision
if 'google_colab' in sys.modules:
    print('google_colab')
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
from myfaceswap.types import (
    List
)

class Base:
    def __init__(
            self,
            rootDir: str,
            domains: List[str],
            bathSize: int=1,
            limit: int=-1,
            gpu: bool=False
    ):
        """

        :param rootDir:     'ioRoot/
        :param domains:     '[domain1, domain2, ..., domainN]'
        :param bathSize:    '-1'
        :param limit:       'frame 1, 2, ..., N
        """
        self.rootDir = rootDir
        self.domains = domains
        self.batchSize = bathSize
        self.limit = limit
        self.gpu = gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() & self.gpu else 'cpu')

        os.makedirs(rootDir, exist_ok=True)
        for file in self.domains:
            if not os.path.isdir(file):
                assert ValueError(f'{file} が存在しません.')

    def video2frame(self):
        for videopath in self.domains:
            domainName = videopath.rsplit('/', 1)[-1].split('.')[0]
            videoCap = cv2.VideoCapture(videopath)
            MAXFRAME = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
            MAXFRAME = MAXFRAME if self.limit < 0 else self.limit
            COUNT = 0
            with tqdm(total=MAXFRAME, unit=' batch') as prev:
                while (videoCap.isOpened()):
                    if COUNT == self.limit: break
                    ret, frame = videoCap.read()
                    if not ret: break
                    path = os.path.join(self.rootDir, 'Datasets', domainName, 'video/head0')
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(path, f'{COUNT:0=5}.png'), frame)
                    prev.update(1)
                    COUNT += 1
            videoCap.release()

class Video2Frame(Base):
    def __init__(self, **kwargs):
        super(Video2Frame, self).__init__(**kwargs)

class CleanBack(Base):
    def __init__(self, **kwargs):
        super(CleanBack, self).__init__(**kwargs)
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        model = model.to(self.device)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.InpTensor = lambda x: self.transforms(x).unsqueeze(0).to(self.device) # Output Shape: batch, C, W, H
        self.model = lambda x: model(x)['out'].argmax(1).byte().cpu().numpy() # Output Shape: batch, W, H

    def video2frame(self):
        for videopath in self.domains:
            domainName = videopath.rsplit('/', 1)[-1].split('.')[0]
            videoCap = cv2.VideoCapture(videopath)
            WIDTH = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
            HEIGHT = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            MAXFRAME = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
            bufferFrame = []
            self.saveCount = 0
            COUNT = 0
            MAXFRAME = MAXFRAME if self.limit < 0 else self.limit
            with tqdm(total=MAXFRAME, unit=' batch') as prev:
                while (videoCap.isOpened()):
                    if COUNT == self.limit: break
                    ret, frame = videoCap.read()
                    if not ret: break
                    bufferFrame.append(frame.copy())
                    if len(bufferFrame) % self.batchSize == 0:
                        self.Masking(bufferFrame, WIDTH, HEIGHT, domainName)
                        bufferFrame.clear()
                    prev.update(1)
                    COUNT += 1
            if len(bufferFrame) > 0:
                self.Masking(bufferFrame, WIDTH, HEIGHT, domainName)
                bufferFrame.clear()

    def Masking(self, inp: list, width, height, path):
        resize = map(lambda x: cv2.resize(x, (320, 320)), inp)
        inps = torch.cat(tuple(map(self.InpTensor, resize)), dim=0)
        mask = (np.where(self.model(inps) > 0, 1, 0) * 255.).astype(np.uint8)
        maskTile = map(lambda x:np.tile(x, (3, 1, 1)).transpose((1, 2, 0)), mask) # Batch, Width, Height -> Batch, Channel, 320, 320 -> Batch, 320, 320, Channel
        maskResize = map(lambda x: cv2.resize(x, (width, height)), maskTile) # Batch, Width, Height, Channel
        output = list(map(lambda x: cv2.bitwise_and(*x), zip(inp, maskResize)))
        self.Save(output, path)

    def Save(self, data: List[np.asarray], path):
        path = os.path.join(self.rootDir, 'Datasets', path, 'video', 'head0')
        os.makedirs(path, exist_ok=True)
        for frame in data:
            _path = os.path.join(path, f'{self.saveCount:0=5}.png')
            cv2.imwrite(_path, frame)
            self.saveCount += 1

if __name__ == '__main__':
    # CleanBack('/ws/ioRoot', ['/ws/output.mp4'], 4, gpu=True).video2frame()
    Video2Frame(rootDir='/ws/ioRoot', domains=['/ws/output.mp4']).video2frame()