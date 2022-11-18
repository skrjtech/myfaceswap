import os
import cv2
import numpy as np
import torch
import torchvision
from typing import List

import myfaceswap.utils
if myfaceswap.utils.is_env_notebook():
     from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class Video2FrameAndCleanBack(object):
    def __init__(
            self,
            rootDir: str,
            domains: List[str],
            batchSize: int=1,
            gpu: bool=False
    ):

        self.rootDir = rootDir
        os.makedirs(rootDir, exist_ok=True)
        self.domains = domains
        self.batchSize = batchSize
        self.gpu = gpu
        # Check files
        for file in self.domains:
            if not os.path.isdir(file):
                assert ValueError(f'{file} が存在しません.')

        self.device = torch.device('cuda' if torch.cuda.is_available() & gpu else 'cpu')
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        if gpu:
            model.cuda()

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(320, torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.InpTensor = lambda x: self.transforms(x).unsqueeze(0).to(self.device) # Output Shape: batch, C, W, H
        self.model = lambda x: model(x)['out'].argmax(1).byte().cpu().numpy() # Output Shape: batch, W, H

    def run(self):
        for videopath in self.domains:
            domainName = videopath.rsplit('/', 1)[-1].split('.')[0]

            videoCap = cv2.VideoCapture(videopath)
            WIDTH = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
            HEIGHT = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            MAXFRAME = videoCap.get(cv2.CAP_PROP_FRAME_COUNT)
            bufferFrame = []
            self.saveCount = 0
            with tqdm(total=MAXFRAME, unit=' batch') as prev:
                while (videoCap.isOpened()):
                    ret, frame = videoCap.read()
                    if not ret: break
                    bufferFrame.append(frame.copy())
                    if len(bufferFrame) % self.batchSize == 0:
                        self.Masking(bufferFrame, WIDTH, HEIGHT, domainName)
                        bufferFrame.clear()
                    prev.update(1)
            if len(bufferFrame) > 0:
                self.Masking(bufferFrame, WIDTH, HEIGHT, domainName)

    def Masking(self, inp: list, width, height, path):
        inps = torch.cat(tuple(map(self.InpTensor, inp)), dim=0)
        mask = np.where(self.model(inps) > 0, 255, 0)
        mask = list(map(lambda x: cv2.resize(x.astype(np.uint8), (width, height)), mask))
        mask = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB), mask))
        output = list(map(lambda x: cv2.bitwise_and(*x), zip(inp, mask)))
        self.Save(output, path)

    def Save(self, data: List[np.asarray], path):
        path = os.path.join(self.rootDir, 'Datasets', path, 'video', 'head0')
        os.makedirs(path, exist_ok=True)
        for frame in data:
            path = os.path.join(path, f'{self.saveCount:0=5}.png')
            cv2.imwrite(path, frame)
            self.saveCount += 1

if __name__ == '__main__':
    Video2FrameAndCleanBack('/ws/ioRoot', ['/ws/base1.mp4', '/ws/base2.mp4'], 4, gpu=True).run()