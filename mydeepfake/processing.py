import os

import cv2
import numpy as np

import torch
import torchvision

from tqdm import tqdm

class Video2FramesAndCleanBack(object):
    def __init__(self, io_root: str, domainA: str, domainB: str, batchSize: int, gpu: bool, verbose: bool):
        
        self.rootDir = io_root
        self.domainA = domainA
        self.domainB = domainB
        self.batchSize = batchSize
        self.verbose = verbose

        self.targetPathDomainA = os.path.join(io_root, 'dataset', 'frames', 'domainA', 'head0')
        os.makedirs(self.targetPathDomainA)
        self.targetPathDomainB = os.path.join(io_root, 'dataset', 'frames', 'domainB', 'head0')
        os.makedirs(self.targetPathDomainB)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        model = model.to(device).eval()
        preprocessImage = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(320, torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.Tensor = lambda x: preprocessImage(x).unsqueeze(0).to(device)
        self.model = lambda x: model(x)['out'][0].argmax(0).byte().cpu().numpy()
    
    def __call__(self):
        targetPathDomainA = os.path.join(self.targetPathDomainA, '{:0=5}.png')
        targetPathDomainB = os.path.join(self.targetPathDomainB, '{:0=5}.png')
        self._read(self.domainA, targetPathDomainA)
        self._read(self.domainB, targetPathDomainB)
    
    def _read(self, sourcePath, targetPath):
        videoCap = cv2.VideoCapture(sourcePath)
        width = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        videoMaxFrame = videoCap.get(cv2.CAP_PROP_FRAME_COUNT)
        batchSize = int(videoMaxFrame // self.batchSize)

        def verbose():
            if self.verbose:
                for idx in tqdm(range(batchSize)):
                    yield idx
            else:
                for idx in range(batchSize):
                    yield idx
        
        if videoCap.isOpened():
            # for idx in range(batchSize):
            for idx in verbose():
                TensroFrame = []
                TensorBatch = []
                while len(TensorBatch) != self.batchSize:
                    ret, frame = videoCap.read()
                    if ret:
                        TensroFrame.append(frame)
                        TensorBatch.append(self.Tensor(frame))
                outTensor = self.model(torch.cat(TensorBatch))
                Output = self._masking(TensroFrame, outTensor, height, width)
                for i in range(self.batchSize):
                    cv2.imwrite(targetPath.format((idx * batchSize) + i), Output[i])
    
    def _masking(self, frames, masks, height, width):
        Output = []
        masks = np.where(masks > 0, 255, 0).astype(np.uint8)
        for idx in range(self.batchSize):
            frame = frames[idx]
            mask = cv2.resize(masks[idx], (height, width)).reshape(width, height, 1)
            mask = np.concatenate([mask, mask, mask], axis=-1)
            Output.append(
                cv2.bitwise_and(frame, mask)
            )
        return Output