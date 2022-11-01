import os

import cv2
import numpy as np

import torch
import torchvision

from tqdm import tqdm

class Video2FramesAndCleanBack(object):
    def __init__(self, io_root: str, domainA: str, domainB: str, batchSize: int, gpu: bool, verbose: bool, limit: int):
        
        self.rootDir = io_root
        self.domainA = domainA
        self.domainB = domainB
        self.batchSize = batchSize
        self.verbose = verbose
        self.limit = limit

        self.targetPathDomainA = os.path.join(io_root, 'dataset', 'domainA', 'video', 'head0')
        os.makedirs(self.targetPathDomainA, exist_ok=True)
        self.targetPathDomainB = os.path.join(io_root, 'dataset', 'domainB', 'video', 'head0')
        os.makedirs(self.targetPathDomainB, exist_ok=True)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        model = model.to(device)
        model.eval()
        preprocessImage = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(320, torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.Tensor = lambda x: preprocessImage(x).unsqueeze(0).to(device)
        self.model = lambda x: np.concatenate([ output.argmax(0).unsqueeze(0).byte().cpu().numpy() for output in model(x)['out']])

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
            Range = list(range(batchSize))
            if self.limit > 0:
                Range = Range[:self.limit]

            if self.verbose:
                for idx in tqdm(Range):
                    yield idx
            else:
                for idx in Range:
                    yield idx
        
        if videoCap.isOpened():
            # for idx in range(batchSize):
            for idx in verbose():
                TensroFrame = []
                TensorBatch = []
                while len(TensorBatch) != self.batchSize:
                    ret, frame = videoCap.read()
                    if ret:
                        TensroFrame.append(frame.copy())
                        TensorBatch.append(self.Tensor(frame.copy()))
                with torch.no_grad():
                    outTensor = self.model(torch.cat(TensorBatch))
                Output = self._masking(TensroFrame, outTensor, height, width)
                for i in range(self.batchSize):
                    cv2.imwrite(targetPath.format((idx * self.batchSize) + i), Output[i])
        videoCap.release()
    
    def _masking(self, frames, masks, height, width):
        Output = []
        masks = np.where(masks > 0, 255, 0)
        for idx in range(self.batchSize):
            frame = frames[idx]
            mask = masks[idx]
            mask = cv2.resize(mask.astype(np.uint8), (width, height))
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            Output.append(
                cv2.bitwise_and(frame, mask)
            )
        return Output
    
    def Imshow(self, name, frames):
        for frame in frames:
            cv2.imshow(name, frame)
            if cv2.waitKey(1) == ord('q'):
                break