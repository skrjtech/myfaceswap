import os
import cv2
import numpy as np
import torch
import torchvision
from typing import List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.cuda()
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(320, torchvision.transforms.InterpolationMode.BICUBIC),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
batchSize = 16
for videopath in ['', '']:
    dirPath, fileName = videopath.rsplit('/', 1)
    fileName, ext = os.path.splitext(fileName)

    videoCap = cv2.VideoCapture(videopath)
    WIDTH = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bufferFrame = []
    while (videoCap.isOpened()):
        ret, frame = videoCap.read()
        if ret:
            bufferFrame.append(frame.copy())
            if len(bufferFrame) % batchSize == 0:
                pass

    if len(bufferFrame) > 0:
        pass