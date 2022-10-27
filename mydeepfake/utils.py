import os
import sys
import cv2
import glob
import torch
import random
import numpy as np
import torchvision
import typing
from tqdm import tqdm
from typing import List, Sequence, Tuple, Any, Optional

def MasProccess(mask: np.ndarray, org: np.ndarray, size: tuple):
    W, H = size
    # mask[mask > 0] = 255
    # mask[mask != 255] = 0
    mask = np.where(mask > 0, 255, 0)
    mask = cv2.resize(mask.astype(np.uint8), (H, W)).reshape(W, H, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    dist = cv2.bitwise_and(org, mask)
    return dist

def CallIndexMattingModel(limit: int=-1) -> typing.Any:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model_ = model_.to(device)
    model_.eval()
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(320, torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = lambda x: preprocess(x).unsqueeze(0).to(device)
    model = lambda x: model_(tensor(x))['out'][0].argmax(0).byte().cpu().numpy()
    def run(frames: typing.List[np.ndarray]) -> typing.List[np.ndarray]:
        ResultFrame = []
        newFrameAppend = ResultFrame.append
        with torch.no_grad():
            for frame in tqdm(frames[:limit]):
                w, h = frame.shape[:2]
                mask = model(frame)
                output = MasProccess(mask, frame, (w, h))
                newFrameAppend(output)
        return ResultFrame
    return run

def Video2Frame(path: str) -> typing.List[np.ndarray]:
    frames = []
    framesAppend = frames.append
    video = cv2.VideoCapture(path)
    frameNum = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if not video.isOpened(): sys.exit()
    for _ in range(frameNum):
        ret, frame = video.read()
        if ret:
            framesAppend(frame)
    video.release()
    return frames

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

# 過去の生成データ(50iter分)を保持しておく
class ReplayBuffer():
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            #
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.autograd.Variable(torch.cat(to_return))

class ImageProcessBackgraund(object):
    def __init__(self, rootDir: str, videoFiles: list[str], frameLimit: int=100) -> None:
        self.RootDir = rootDir
        self.CheckDir(rootDir)
        self.videoFiles = videoFiles
        self.frameLimit = frameLimit

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        model_ = model_.to(device)
        model_.eval()
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(320, torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = lambda x: preprocess(x).unsqueeze(0).to(device)
        self.model = lambda x: model_(tensor(x))['out'][0].argmax(0).byte().cpu().numpy()
    
    def __call__(self) -> Any:
        # [ ./videofilename1.video, ./videofilename2.video, ..., ./videofilenameN.video]
        for videoPath in self.videoFiles:
            self.forward(videoPath)

    def forward(self, path):
        # ./videofilename.video -> filename
        fileName = path.split('/')[-1].split('.')[0]
        # RootDir/video/videofilename
        outputPath = os.path.join(self.RootDir, 'video', fileName)
        self.CheckDir(outputPath)
        # video data read
        videoCap: cv2.VideoCapture = cv2.VideoCapture(path)
        videoMaxFrame: int = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("path: ", path, " videoCap.isOpened(): ", videoCap.isOpened())
        if videoCap.isOpened():
            headCount = 0
            # RootDir/video/videofilename/head000/000.png
            #                                       .
            # RootDir/video/videofilename/headNNN/NNN.png
            outputFormat = "head{:0=3}/{:0=5}.png"
            # 0...Frame All
            for i in range(videoMaxFrame):
                if i == self.frameLimit: break
                # video read -> boll, image
                _, frame = videoCap.read()
                # Output <- shape Width Height Channel
                Output = self.Masking(frame, self.model(frame))
                # Check
                path_ = os.path.join(outputPath, outputFormat.format(headCount, i))
                print(path_)
                check = path_.split('/')[:-1]
                self.CheckDir(os.path.join(*check))
                # Image no shutsuryoku 
                cv2.imwrite(path_, Output)
                if headCount > 0 and headCount % self.frameLimit == 0:
                    headCount += 1

    def Masking(self, frame, mask):
        """
        input: frame shape width height channel
        input: mask shape width height
        """
        # Width, Height
        W, H = frame.shape[:2]
        """
        0 0 0
        0 1 0
        0 0 0
        [ 1 -> 255 ]
        [ 0 ->  0  ]
        0  0  0
        0 255 0
        0  0  0
        """
        mask = np.where(mask > 0, 255, 0)
        # Resize org 320 320 -> target size Height Width -> Reshape Width Height 1
        mask = cv2.resize(mask.astype(np.uint8), (H, W)).reshape(W, H, 1)
        mask = np.concatenate([mask, mask, mask], axis=-1) # -> Width Height Channel
        # Gousei
        return cv2.bitwise_and(frame, mask)
    
    def CheckDir(self, path):
        check = not os.path.isdir(path)
        if check:
            os.makedirs(path)
        return check

def TestSampleMakeVideo(sec: int=5, frameRate: float=24.0):
    totalFrame = int(sec * frameRate)
    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    def Writer(path):
        videoWrite = cv2.VideoWriter(path, fourcc, frameRate, (1920, 1080))
        for t in range(totalFrame):
            img = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
            videoWrite.write(img)
            cv2.imshow("frame", img)
            if cv2.waitKey(1) == ord('q'): break
        videoWrite.release()
    Writer("output.mp4")
    Writer("output.mp4")