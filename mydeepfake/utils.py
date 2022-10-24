import os
import sys
import cv2
import glob
import torch
import random
import numpy as np
import torchvision
from PIL import Image
import typing
from tqdm import tqdm

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

def BackgroundClear(srcPath: str, tagPath: str):
    assert not os.path.isdir(srcPath), f'{srcPath} Directoryが存在しない...'
    if not os.path.isdir(tagPath):
        os.mkdir(tagPath)
    framePaths = sorted(glob.glob(srcPath))
    for path in framePaths:
        img = cv2.imread(path)
        
        

def tensor2image(tensor: torch.tensor) -> np.asarray:
    image = 127.5 * (tensor[0].cpu().float().detach().numpy() + 1.0)
    if image.shape[0] == 1: image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)

def tensor2image_ver2(tensor: torch.tensor) -> np.asarray:
    image = 127.5 * (tensor[0].cpu().float().detach().numpy() + 1.0)
    if image.shape[0] == 1: image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8).transpose(1, 2, 0)

def save_img(img_real: np.asarray, img_fake: np.asarray, img_rec: np.asarray, save2path: str):
    Image.fromarray(np.concatenate((img_real, img_fake, img_rec), axis=1)).save(save2path)

def save_saveral_img(imgs: tuple, save2path: str):
    Image.fromarray(np.concatenate(imgs, axis=1)).save(save2path)

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