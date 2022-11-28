import os
import glob
import random
import numpy as np
from torch.utils import data as torchUtilsData
from torchvision import transforms
from PIL import Image
from typing import Optional, List
OptList = Optional[List]
from collections import OrderedDict
class FaceDatasetSquence(torchUtilsData.Dataset):
    def __init__(
            self,
            source: str,
            target: str,
            transform: OptList=None,
            unaligned: bool=False,
            skip: int=0
    ):
        """
        (input) :param source (str): /ioRoot/datasets/domainA/videos
        (input) :param target (str): /ioRoot/datasets/domainA/videos
        (input) :param transform (list): [transform1, transform2, ..., transformN]
        (input) :param unaligned (bool): False
        (input) :param skip (int): 0
        """
        self.domainA = source
        self.domainB = target
        self.transform = transforms.Compose(transform) if transform is not None else transform
        self.unaligned = unaligned
        self.skip = skip
        self.remove_num = (skip + 1) * 2

        framesA, framesB = [], []
        dirA = sorted(os.listdir(source))
        dirB = sorted(os.listdir(target))
        for dir in dirA:
            framesA += sorted(glob.glob(os.path.join(source, dir, '*')))[:-self.remove_num]
        for dri in dirB:
            framesB += sorted(glob.glob(os.path.join(target, dri, '*')))[:-self.remove_num]
        domainALen, domainBLen = len(framesA), len(framesB)
        print(f'Frame Max (source: {domainALen}) (target: {domainBLen})')
        self.framesA, self.framesB = framesA, framesB
        self.domainALen, self.domainBLen = domainALen, domainBLen
        self.count = 1234

    def GetSquentialData(self, file1, seed):
        file, exe = os.path.splitext(file1)
        dir, file = file.rsplit('/', 1)
        file2 = os.path.join(dir, f'{int(file) + self.skip:0=5}{exe}')
        file3 = os.path.join(dir, f'{int(file) + self.skip * 2:0=5}{exe}')
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
        seed = self.count
        filesA = self.framesA[item % self.domainALen]
        A1, A2, A3 = self.GetSquentialData(filesA, seed)

        filesB = self.framesB[np.random.randint(0, self.domainBLen - 1)] if self.unaligned else self.framesB[item % self.domainBLen]
        B1, B2, B3 = self.GetSquentialData(filesB, seed + 1)
        self.count += 1
        return OrderedDict({'A1': A1, 'A2': A2, 'A3': A3, 'B1': B1, 'B2': B2, 'B3': B3})

    def __len__(self):
        return max(self.domainALen, self.domainALen)

class FaceDatasetVideo(torchUtilsData.Dataset):
    def __init__(
            self,
            file: str,
            transform: OptList=None,
            unaligned: bool=False,
            skip: int=0
    ):
        """
        (input) :param file (str): ioRoot/datasets/domainX/video/head
        (input) :param transform (list): [transform1, transform2, ..., transformN]
        (input) :param unaligned (bool): False
        (input) :param skip (int): 0
        """
        self.path = file
        self.transform = transforms.Compose(transform) if transform is not None else transform
        self.unaligned = unaligned
        self.skip = skip
        self.remove_num = (skip + 1) * 2
        self.files = sorted(glob.glob(os.path.join(file, '*')))[:-self.remove_num]
        self.filesNum = len(self.files)

    def GetSquentialData(self, file1, seed):
        file, exe = os.path.splitext(file1)
        dir, file = file.rsplit('/', 1)
        file2 = os.path.join(dir, f'{int(file) + self.skip:0=5}{exe}')
        file3 = os.path.join(dir, f'{int(file) + self.skip * 2:0=5}{exe}')
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
        file = self.files[item % self.filesNum]
        seed = 123
        item1, item2, item3 = self.GetSquentialData(file, seed)
        return OrderedDict({'1': item1, '2': item2, '3': item3})

    def __len__(self):
        return self.filesNum

if __name__ == '__main__':

    # dataset = FaceDatasetSquence("/ws/ioRoot/Datasets/output1/video", "/ws/ioRoot/Datasets/output2/video", skip=2)
    # print(dataset[0].values())
    # dataset = FaceDatasetVideo("/ws/ioRoot/Datasets/output1/video/head0", skip=2)
    # print(dataset[0].values())

    pass