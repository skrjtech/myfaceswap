import os
import glob
import random
import numpy as np
from torch.utils import data as torchUtilsData
from torchvision import transforms
from PIL import Image
from typing import Optional, List
OptList = Optional[List]

class FaceDatasetSquence(torchUtilsData.Dataset):
    def __init__(
            self,
            domainA: str,
            domainB: str,
            transform: OptList=None,
            unaligned: bool=False,
            skip: int=0
    ):
        """
        (input) :param domainA (str): /ioRoot/datasets/domainA/videos
        (input) :param domainB (str): /ioRoot/datasets/domainA/videos
        (input) :param transform (list): [transform1, transform2, ..., transformN]
        (input) :param unaligned (bool): False
        (input) :param skip (int): 0
        """
        self.domainA = domainA
        self.domainB = domainB
        self.transform = transforms.Compose(transform)
        self.unaligned = unaligned
        self.skip = skip
        self.remove_num = (skip + 1) * 2
        directoryA, directoryB = list(map(os.listdir, [domainA, domainB]))
        filesA, filesB = [], []
        for dir in directoryA:
            filesA += sorted(glob.glob(os.path.join(domainA, dir + '/*')))[:- self.remove_num]
        for dir in directoryB:
            filesB += sorted(glob.glob(os.path.join(domainB, dir + '/*')))[:- self.remove_num]
        self.count = 1234
        self.filesA = filesA
        self.filesB = filesB
        self.filesAnum = len(filesA)
        self.filesBnum = len(filesB)
        print(f'Size DomainA {self.filesAnum:0=5} | DomainB {self.filesBnum:0=5}')

    def GetSquentialData(self, file1, seed):
        file, exe = os.path.splitext(file1)
        dir, file = file.rsplit('/', 1)
        file2 = os.path.join(dir, f'{int(file) + self.skip:0=5}' + exe)
        file3 = os.path.join(dir, f'{int(file) + self.skip * 2:0=5}' + exe)
        random.seed(seed)
        file1 = self.transform(Image.open(file1).convert('RGB'))
        random.seed(seed)
        file2 = self.transform(Image.open(file2).convert('RGB'))
        random.seed(seed)
        file3 = self.transform(Image.open(file3).convert('RGB'))
        return file1, file2, file3

    def __getitem__(self, item):
        seed = self.count
        filesA = self.filesA[item % self.filesAnum]
        filesB = self.filesB[np.random.randint(0, self.filesBnum - 1)] if self.unaligned else self.filesB[item % self.filesBnum]
        A1, A2, A3 = self.GetSquentialData(filesA, seed)
        B1, B2, B3 = self.GetSquentialData(filesB, seed + 1)
        self.count += 1
        return {'A1': A1, 'A2': A2, 'A3': A3, 'B1': B1, 'B2': B2, 'B3': B3}

    def __len__(self):
        return max(self.filesAnum, self.filesBnum)

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
        self.transform = transforms.Compose(transform)
        self.unaligned = unaligned
        self.skip = skip
        self.remove_num = (skip + 1) * 2
        self.files = sorted(glob.glob(os.path.join(file, '*')))[:-self.remove_num]
        self.filesNum = len(self.files)

    def GetSquentialData(self, file1, seed):
        file, exe = os.path.splitext(file1)
        dir, file = file.rsplit('/', 1)
        file2 = os.path.join(dir, f'{int(file) + self.skip:0=5}' + exe)
        file3 = os.path.join(dir, f'{int(file) + self.skip * 2:0=5}' + exe)
        random.seed(seed)
        file1 = self.transform(Image.open(file1).convert('RGB'))
        random.seed(seed)
        file2 = self.transform(Image.open(file2).convert('RGB'))
        random.seed(seed)
        file3 = self.transform(Image.open(file3).convert('RGB'))
        return file1, file2, file3

    def __getitem__(self, item):
        file = self.files[item % self.filesNum]
        seed = 123
        item1, item2, item3 = self.GetSquentialData(file, seed)
        return {'1': item1, '2': item2, '3': item3}

    def __len__(self):
        return self.filesNum

if __name__ == '__main__':

    dataset = FaceDatasetSquence("/ws/ioRoot/dataset/domainA/video", "/ws/ioRoot/dataset/domainB/video", skip=2)
    print(dataset[0])
    dataset = FaceDatasetVideo("/ws/ioRoot/dataset/domainA/video/head0", skip=2)
    print(dataset[0])