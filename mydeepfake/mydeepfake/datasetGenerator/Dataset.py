import os
import glob
import random

import torch
import torchvision
from PIL import Image

class FaceDatasetSquence(torch.utils.data.Dataset):
    def __init__(self, path: str, transforms: torchvision.transforms=None, unaligned: bool=False, mode: str='train', filterA: str='fadg0/video/', filterB: str='faks0/video', skip: int=0) -> None:
        super().__init__()
        self.transform = torchvision.transforms.Compose(transforms)
        self.unaligned = unaligned
        self.mode = mode
        self.skip = skip
        directoryA = os.listdir(os.path.join(path, filterA))
        directoryB = os.listdir(os.path.join(path, filterB))
        self.filterListA = []
        self.filterListB = []
        self.remove_num = remove_num = (skip + 1) * 2
        for dirA, dirB, in zip(directoryA, directoryB):
            _A = sorted(glob.glob(os.path.join(path, filterA, dirA + '/*')))
            _B = sorted(glob.glob(os.path.join(path, filterB, dirB + '/*')))
            self.filterListA += _A[: -remove_num]
            self.filterListB += _B[: -remove_num]
        self.filterListANum = len(self.filterListA)
        self.filterListBNum = len(self.filterListB)
        
        self.count = 1234
    def __getitem__(self, idx):
        seed = self.count
        def SequenceData(file1, seed=seed):
            def Transform(x):
                random.seed(seed)
                return self.transform(Image.open(x).convert('RGB'))
            dirName, fileNum = file1.rsplit('/', 1)
            file2 = os.path.join(dirName, '{:0=3}'.format(int(fileNum) + self.skip))
            file3 = os.path.join(dirName, '{:0=3}'.format(int(fileNum) + self.skip * 2))
            return Transform(file1), Transform(file2), Transform(file3)
            
        fileA1 = self.filterListA[idx % self.filterListANum]
        itemA1, itemA2, itemA3 = SequenceData(fileA1)
        if self.unaligned: fileB1 = self.filterListB[random.randint(0, self.filterListBNum - 1)]
        else: fileB1 = self.filterListB[idx * self.filterListBNum]
        itemB1, itemB2, itemB3 = SequenceData(fileB1, seed=seed+1)
        self.count += 1
        return {
            "A1": itemA1, "A2": itemA2, "A3": itemA3,
            "AB": itemA1, "B2": itemB2, "B3": itemB3
        }
    
    def __len__(self):
        return max(self.filterListANum, self.filterListBNum)