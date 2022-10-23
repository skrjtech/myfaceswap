import os
import glob
import random

import torch
import torchvision
from PIL import Image

class FaceDatasetSquence(torch.utils.data.Dataset):
    def __init__(self, path: str, transforms: torchvision.transforms=None, unaligned: bool=False, mode: str='train', domainA: str='fadg0/video/', domainB: str='faks0/video', skip: int=0) -> None:
        super().__init__()
        self.transform = torchvision.transforms.Compose(transforms)
        self.unaligned = unaligned
        self.mode = mode
        self.skip = skip
        directoryA = os.listdir(os.path.join(path, domainA))
        directoryB = os.listdir(os.path.join(path, domainB))
        self.filterListA = []
        self.filterListB = []
        self.remove_num = remove_num = (skip + 1) * 2
        for dirA, dirB, in zip(directoryA, directoryB):
            _A = sorted(glob.glob(os.path.join(path, domainA, dirA + '/*')))
            _B = sorted(glob.glob(os.path.join(path, domainB, dirB + '/*')))
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
            file2 = os.path.join(dirName, '{:0=5}'.format(int(fileNum) + self.skip))
            file3 = os.path.join(dirName, '{:0=5}'.format(int(fileNum) + self.skip * 2))
            return Transform(file1), Transform(file2), Transform(file3)
            
        fileA1 = self.filterListA[idx % self.filterListANum]
        itemA1, itemA2, itemA3 = SequenceData(fileA1)
        if self.unaligned: fileB1 = self.filterListB[random.randint(0, self.filterListBNum - 1)]
        else: fileB1 = self.filterListB[idx % self.filterListBNum]
        itemB1, itemB2, itemB3 = SequenceData(fileB1, seed=seed+1)
        self.count += 1
        return {
            "A1": itemA1, "A2": itemA2, "A3": itemA3,
            "B1": itemB1, "B2": itemB2, "B3": itemB3
        }
    
    def __len__(self):
        return max(self.filterListANum, self.filterListBNum)

class FaceDatasetVideo(Dataset):
    def __init__(self, root, transforms=None, unaligned=False, mode='train', 
                 video='dataset/videoframe/domain_a', skip=2):
        self.skip = skip
        self.remove_num = (skip + 1) * 2
        self.transform = torchvision.transforms.Compose(transforms)
        self.unaligned = unaligned
        all_files = sorted(glob.glob(os.path.join(root, files) + '/*'))
        self.files = all_files[:-self.remove_num]

    def __getitem__(self, index):
        file = self.files[index % len(self.files)]
        seed = 1234
        item_1, item_2, item_3 = self.get_sequential_data(file, seed)
        return {'1': item_1, '2': item_2, '3' : item_3}

    def get_sequential_data(self, file1, seed):
        dir_name, file_num = file1.rsplit('/', 1)
        file2 = os.path.join(dir_name, '{:0=5}'.format(int(file_num) + self.skip))
        file3 = os.path.join(dir_name, '{:0=5}'.format(int(file_num) + self.skip * 2))
        random.seed(seed)
        item1 = self.transform(Image.open(file1).convert('RGB'))
        random.seed(seed)
        item2 = self.transform(Image.open(file2).convert('RGB'))
        random.seed(seed)
        item3 = self.transform(Image.open(file3).convert('RGB'))
        return item1, item2, item3
        
    def __len__(self):s
        return len(self.files)