import os
import glob
import random
from typing import Optional, List
import torch
import torch.utils
import torch.utils.data

import torchvision
import torchvision.transforms
from PIL import Image

class FaceDatasetSquence(torch.utils.data.Dataset):
    def __init__(
        self, 
        path: str, 
        transforms: Optional[List]=None, 
        unaligned: bool=False, 
        mode: str='train', 
        domainA: str='fadg0/video/', 
        domainB: str='faks0/video', 
        skip: int=0) -> None:
        """
        :param path: str
        :param transforms: list
        :param unaligned: bool
        :param mode: str
        :param domainA: str
        :param domainB: str
        :param skip: int
        """
        super(FaceDatasetSquence, self).__init__()
        self.transform = torchvision.transforms.Compose(transforms)
        self.unaligned = unaligned
        self.mode = mode
        self.skip = skip
        directoryAB = list(map(lambda x: sorted(os.listdir(os.path.join(path, "dataset", x))), [domainA, domainB]))
        self.filterListA = []
        self.filterListB = []
        self.remove_num = remove_num = (skip + 1) * 2
        def dirTenkai(directory, domain):
            res = []
            for dir in directory:
                A = sorted(glob.glob(os.path.join(path, domain, dir + '/*')))
                res += A[: -remove_num]
            return res
        self.filterListAB = list(map(lambda x: dirTenkai(*x), [[directoryAB[0], domainA], [directoryAB[1], domainB]]))
        self.filterListANum = len(self.filterListAB[0])
        self.filterListBNum = len(self.filterListAB[1])
        self.count = 1234
    def __getitem__(self, idx):
        seed = self.count
        def SequenceData(file1, seed=seed):
            def Transform(x):
                random.seed(seed)
                return self.transform(Image.open(x).convert('RGB'))
            dirName, fileNum = file1.rsplit('/', 1)
            fileNum = fileNum.split('.')[0]
            file2 = os.path.join(dirName, '{:0=5}.png'.format(int(fileNum) + self.skip))
            file3 = os.path.join(dirName, '{:0=5}.png'.format(int(fileNum) + self.skip * 2))
            return tuple(map(Transform, [file1, file2, file3]))
        fileA1 = self.filterListAB[0][idx % self.filterListANum]
        itemA1, itemA2, itemA3 = SequenceData(fileA1)
        if self.unaligned: fileB1 = self.filterListB[random.randint(0, self.filterListBNum - 1)]
        else: fileB1 = self.filterListAB[1][idx % self.filterListBNum]
        itemB1, itemB2, itemB3 = SequenceData(fileB1, seed=seed+1)
        self.count += 1
        return {
            "A1": itemA1, "A2": itemA2, "A3": itemA3,
            "B1": itemB1, "B2": itemB2, "B3": itemB3
        }
    def __len__(self):
        return max(self.filterListANum, self.filterListBNum)

class FaceDatasetVideo(torch.utils.data.Dataset):
    def __init__(self, root, transforms: Optional[List]=None, unaligned=False, mode='train', 
                 video='dataset/videoframe/domain_a', skip=2):
        self.skip = skip
        self.remove_num = (skip + 1) * 2
        self.transform = torchvision.transforms.Compose(transforms)
        self.unaligned = unaligned
        all_files = sorted(glob.glob(os.path.join(root, video) + '/*'))
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
        
    def __len__(self):
        return len(self.files)

if __name__ == "__main__":
    dataset = FaceDatasetSquence("/ws/ioRoot", unaligned=True, domainA="domainA", domainB="domainB", skip=2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    batch = next(iter(dataloader))
    print(batch)